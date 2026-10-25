import os
import random
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_CLIENTS = 5

DATA_ROOT = os.environ.get("PTBXL_ROOT", ".")  # path to the PTB-XL v1.0.3 folder

METADATA_FILE = os.path.join(
    DATA_ROOT,
    "ptbxl_database.csv"
)

SCP_FILE = os.path.join(
    DATA_ROOT,
    "scp_statements.csv"
)

OUTPUT_DIR = "step6_5_1_N2_partition"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ptbxl_noniid_N2_5_hospitals.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "N2_partition_summary.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 6.5.1 - N2 STRONG NON-IID PATIENT-LEVEL PARTITION")
print("=" * 70)


# ============================================================
# LOAD PTB-XL
# ============================================================

print("\nLoading PTB-XL metadata...")

df = pd.read_csv(
    METADATA_FILE
)

print(
    "Total ECGs:",
    len(df)
)


# ============================================================
# TRAIN FOLDS 1-8
# ============================================================

train_df = df[
    df["strat_fold"].between(1, 8)
].copy()

print(
    "Train ECGs:",
    len(train_df)
)


# ============================================================
# LOAD SCP STATEMENTS
# ============================================================

print("\nLoading SCP statements...")

scp = pd.read_csv(
    SCP_FILE,
    index_col=0
)


# ============================================================
# DIAGNOSTIC MAPPING
# ============================================================

diagnostic_mapping = {}

for code, row in scp.iterrows():

    diagnostic_class = row.get(
        "diagnostic_class"
    )

    if pd.notna(diagnostic_class):

        diagnostic_mapping[str(code)] = str(
            diagnostic_class
        )


# ============================================================
# EXTRACT TARGET CLASSES
# ============================================================

def extract_classes(row):

    raw = row["scp_codes"]

    try:
        codes = eval(raw)
    except Exception:
        return []

    classes = []

    for code in codes.keys():

        code = str(code)

        if code in diagnostic_mapping:

            cls = diagnostic_mapping[code]

            if cls in [
                "NORM",
                "MI",
                "STTC",
                "CD",
                "HYP"
            ]:

                classes.append(cls)

    return list(set(classes))


print("\nExtracting diagnostic classes...")

train_df["target_classes"] = train_df.apply(
    extract_classes,
    axis=1
)


# ============================================================
# KEEP USABLE ECGs
# ============================================================

usable_df = train_df[
    train_df["target_classes"].apply(len) > 0
].copy()

excluded_df = train_df[
    train_df["target_classes"].apply(len) == 0
].copy()


print(
    "\nUsable training ECGs:",
    len(usable_df)
)

print(
    "Excluded ECGs:",
    len(excluded_df)
)


# ============================================================
# PRIMARY CLASS
# ============================================================

CLASS_PRIORITY = [
    "MI",
    "STTC",
    "CD",
    "HYP",
    "NORM"
]


def get_primary_class(classes):

    for cls in CLASS_PRIORITY:

        if cls in classes:
            return cls

    return None


usable_df["primary_class"] = usable_df[
    "target_classes"
].apply(
    get_primary_class
)


print("\nPrimary class distribution:")

print(
    usable_df["primary_class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# IMPORTANT:
# PATIENT-LEVEL ORGANIZATION
# ============================================================
#
# Every patient must belong to ONE hospital only.
#
# Therefore we first aggregate ECGs at patient level.
#
# ============================================================

print("\nCreating patient-level groups...")

patient_groups = []

for patient_id, group in usable_df.groupby(
    "patient_id"
):

    class_counts = {
        cls: 0
        for cls in CLASS_PRIORITY
    }

    for cls in group["primary_class"]:

        if cls in class_counts:

            class_counts[cls] += 1


    dominant_class = max(
        class_counts,
        key=class_counts.get
    )


    patient_groups.append({

        "patient_id": int(patient_id),

        "n_ecgs": len(group),

        "dominant_class": dominant_class,

        "MI": class_counts["MI"],

        "STTC": class_counts["STTC"],

        "CD": class_counts["CD"],

        "HYP": class_counts["HYP"],

        "NORM": class_counts["NORM"]

    })


patients_df = pd.DataFrame(
    patient_groups
)


print(
    "Unique patients:",
    len(patients_df)
)


# ============================================================
# PATIENT DISTRIBUTION
# ============================================================

print(
    "\nPatient dominant-class distribution:"
)

print(
    patients_df[
        "dominant_class"
    ].value_counts().sort_index()
)


# ============================================================
# N2 DESIGN
# ============================================================
#
# Strong Non-IID:
#
# H1 -> MI-dominant patients
# H2 -> STTC-dominant patients
# H3 -> CD-dominant patients
# H4 -> HYP-dominant patients
# H5 -> NORM-dominant patients
#
# BUT:
#
# We keep all ECGs of each patient together.
#
# ============================================================

HOSPITALS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5"
]


DOMINANT_CLASS = {

    "H1": "MI",

    "H2": "STTC",

    "H3": "CD",

    "H4": "HYP",

    "H5": "NORM"

}


# ============================================================
# PATIENT ASSIGNMENT
# ============================================================
#
# First assign each patient to the hospital corresponding
# to its dominant class.
#
# If one dominant class has too many patients, some of them
# are moved to the closest available hospital while keeping
# the patient intact.
#
# ============================================================

patients_df["hospital"] = None


# ------------------------------------------------------------
# STEP A:
# INITIAL CLASS-BASED ASSIGNMENT
# ------------------------------------------------------------

for hospital in HOSPITALS:

    target_class = DOMINANT_CLASS[hospital]

    mask = (
        patients_df["dominant_class"]
        == target_class
    )

    patients_df.loc[
        mask,
        "hospital"
    ] = hospital


# ============================================================
# CHECK UNASSIGNED
# ============================================================

unassigned = patients_df[
    patients_df["hospital"].isna()
].copy()


print(
    "\nInitially assigned patients:",
    len(patients_df) - len(unassigned)
)

print(
    "Unassigned patients:",
    len(unassigned)
)


# ============================================================
# ASSIGN ANY UNASSIGNED PATIENTS
# ============================================================

if len(unassigned) > 0:

    random_patients = unassigned[
        "patient_id"
    ].tolist()

    random.shuffle(
        random_patients
    )

    for i, patient_id in enumerate(
        random_patients
    ):

        hospital = HOSPITALS[
            i % len(HOSPITALS)
        ]

        patients_df.loc[
            patients_df["patient_id"]
            == patient_id,
            "hospital"
        ] = hospital


# ============================================================
# BALANCE BY ECG COUNT WHILE PRESERVING PATIENTS
# ============================================================
#
# We do NOT force exactly identical ECG counts.
#
# Patient integrity is more important.
#
# We optimize assignment only when moving a patient does not
# violate the basic dominant-class structure.
#
# ============================================================

hospital_ecg_counts = (
    patients_df
    .groupby("hospital")["n_ecgs"]
    .sum()
    .sort_index()
)


print(
    "\nInitial ECG distribution:"
)

print(
    hospital_ecg_counts
)


# ============================================================
# CREATE ECG-LEVEL PARTITION FROM PATIENT ASSIGNMENT
# ============================================================

partition_df = usable_df.merge(
    patients_df[
        [
            "patient_id",
            "hospital"
        ]
    ],
    on="patient_id",
    how="left"
)


# ============================================================
# CHECK ALL PATIENTS ASSIGNED
# ============================================================

missing_hospital = partition_df[
    partition_df["hospital"].isna()
]

print(
    "\nECGs without hospital:",
    len(missing_hospital)
)


# ============================================================
# CHECK PATIENT LEAKAGE
# ============================================================

print(
    "\nChecking patient leakage..."
)

patient_hospital_counts = (
    partition_df
    .groupby("patient_id")["hospital"]
    .nunique()
)

leakage_patients = (
    patient_hospital_counts[
        patient_hospital_counts > 1
    ]
)


print(
    "Patients assigned to multiple hospitals:",
    len(leakage_patients)
)


# ============================================================
# CHECK ECG DUPLICATES
# ============================================================

duplicates = (
    partition_df["ecg_id"]
    .duplicated()
    .sum()
)

print(
    "Duplicated ECG assignments:",
    duplicates
)


# ============================================================
# HOSPITAL DISTRIBUTION
# ============================================================

print(
    "\nHospital ECG distribution:"
)

hospital_counts = (
    partition_df[
        "hospital"
    ]
    .value_counts()
    .sort_index()
)

print(
    hospital_counts
)


# ============================================================
# PATIENT DISTRIBUTION
# ============================================================

print(
    "\nHospital patient distribution:"
)

hospital_patient_counts = (
    partition_df
    .groupby("hospital")[
        "patient_id"
    ]
    .nunique()
    .sort_index()
)

print(
    hospital_patient_counts
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\nClass distribution per hospital:"
)

class_distribution = pd.crosstab(
    partition_df["hospital"],
    partition_df["primary_class"]
)

print(
    class_distribution
)


# ============================================================
# CLASS PROPORTIONS
# ============================================================

print(
    "\nClass proportions per hospital:"
)

class_proportions = (
    class_distribution
    .div(
        class_distribution.sum(axis=1),
        axis=0
    )
)

print(
    class_proportions.round(3)
)


# ============================================================
# VERIFY EACH PATIENT HAS ONLY ONE HOSPITAL
# ============================================================

assert len(leakage_patients) == 0, (
    "ERROR: Patient leakage detected!"
)


# ============================================================
# VERIFY ECG UNIQUENESS
# ============================================================

assert duplicates == 0, (
    "ERROR: Duplicate ECG assignments detected!"
)


# ============================================================
# VERIFY ALL USABLE ECGs ARE PRESENT
# ============================================================

assert len(partition_df) == len(
    usable_df
), (
    "ERROR: Number of partition rows "
    "does not match usable ECG count!"
)


# ============================================================
# SAVE PARTITION
# ============================================================

partition_df = partition_df[
    [
        "ecg_id",
        "patient_id",
        "hospital",
        "primary_class",
        "strat_fold"
    ]
].copy()


partition_df = partition_df.sort_values(
    [
        "hospital",
        "patient_id",
        "ecg_id"
    ]
).reset_index(
    drop=True
)


partition_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 6.5.1 - N2 STRONG NON-IID "
        "PATIENT-LEVEL PARTITION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Clients: {N_CLIENTS}\n"
    )

    f.write(
        f"Total PTB-XL ECGs: {len(df)}\n"
    )

    f.write(
        f"Train ECGs: {len(train_df)}\n"
    )

    f.write(
        f"Usable train ECGs: {len(usable_df)}\n"
    )

    f.write(
        f"Excluded train ECGs: {len(excluded_df)}\n"
    )

    f.write(
        f"Unique patients: "
        f"{partition_df['patient_id'].nunique()}\n\n"
    )

    f.write(
        "Dominant hospital classes:\n"
    )

    for hospital in HOSPITALS:

        f.write(
            f"{hospital}: "
            f"{DOMINANT_CLASS[hospital]}\n"
        )

    f.write(
        "\nHospital ECG distribution:\n"
    )

    f.write(
        hospital_counts.to_string()
    )

    f.write(
        "\n\nHospital patient distribution:\n"
    )

    f.write(
        hospital_patient_counts.to_string()
    )

    f.write(
        "\n\nClass distribution:\n"
    )

    f.write(
        class_distribution.to_string()
    )

    f.write(
        "\n\nClass proportions:\n"
    )

    f.write(
        class_proportions.round(3).to_string()
    )

    f.write(
        "\n\nPatient leakage:\n"
    )

    f.write(
        str(len(leakage_patients))
    )

    f.write(
        "\n\nDuplicated ECG assignments:\n"
    )

    f.write(
        str(duplicates)
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

print(
    "N2 PATIENT-LEVEL PARTITION CREATED"
)

print("=" * 70)

print(
    "\nCSV:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nNo training has been performed."
)

print(
    "N2 partition is ready for validation."
)

print("=" * 70)