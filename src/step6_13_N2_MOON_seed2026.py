# ======================================================================
# STEP 6.13 - N2 MOON-STYLE - SEED 2026
# TRUE MULTI-LABEL
# MEMORY-SAFE + GRADIENT ACCUMULATION + CRASH-SAFE RESUME
# ======================================================================

from pathlib import Path
import ast
import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import wfdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 2026
ROUNDS = 20
LOCAL_EPOCHS = 2

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

ACCUMULATION_STEPS = (
    EFFECTIVE_BATCH_SIZE
    // MICRO_BATCH_SIZE
)

LR = 0.001

MOON_MU = 1.0
TEMPERATURE = 0.5

CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    name: index
    for index, name
    in enumerate(CLASS_NAMES)
}

CLIENTS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
]

EXPECTED_CLIENT_SIZES = {
    "H1": 4603,
    "H2": 3165,
    "H3": 1884,
    "H4": 404,
    "H5": 7028,
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RESUME_VERSION = (
    "step6_13_moon_seed2026_micro4_v1"
)


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(__file__).resolve().parent

DATABASE_FILE = (
    ROOT
    / "ptbxl_database.csv"
)

SCP_FILE = (
    ROOT
    / "scp_statements.csv"
)

PARTITION_FILE = (
    ROOT
    / "step6_5_1_N2_partition"
    / "ptbxl_noniid_N2_5_hospitals.csv"
)

OUTPUT_DIR = (
    ROOT
    / "step6_13_multiseed"
    / "N2_MOON_seed2026"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESUME_FILE = (
    OUTPUT_DIR
    / "resume_state.pt"
)

TRAINING_LOG_FILE = (
    OUTPUT_DIR
    / "N2_MOON_multilabel_training_log.csv"
)

ROUND_SUMMARY_FILE = (
    OUTPUT_DIR
    / "round_summary.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "N2_multilabel_training_manifest.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "configuration.txt"
)

COMPLETION_FILE = (
    OUTPUT_DIR
    / "TRAINING_COMPLETED.txt"
)


# ======================================================================
# NORMALIZATION
# TRAIN-FOLD STATISTICS ONLY
# ======================================================================

TRAIN_MEAN = np.array(
    [
        -0.001747,
        -0.001521,
         0.000225,
         0.001617,
        -0.000932,
        -0.000625,
         0.000178,
        -0.000952,
        -0.001565,
        -0.001356,
        -0.000800,
        -0.002424,
    ],
    dtype=np.float32,
)

TRAIN_STD = np.array(
    [
        0.160010,
        0.159230,
        0.159972,
        0.138148,
        0.138767,
        0.137978,
        0.234350,
        0.334216,
        0.328469,
        0.292764,
        0.272069,
        0.280968,
    ],
    dtype=np.float32,
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ======================================================================
# MEMORY
# ======================================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# ATOMIC SAVE
# ======================================================================

def atomic_torch_save(obj, path):

    path = Path(path)

    temp_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        obj,
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


# ======================================================================
# RNG STATE
# ======================================================================

def get_rng_state():

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(state):

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch"]
    )

    if (
        torch.cuda.is_available()
        and
        "cuda" in state
    ):

        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


# ======================================================================
# CNN
# ======================================================================

class ECGCNN(nn.Module):

    def __init__(
        self,
        num_classes=5,
    ):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                12,
                32,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                128,
                256,
                kernel_size=5,
                padding=2,
            ),
            nn.GroupNorm(
                8,
                256,
            ),
        )

        self.pool = nn.MaxPool1d(2)

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            256,
            num_classes,
        )


    def backbone(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.global_pool(x)

        representation = (
            x.squeeze(-1)
        )

        return representation


    def forward_with_representation(
        self,
        x,
    ):

        representation = (
            self.backbone(x)
        )

        logits = self.fc(
            representation
        )

        return (
            logits,
            representation,
        )


    def get_representation(
        self,
        x,
    ):

        return self.backbone(x)


    def forward(self, x):

        logits, _ = (
            self.forward_with_representation(
                x
            )
        )

        return logits


# ======================================================================
# HEADER
# ======================================================================

print("=" * 80)
print("STEP 6.13 - N2 + MOON MULTI-LABEL")
print("SEED 2026 - MEMORY SAFE")
print("=" * 80)

print(f"Seed                 : {SEED}")
print(f"Device               : {DEVICE}")
print(f"Clients              : {len(CLIENTS)}")
print(f"Rounds               : {ROUNDS}")
print(f"Local epochs         : {LOCAL_EPOCHS}")
print(f"Effective batch size : {EFFECTIVE_BATCH_SIZE}")
print(f"Micro-batch size     : {MICRO_BATCH_SIZE}")
print(f"Accumulation steps   : {ACCUMULATION_STEPS}")
print(f"Learning rate        : {LR}")
print(f"MOON mu              : {MOON_MU}")
print(f"Temperature          : {TEMPERATURE}")
print(f"Classes              : {CLASS_NAMES}")


# ======================================================================
# CNN ARCHITECTURE CHECK
# ======================================================================

temporary_model = ECGCNN(
    NUM_CLASSES
)

parameter_count = sum(
    p.numel()
    for p
    in temporary_model.parameters()
)

print(
    f"\nCNN parameters: "
    f"{parameter_count}"
)

if parameter_count != 224549:

    raise RuntimeError(
        "CNN architecture mismatch."
    )

del temporary_model

cleanup_memory()


# ======================================================================
# INPUT FILE CHECK
# ======================================================================

for name, path in {

    "ptbxl_database.csv":
        DATABASE_FILE,

    "scp_statements.csv":
        SCP_FILE,

    "N2 partition":
        PARTITION_FILE,

}.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{name} missing:\n{path}"
        )


# ======================================================================
# SAFETY AGAINST ACCIDENTAL OVERWRITE
# ======================================================================

if (
    COMPLETION_FILE.exists()
    and
    not RESUME_FILE.exists()
):

    print(
        "\nThis seed 2026 MOON run "
        "is already marked COMPLETED."
    )

    print(
        "Nothing will be overwritten."
    )

    sys.exit(0)


existing_checkpoints = list(
    OUTPUT_DIR.glob("*.pt")
)

if (
    existing_checkpoints
    and
    not RESUME_FILE.exists()
):

    print("\n" + "=" * 80)

    print(
        "PARTIAL / INCOMPATIBLE RUN DETECTED"
    )

    print("=" * 80)

    print(
        "\nDelete this directory once:"
    )

    print(
        r'Remove-Item -Recurse -Force '
        r'".\step6_13_multiseed\N2_MOON_seed2026"'
    )

    print(
        "\nThen run the script again."
    )

    sys.exit(1)


# ======================================================================
# LOAD PTB-XL
# ======================================================================

print(
    "\nLoading PTB-XL metadata..."
)

database = pd.read_csv(
    DATABASE_FILE
)

database["ecg_id"] = (
    database["ecg_id"]
    .astype(int)
)

print(
    "Database rows:",
    len(database)
)

if len(database) != 21799:

    raise RuntimeError(
        "Expected 21799 database rows."
    )


# ======================================================================
# DIAGNOSTIC MAPPING
# ======================================================================

scp = pd.read_csv(
    SCP_FILE,
    index_col=0,
)

diagnostic_mapping = {}


for code, row in scp.iterrows():

    diagnostic_class = row.get(
        "diagnostic_class"
    )

    if pd.notna(
        diagnostic_class
    ):

        diagnostic_class = (
            str(diagnostic_class)
            .strip()
        )

        if diagnostic_class in CLASS_NAMES:

            diagnostic_mapping[
                str(code)
            ] = diagnostic_class


print(
    "Diagnostic mappings:",
    len(diagnostic_mapping)
)

if len(diagnostic_mapping) != 44:

    raise RuntimeError(
        "Expected 44 mappings."
    )


# ======================================================================
# TARGET FUNCTIONS
# ======================================================================

def parse_scp_codes(raw):

    if isinstance(raw, dict):
        return raw

    if pd.isna(raw):
        return {}

    try:

        parsed = ast.literal_eval(
            str(raw)
        )

        if isinstance(
            parsed,
            dict,
        ):

            return parsed

    except Exception:
        pass

    return {}


def extract_classes(raw):

    codes = parse_scp_codes(
        raw
    )

    classes = []

    for code in codes.keys():

        code = str(code)

        if code in diagnostic_mapping:

            cls = diagnostic_mapping[
                code
            ]

            if (
                cls in CLASS_TO_INDEX
                and
                cls not in classes
            ):

                classes.append(cls)

    return classes


def make_multihot(classes):

    target = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for cls in classes:

        target[
            CLASS_TO_INDEX[cls]
        ] = 1.0

    return target


# ======================================================================
# TRUE MULTI-LABEL TARGETS
# ======================================================================

print(
    "\nBuilding TRUE multi-label targets..."
)

database[
    "target_classes"
] = database[
    "scp_codes"
].apply(
    extract_classes
)

database[
    "target"
] = database[
    "target_classes"
].apply(
    make_multihot
)


usable_database = database[
    database[
        "target_classes"
    ].apply(len) > 0
].copy()


excluded_count = (
    len(database)
    -
    len(usable_database)
)


print(
    "Usable ECGs:",
    len(usable_database)
)

print(
    "Excluded ECGs:",
    excluded_count
)


if len(usable_database) != 21388:

    raise RuntimeError(
        "Expected 21388 usable ECGs."
    )

if excluded_count != 411:

    raise RuntimeError(
        "Expected 411 excluded ECGs."
    )


# ======================================================================
# OFFICIAL SPLITS
# ======================================================================

train_database = usable_database[
    usable_database[
        "strat_fold"
    ].isin(
        range(1, 9)
    )
].copy()

val_database = usable_database[
    usable_database[
        "strat_fold"
    ] == 9
]

test_database = usable_database[
    usable_database[
        "strat_fold"
    ] == 10
]


print(
    "\nOfficial usable split:"
)

print(
    "Train:",
    len(train_database)
)

print(
    "Val  :",
    len(val_database)
)

print(
    "Test :",
    len(test_database)
)


if len(train_database) != 17084:

    raise RuntimeError(
        "Train split mismatch."
    )

if len(val_database) != 2146:

    raise RuntimeError(
        "Validation split mismatch."
    )

if len(test_database) != 2158:

    raise RuntimeError(
        "Test split mismatch."
    )


# ======================================================================
# N2 PARTITION
# ======================================================================

print(
    "\nLoading N2 partition..."
)

partition = pd.read_csv(
    PARTITION_FILE
)

partition["ecg_id"] = (
    partition["ecg_id"]
    .astype(int)
)

partition["hospital"] = (
    partition["hospital"]
    .astype(str)
    .str.strip()
)


print(
    "Partition rows:",
    len(partition)
)

if len(partition) != 17084:

    raise RuntimeError(
        "Expected 17084 partition rows."
    )


# ======================================================================
# PATIENT LEAKAGE
# ======================================================================

patient_hospital_counts = (
    partition
    .groupby(
        "patient_id"
    )["hospital"]
    .nunique()
)

patient_leakage = int(
    (
        patient_hospital_counts
        > 1
    ).sum()
)

print(
    "Patient leakage:",
    patient_leakage
)

if patient_leakage != 0:

    raise RuntimeError(
        "Patient leakage detected."
    )


# ======================================================================
# VERIFY TRAIN IDS
# ======================================================================

train_ids = set(
    train_database[
        "ecg_id"
    ].tolist()
)

partition_ids = set(
    partition[
        "ecg_id"
    ].tolist()
)

if train_ids != partition_ids:

    raise RuntimeError(
        "Partition/train ECG mismatch."
    )


# ======================================================================
# MERGE
# ======================================================================

train_data = train_database.merge(

    partition[
        [
            "ecg_id",
            "hospital",
        ]
    ],

    on="ecg_id",

    how="inner",

    validate="one_to_one",
)


if len(train_data) != 17084:

    raise RuntimeError(
        "Merged training rows mismatch."
    )


# ======================================================================
# HOSPITAL DISTRIBUTION
# ======================================================================

hospital_counts = (
    train_data[
        "hospital"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nN2 hospital distribution:"
)

print(
    hospital_counts
)


for client_name in CLIENTS:

    actual = int(
        hospital_counts.get(
            client_name,
            0,
        )
    )

    expected = (
        EXPECTED_CLIENT_SIZES[
            client_name
        ]
    )

    if actual != expected:

        raise RuntimeError(
            f"{client_name}: "
            f"expected {expected}, "
            f"found {actual}."
        )


# ======================================================================
# MULTI-LABEL DISTRIBUTION
# ======================================================================

print(
    "\nMulti-label distribution per hospital:"
)

for client_name in CLIENTS:

    subset = train_data[
        train_data[
            "hospital"
        ] == client_name
    ]

    targets = np.stack(
        subset[
            "target"
        ].to_numpy()
    )

    print(
        f"\n{client_name} | "
        f"n={len(subset)}"
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        positives = int(
            targets[
                :,
                class_index
            ].sum()
        )

        prevalence = (
            positives
            /
            len(subset)
        )

        print(
            f"  {class_name:<5}: "
            f"{positives:5d} "
            f"({prevalence:.4f})"
        )


# ======================================================================
# MANIFEST
# ======================================================================

manifest = train_data[
    [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "filename_lr",
        "hospital",
    ]
].copy()


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    manifest[
        class_name
    ] = train_data[
        "target"
    ].apply(
        lambda target:
            int(
                target[
                    class_index
                ]
            )
    )


manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ======================================================================
# LIGHTWEIGHT CLIENT STORAGE
# ======================================================================

client_data = {}


for client_name in CLIENTS:

    subset = train_data[
        train_data[
            "hospital"
        ] == client_name
    ]

    filenames = (
        subset[
            "filename_lr"
        ]
        .astype(str)
        .tolist()
    )

    targets = np.stack(
        subset[
            "target"
        ].to_numpy()
    ).astype(
        np.float32
    )

    client_data[
        client_name
    ] = {
        "filenames":
            filenames,

        "targets":
            targets,

        "size":
            len(filenames),
    }


# Free metadata RAM.

del database
del usable_database
del train_database
del val_database
del test_database
del train_data
del partition
del patient_hospital_counts
del manifest
del scp

cleanup_memory()


# ======================================================================
# DATASET
# ======================================================================

class PTBXLDataset(Dataset):

    def __init__(
        self,
        filenames,
        targets,
    ):

        self.filenames = filenames
        self.targets = targets


    def __len__(self):

        return len(
            self.filenames
        )


    def __getitem__(
        self,
        index,
    ):

        filename = (
            self.filenames[
                index
            ]
        )

        record_path = (
            ROOT
            / filename
        )

        signal, _ = wfdb.rdsamp(
            str(record_path)
        )

        signal = signal.astype(
            np.float32,
            copy=False,
        ).T

        if signal.shape != (
            12,
            1000,
        ):

            raise RuntimeError(
                f"Unexpected ECG shape "
                f"{signal.shape}"
            )

        signal = (
            signal
            -
            TRAIN_MEAN[:, None]
        ) / (
            TRAIN_STD[:, None]
        )

        # Writable arrays.
        signal = np.array(
            signal,
            dtype=np.float32,
            copy=True,
        )

        target = np.array(
            self.targets[
                index
            ],
            dtype=np.float32,
            copy=True,
        )

        return (
            torch.from_numpy(
                signal
            ),

            torch.from_numpy(
                target
            ),
        )


# ======================================================================
# STATE COPY
# ======================================================================

def state_to_cpu(model):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


# ======================================================================
# FREEZE REFERENCE MODEL
# ======================================================================

def freeze_model(model):

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    return model


# ======================================================================
# NORMALIZE ACCUMULATED GRADIENTS
#
# For each micro-batch:
#
#   numerator =
#       BCE_sum / NUM_CLASSES
#       +
#       MU * MOON_CE_sum
#
# After up to 32 samples:
#
#   grad /= number_of_samples
#
# Therefore:
#
#   BCE -> mean across samples AND classes
#   MOON -> mean across samples
# ======================================================================

def normalize_gradients(
    model,
    accumulated_samples,
):

    denominator = float(
        accumulated_samples
    )

    if denominator <= 0:

        raise RuntimeError(
            "Invalid accumulated_samples."
        )

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameter.grad.div_(
                denominator
            )


# ======================================================================
# LOCAL MOON TRAINING
# ======================================================================

def train_local_moon(
    global_state,
    client_name,
    round_number,
):

    cleanup_memory()

    data = client_data[
        client_name
    ]


    dataset = PTBXLDataset(

        filenames=data[
            "filenames"
        ],

        targets=data[
            "targets"
        ],
    )


    loader = DataLoader(

        dataset,

        batch_size=MICRO_BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=False,

        drop_last=False,
    )


    # --------------------------------------------------------------
    # CURRENT LOCAL MODEL
    # Starts from round-start global parameters.
    # --------------------------------------------------------------

    local_model = ECGCNN(
        NUM_CLASSES
    )

    local_model.load_state_dict(
        global_state,
        strict=True,
    )

    local_model.to(
        DEVICE
    )

    local_model.train()


    # --------------------------------------------------------------
    # REFERENCE MODELS FOR MOON
    # Round 1: no previous local model -> MOON loss = 0.
    # --------------------------------------------------------------

    use_moon = (
        round_number > 1
    )


    global_reference = None
    previous_reference = None


    if use_moon:

        global_reference = ECGCNN(
            NUM_CLASSES
        )

        global_reference.load_state_dict(
            global_state,
            strict=True,
        )

        global_reference.to(
            DEVICE
        )

        freeze_model(
            global_reference
        )


        previous_path = (

            OUTPUT_DIR

            / (
                f"local_"
                f"{client_name}_"
                f"round_"
                f"{round_number - 1:02d}.pt"
            )
        )


        if not previous_path.exists():

            raise FileNotFoundError(
                "Previous local MOON model "
                f"missing:\n{previous_path}"
            )


        previous_state = torch.load(

            previous_path,

            map_location="cpu",

            weights_only=True,
        )


        previous_reference = ECGCNN(
            NUM_CLASSES
        )

        previous_reference.load_state_dict(
            previous_state,
            strict=True,
        )

        previous_reference.to(
            DEVICE
        )

        freeze_model(
            previous_reference
        )


        del previous_state

        cleanup_memory()


    # --------------------------------------------------------------
    # LOSSES / OPTIMIZER
    # --------------------------------------------------------------

    task_criterion = (
        nn.BCEWithLogitsLoss(
            reduction="sum"
        )
    )


    optimizer = torch.optim.Adam(

        local_model.parameters(),

        lr=LR,
    )


    total_task_sum = 0.0
    total_moon_sum = 0.0
    total_samples = 0

    optimizer_step_count = 0


    # --------------------------------------------------------------
    # LOCAL EPOCHS
    # --------------------------------------------------------------

    for local_epoch in range(
        LOCAL_EPOCHS
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_samples = 0
        accumulated_micro_batches = 0

        n_micro_batches = len(
            loader
        )


        for micro_index, (
            x,
            y,
        ) in enumerate(
            loader
        ):

            x = x.to(
                DEVICE,
                dtype=torch.float32,
            )

            y = y.to(
                DEVICE,
                dtype=torch.float32,
            )

            micro_size = int(
                x.size(0)
            )


            # ------------------------------------------------------
            # GLOBAL / PREVIOUS REPRESENTATIONS
            # No gradient.
            # ------------------------------------------------------

            if use_moon:

                with torch.no_grad():

                    positive_representation = (
                        global_reference
                        .get_representation(
                            x
                        )
                    )

                    negative_representation = (
                        previous_reference
                        .get_representation(
                            x
                        )
                    )


            # ------------------------------------------------------
            # CURRENT LOCAL FORWARD
            # ------------------------------------------------------

            logits, current_representation = (
                local_model
                .forward_with_representation(
                    x
                )
            )


            # ------------------------------------------------------
            # TRUE MULTI-LABEL TASK LOSS
            # SUM over samples and labels.
            # ------------------------------------------------------

            task_loss_sum = (
                task_criterion(
                    logits,
                    y,
                )
            )


            # ------------------------------------------------------
            # MOON CONTRASTIVE LOSS
            # ------------------------------------------------------

            if use_moon:

                positive_similarity = (
                    F.cosine_similarity(

                        current_representation,

                        positive_representation,

                        dim=1,
                    )
                )


                negative_similarity = (
                    F.cosine_similarity(

                        current_representation,

                        negative_representation,

                        dim=1,
                    )
                )


                contrastive_logits = (
                    torch.stack(
                        [
                            positive_similarity,
                            negative_similarity,
                        ],
                        dim=1,
                    )
                    /
                    TEMPERATURE
                )


                contrastive_target = torch.zeros(

                    micro_size,

                    dtype=torch.long,

                    device=DEVICE,
                )


                moon_loss_sum = (
                    F.cross_entropy(

                        contrastive_logits,

                        contrastive_target,

                        reduction="sum",
                    )
                )

            else:

                moon_loss_sum = (
                    task_loss_sum
                    *
                    0.0
                )


            # ------------------------------------------------------
            # LOSS NUMERATOR FOR GRADIENT ACCUMULATION
            #
            # task_loss_sum / NUM_CLASSES:
            #     sum per sample, averaged across 5 labels.
            #
            # moon_loss_sum:
            #     sum per sample.
            #
            # Later gradients are divided by accumulated_samples.
            # ------------------------------------------------------

            gradient_loss = (

                task_loss_sum
                /
                NUM_CLASSES

                +

                MOON_MU
                *
                moon_loss_sum
            )


            gradient_loss.backward()


            # ------------------------------------------------------
            # LOGGING SUMS
            # ------------------------------------------------------

            total_task_sum += float(
                task_loss_sum.item()
            )


            if use_moon:

                total_moon_sum += float(
                    moon_loss_sum.item()
                )


            total_samples += micro_size

            accumulated_samples += (
                micro_size
            )

            accumulated_micro_batches += 1


            # ------------------------------------------------------
            # FREE MICRO-BATCH ACTIVATIONS
            # ------------------------------------------------------

            del logits
            del current_representation
            del task_loss_sum
            del moon_loss_sum
            del gradient_loss


            if use_moon:

                del positive_representation
                del negative_representation
                del positive_similarity
                del negative_similarity
                del contrastive_logits
                del contrastive_target


            del x
            del y


            # ------------------------------------------------------
            # ONE ADAM STEP FOR EFFECTIVE BATCH <=32
            # ------------------------------------------------------

            full_effective_batch = (
                accumulated_micro_batches
                ==
                ACCUMULATION_STEPS
            )

            last_micro_batch = (
                micro_index
                ==
                n_micro_batches - 1
            )


            if (
                full_effective_batch
                or
                last_micro_batch
            ):

                normalize_gradients(

                    local_model,

                    accumulated_samples,
                )


                optimizer.step()


                optimizer.zero_grad(
                    set_to_none=True
                )


                optimizer_step_count += 1


                accumulated_samples = 0
                accumulated_micro_batches = 0


                if (
                    optimizer_step_count
                    % 25
                    == 0
                ):

                    cleanup_memory()


        cleanup_memory()


    # --------------------------------------------------------------
    # CLIENT LOSSES
    # --------------------------------------------------------------

    if total_samples <= 0:

        raise RuntimeError(
            "No training samples."
        )


    mean_task_loss = (

        total_task_sum

        /
        (
            total_samples
            *
            NUM_CLASSES
        )
    )


    if use_moon:

        mean_moon_loss = (

            total_moon_sum

            /
            total_samples
        )

    else:

        mean_moon_loss = 0.0


    mean_total_loss = (

        mean_task_loss

        +

        MOON_MU
        *
        mean_moon_loss
    )


    local_state = state_to_cpu(
        local_model
    )


    # --------------------------------------------------------------
    # RELEASE MODELS
    # --------------------------------------------------------------

    del optimizer
    del task_criterion
    del loader
    del dataset
    del local_model


    if global_reference is not None:
        del global_reference

    if previous_reference is not None:
        del previous_reference


    cleanup_memory()


    return (
        local_state,
        float(mean_task_loss),
        float(mean_moon_loss),
        float(mean_total_loss),
    )


# ======================================================================
# MEMORY-EFFICIENT WEIGHTED FEDAVG
# ======================================================================

def create_accumulator(
    reference_state,
):

    return {
        key:
            torch.zeros_like(
                value,
                device="cpu",
            )

        for key, value
        in reference_state.items()
    }


def add_to_accumulator(
    accumulator,
    local_state,
    n_samples,
):

    weight = float(
        n_samples
    )

    for key in accumulator.keys():

        accumulator[
            key
        ].add_(

            local_state[
                key
            ],

            alpha=weight,
        )


def finalize_fedavg(
    accumulator,
    total_weight,
):

    if total_weight <= 0:

        raise RuntimeError(
            "Invalid total FedAvg weight."
        )


    new_state = {}


    for key, value in accumulator.items():

        new_state[
            key
        ] = (

            value
            /
            float(total_weight)
        )


    return new_state


# ======================================================================
# SAVE CSV LOGS
# ======================================================================

def save_logs(
    training_rows,
    round_rows,
):

    pd.DataFrame(
        training_rows
    ).to_csv(
        TRAINING_LOG_FILE,
        index=False,
    )

    pd.DataFrame(
        round_rows
    ).to_csv(
        ROUND_SUMMARY_FILE,
        index=False,
    )


# ======================================================================
# CONFIGURATION
# ======================================================================

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "STEP 6.13 - N2 MOON MULTI-LABEL\n"
    )

    file.write(
        "=" * 70
        + "\n"
    )

    file.write(
        f"Seed: {SEED}\n"
    )

    file.write(
        f"Rounds: {ROUNDS}\n"
    )

    file.write(
        f"Local epochs: "
        f"{LOCAL_EPOCHS}\n"
    )

    file.write(
        f"Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}\n"
    )

    file.write(
        f"Micro-batch size: "
        f"{MICRO_BATCH_SIZE}\n"
    )

    file.write(
        f"Accumulation steps: "
        f"{ACCUMULATION_STEPS}\n"
    )

    file.write(
        f"Learning rate: "
        f"{LR}\n"
    )

    file.write(
        f"MOON mu: "
        f"{MOON_MU}\n"
    )

    file.write(
        f"Temperature: "
        f"{TEMPERATURE}\n"
    )

    file.write(
        "Task: TRUE multi-label\n"
    )

    file.write(
        "Task loss: BCEWithLogitsLoss\n"
    )

    file.write(
        "Representation: final pooled "
        "256-dimensional backbone output\n"
    )

    file.write(
        "Positive: round-start global model\n"
    )

    file.write(
        "Negative: previous-round local client model\n"
    )

    file.write(
        "Round 1 MOON loss: 0\n"
    )

    file.write(
        "Projection head: NO\n"
    )

    file.write(
        "Aggregation: Weighted FedAvg\n"
    )

    file.write(
        "Classes: NORM, MI, STTC, CD, HYP\n"
    )

    file.write(
        "Partition: N2 extreme Non-IID\n"
    )

    file.write(
        "Gradient accumulation: enabled\n"
    )

    file.write(
        "Crash-safe resume: enabled\n"
    )


# ======================================================================
# INITIALIZE / RESUME
# ======================================================================

training_rows = []
round_rows = []


if RESUME_FILE.exists():

    print("\n" + "=" * 80)
    print("RESUME STATE FOUND")
    print("=" * 80)


    checkpoint = torch.load(

        RESUME_FILE,

        map_location="cpu",

        weights_only=False,
    )


    if checkpoint.get(
        "resume_version"
    ) != RESUME_VERSION:

        raise RuntimeError(
            "Incompatible resume checkpoint. "
            "Delete N2_MOON_seed2026 folder "
            "and restart."
        )


    current_round = int(
        checkpoint[
            "round_number"
        ]
    )

    next_client_index = int(
        checkpoint[
            "next_client_index"
        ]
    )

    global_state = checkpoint[
        "global_state"
    ]

    accumulator = checkpoint[
        "accumulator"
    ]

    accumulated_weight = int(
        checkpoint[
            "accumulated_weight"
        ]
    )

    current_task_losses = checkpoint[
        "current_task_losses"
    ]

    current_moon_losses = checkpoint[
        "current_moon_losses"
    ]

    current_total_losses = checkpoint[
        "current_total_losses"
    ]

    training_rows = checkpoint[
        "training_rows"
    ]

    round_rows = checkpoint[
        "round_rows"
    ]


    restore_rng_state(
        checkpoint[
            "rng_state"
        ]
    )


    print(
        f"Round to continue : "
        f"{current_round}"
    )

    if next_client_index < len(
        CLIENTS
    ):

        print(
            f"Next client       : "
            f"{CLIENTS[next_client_index]}"
        )


    del checkpoint

    cleanup_memory()


else:

    print("\n" + "=" * 80)

    print(
        "STARTING NEW MOON SEED 2026 RUN"
    )

    print("=" * 80)


    # IMPORTANT:
    # Same model-initialization pattern as
    # the memory-safe FedAvg seed2026 script.
    initial_model = ECGCNN(
        NUM_CLASSES
    )


    global_state = state_to_cpu(
        initial_model
    )


    del initial_model

    cleanup_memory()


    atomic_torch_save(

        global_state,

        OUTPUT_DIR
        / "global_round_00.pt",
    )


    current_round = 1

    next_client_index = 0

    accumulator = create_accumulator(
        global_state
    )

    accumulated_weight = 0

    current_task_losses = []
    current_moon_losses = []
    current_total_losses = []


    atomic_torch_save(

        {
            "resume_version":
                RESUME_VERSION,

            "round_number":
                current_round,

            "next_client_index":
                next_client_index,

            "global_state":
                global_state,

            "accumulator":
                accumulator,

            "accumulated_weight":
                accumulated_weight,

            "current_task_losses":
                current_task_losses,

            "current_moon_losses":
                current_moon_losses,

            "current_total_losses":
                current_total_losses,

            "training_rows":
                training_rows,

            "round_rows":
                round_rows,

            "rng_state":
                get_rng_state(),
        },

        RESUME_FILE,
    )


# ======================================================================
# FEDERATED TRAINING
# ======================================================================

EXPECTED_TOTAL_TRAIN = sum(
    EXPECTED_CLIENT_SIZES.values()
)


print("\n" + "=" * 80)
print("STARTING / RESUMING N2 MOON TRAINING")
print("=" * 80)


while current_round <= ROUNDS:

    print(
        f"\n"
        f"{'=' * 80}"
    )

    print(
        f"FEDERATED ROUND "
        f"{current_round}/{ROUNDS}"
    )

    print(
        "=" * 80
    )


    for client_index in range(

        next_client_index,

        len(CLIENTS),
    ):

        client_name = (
            CLIENTS[
                client_index
            ]
        )

        client_size = (
            client_data[
                client_name
            ]["size"]
        )


        print(
            f"\nTraining "
            f"{client_name}..."
        )


        cleanup_memory()


        (
            local_state,
            task_loss,
            moon_loss,
            total_loss,

        ) = train_local_moon(

            global_state=
                global_state,

            client_name=
                client_name,

            round_number=
                current_round,
        )


        print(
            f"{client_name} | "
            f"task={task_loss:.6f} | "
            f"MOON={moon_loss:.6f} | "
            f"total={total_loss:.6f}"
        )


        # ----------------------------------------------------------
        # SAVE LOCAL
        # ----------------------------------------------------------

        local_path = (

            OUTPUT_DIR

            / (
                f"local_"
                f"{client_name}_"
                f"round_"
                f"{current_round:02d}.pt"
            )
        )


        atomic_torch_save(
            local_state,
            local_path,
        )


        print(
            f"Saved: "
            f"{local_path.name}"
        )


        # ----------------------------------------------------------
        # WEIGHTED FEDAVG
        # ----------------------------------------------------------

        add_to_accumulator(

            accumulator,

            local_state,

            client_size,
        )


        accumulated_weight += (
            client_size
        )


        current_task_losses.append(
            task_loss
        )

        current_moon_losses.append(
            moon_loss
        )

        current_total_losses.append(
            total_loss
        )


        training_rows.append(
            {
                "seed":
                    SEED,

                "round":
                    current_round,

                "client":
                    client_name,

                "n_samples":
                    client_size,

                "task_loss":
                    task_loss,

                "moon_loss":
                    moon_loss,

                "total_loss":
                    total_loss,
            }
        )


        del local_state

        cleanup_memory()


        save_logs(
            training_rows,
            round_rows,
        )


        # ----------------------------------------------------------
        # CRASH-SAFE RESUME AFTER EACH CLIENT
        # ----------------------------------------------------------

        atomic_torch_save(

            {
                "resume_version":
                    RESUME_VERSION,

                "round_number":
                    current_round,

                "next_client_index":
                    client_index + 1,

                "global_state":
                    global_state,

                "accumulator":
                    accumulator,

                "accumulated_weight":
                    accumulated_weight,

                "current_task_losses":
                    current_task_losses,

                "current_moon_losses":
                    current_moon_losses,

                "current_total_losses":
                    current_total_losses,

                "training_rows":
                    training_rows,

                "round_rows":
                    round_rows,

                "rng_state":
                    get_rng_state(),
            },

            RESUME_FILE,
        )


    # ==================================================================
    # AGGREGATE
    # ==================================================================

    if (
        accumulated_weight
        !=
        EXPECTED_TOTAL_TRAIN
    ):

        raise RuntimeError(
            f"Round {current_round}: "
            f"expected {EXPECTED_TOTAL_TRAIN}, "
            f"found {accumulated_weight}."
        )


    new_global_state = (
        finalize_fedavg(

            accumulator,

            accumulated_weight,
        )
    )


    del global_state

    global_state = (
        new_global_state
    )

    del new_global_state


    mean_task_loss = float(
        np.mean(
            current_task_losses
        )
    )

    mean_moon_loss = float(
        np.mean(
            current_moon_losses
        )
    )

    mean_total_loss = float(
        np.mean(
            current_total_losses
        )
    )


    print(
        "\nROUND SUMMARY"
    )

    print(
        f"Mean task loss : "
        f"{mean_task_loss:.6f}"
    )

    print(
        f"Mean MOON loss : "
        f"{mean_moon_loss:.6f}"
    )

    print(
        f"Mean total loss: "
        f"{mean_total_loss:.6f}"
    )


    # ==================================================================
    # SAVE GLOBAL
    # ==================================================================

    global_path = (

        OUTPUT_DIR

        / (
            f"global_round_"
            f"{current_round:02d}.pt"
        )
    )


    atomic_torch_save(
        global_state,
        global_path,
    )


    print(
        f"Global saved   : "
        f"{global_path.name}"
    )


    round_rows.append(
        {
            "seed":
                SEED,

            "round":
                current_round,

            "mean_task_loss":
                mean_task_loss,

            "mean_moon_loss":
                mean_moon_loss,

            "mean_total_loss":
                mean_total_loss,

            "total_samples":
                accumulated_weight,
        }
    )


    save_logs(
        training_rows,
        round_rows,
    )


    # ==================================================================
    # NEXT ROUND
    # ==================================================================

    del accumulator

    cleanup_memory()


    current_round += 1


    if current_round <= ROUNDS:

        next_client_index = 0

        accumulator = (
            create_accumulator(
                global_state
            )
        )

        accumulated_weight = 0

        current_task_losses = []
        current_moon_losses = []
        current_total_losses = []


        atomic_torch_save(

            {
                "resume_version":
                    RESUME_VERSION,

                "round_number":
                    current_round,

                "next_client_index":
                    0,

                "global_state":
                    global_state,

                "accumulator":
                    accumulator,

                "accumulated_weight":
                    0,

                "current_task_losses":
                    [],

                "current_moon_losses":
                    [],

                "current_total_losses":
                    [],

                "training_rows":
                    training_rows,

                "round_rows":
                    round_rows,

                "rng_state":
                    get_rng_state(),
            },

            RESUME_FILE,
        )


# ======================================================================
# FINAL GLOBAL MODEL
# ======================================================================

final_global_path = (

    OUTPUT_DIR
    / "global_final_round_20.pt"
)


atomic_torch_save(
    global_state,
    final_global_path,
)


# ======================================================================
# CHECKPOINT VERIFICATION
# ======================================================================

print("\n" + "=" * 80)
print("VERIFYING CHECKPOINTS")
print("=" * 80)


missing_files = []


for round_number in range(
    1,
    ROUNDS + 1,
):

    global_path = (

        OUTPUT_DIR

        / (
            f"global_round_"
            f"{round_number:02d}.pt"
        )
    )

    if not global_path.exists():

        missing_files.append(
            global_path
        )


    for client_name in CLIENTS:

        local_path = (

            OUTPUT_DIR

            / (
                f"local_"
                f"{client_name}_"
                f"round_"
                f"{round_number:02d}.pt"
            )
        )

        if not local_path.exists():

            missing_files.append(
                local_path
            )


if missing_files:

    print(
        "\nMissing checkpoints:"
    )

    for path in missing_files:

        print(
            " -",
            path
        )

    raise RuntimeError(
        "Checkpoint verification failed."
    )


print(
    "100 local checkpoints : OK"
)

print(
    "20 global checkpoints : OK"
)

print(
    "Final global model     : OK"
)


# ======================================================================
# COMPLETION FILE
# ======================================================================

with open(
    COMPLETION_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "N2 MOON seed 2026 completed successfully.\n"
    )

    file.write(
        f"Seed: {SEED}\n"
    )

    file.write(
        f"Rounds: {ROUNDS}\n"
    )

    file.write(
        f"Local epochs: {LOCAL_EPOCHS}\n"
    )

    file.write(
        f"Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}\n"
    )

    file.write(
        f"Micro-batch size: "
        f"{MICRO_BATCH_SIZE}\n"
    )

    file.write(
        f"MOON mu: {MOON_MU}\n"
    )

    file.write(
        f"Temperature: "
        f"{TEMPERATURE}\n"
    )

    file.write(
        "Task: TRUE multi-label\n"
    )

    file.write(
        "Task loss: BCEWithLogitsLoss\n"
    )

    file.write(
        "Projection head: NO\n"
    )

    file.write(
        "Representation: pooled 256d backbone\n"
    )


# ======================================================================
# DELETE RESUME ONLY AFTER FULL SUCCESS
# ======================================================================

if RESUME_FILE.exists():

    RESUME_FILE.unlink()


del global_state

cleanup_memory()


# ======================================================================
# DONE
# ======================================================================

print("\n" + "=" * 80)

print(
    "STEP 6.13 N2 MOON SEED 2026 "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 80)


print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)


print(
    "\nConfiguration:"
)

print(
    f"  Seed                 : "
    f"{SEED}"
)

print(
    f"  Effective batch      : "
    f"{EFFECTIVE_BATCH_SIZE}"
)

print(
    f"  Micro-batch          : "
    f"{MICRO_BATCH_SIZE}"
)

print(
    f"  Accumulation steps   : "
    f"{ACCUMULATION_STEPS}"
)

print(
    f"  MOON mu              : "
    f"{MOON_MU}"
)

print(
    f"  Temperature          : "
    f"{TEMPERATURE}"
)

print(
    "  True multi-label     : YES"
)

print(
    "  BCEWithLogitsLoss    : YES"
)

print(
    "  Previous local model : YES"
)

print(
    "  Round-start global   : YES"
)

print(
    "  Projection head      : NO"
)

print(
    "  Memory-safe          : YES"
)

print(
    "  Crash-safe resume    : YES"
)

print("=" * 80)