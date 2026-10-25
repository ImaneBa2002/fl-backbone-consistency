# ======================================================================
# STEP 6.14
# MULTI-SEED FINAL PERFORMANCE + BACKBONE CKA ANALYSIS
#
# FAIR COMPARISON:
#   FedAvg : seeds 42_matched, 123, 2026
#   MOON   : seeds 42_matched, 123, 2026
#
# Performance:
#   Final global model only (Round 20)
#   Official held-out PTB-XL test fold 10
#
# CKA:
#   Fixed shared 500 validation ECGs from fold 9
#   Rounds: 1, 2, 5, 10, 15, 20
#   Client-Client
#   Local-Global
#   Conv1, Conv2, Conv3, Conv4, Final
#
# Outputs:
#   Raw results
#   mean +/- std across 3 seeds
#   paired MOON - FedAvg deltas
#
# ======================================================================

from pathlib import Path
from itertools import combinations
import ast
import gc
import sys

import numpy as np
import pandas as pd
import wfdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

SEEDS = [
    42,
    123,
    2026,
]

METHODS = [
    "FedAvg",
    "MOON",
]

CLIENTS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
]

CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

NUM_CLASSES = len(
    CLASS_NAMES
)

CLASS_TO_INDEX = {
    name: index
    for index, name
    in enumerate(CLASS_NAMES)
}

CKA_ROUNDS = [
    1,
    2,
    5,
    10,
    15,
    20,
]

LAYERS = [
    "Conv1",
    "Conv2",
    "Conv3",
    "Conv4",
    "Final",
]

THRESHOLD = 0.5

# Small evaluation batch because PC RAM is limited.
EVAL_BATCH_SIZE = 8
CKA_BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# 2. PATHS
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

SHARED_500_FILE = (
    ROOT
    / "step6_4_2_shared_cka_N0_N1"
    / "shared_500_validation_ecgs_used.csv"
)

OUTPUT_DIR = (
    ROOT
    / "step6_14_multiseed_evaluation_CKA"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_DIRS = {

    "FedAvg": {

        42:
            ROOT
            / "step6_13_multiseed"
            / "N2_FedAvg_seed42_matched",

        123:
            ROOT
            / "step6_13_multiseed"
            / "N2_FedAvg_seed123",

        2026:
            ROOT
            / "step6_13_multiseed"
            / "N2_FedAvg_seed2026",
    },

    "MOON": {

        42:
            ROOT
            / "step6_13_multiseed"
            / "N2_MOON_seed42_matched",

        123:
            ROOT
            / "step6_13_multiseed"
            / "N2_MOON_seed123",

        2026:
            ROOT
            / "step6_13_multiseed"
            / "N2_MOON_seed2026",
    },
}


# ======================================================================
# 3. TRAIN-ONLY NORMALIZATION
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
# 4. MEMORY
# ======================================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# 5. CNN
# EXACT SAME ARCHITECTURE
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

        self.pool = nn.MaxPool1d(
            2
        )

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            256,
            num_classes,
        )


    def forward(self, x):

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

        x = x.squeeze(-1)

        return self.fc(x)


    def extract_cka_features(
        self,
        x,
    ):

        features = {}

        # ----------------------------------------------------------
        # Conv1:
        # after Conv + GN + ReLU
        # BEFORE MaxPool
        # temporal mean
        # ----------------------------------------------------------

        x = self.conv1(x)
        x = torch.relu(x)

        features[
            "Conv1"
        ] = x.mean(
            dim=-1
        )

        x = self.pool(x)


        # ----------------------------------------------------------
        # Conv2
        # ----------------------------------------------------------

        x = self.conv2(x)
        x = torch.relu(x)

        features[
            "Conv2"
        ] = x.mean(
            dim=-1
        )

        x = self.pool(x)


        # ----------------------------------------------------------
        # Conv3
        # ----------------------------------------------------------

        x = self.conv3(x)
        x = torch.relu(x)

        features[
            "Conv3"
        ] = x.mean(
            dim=-1
        )

        x = self.pool(x)


        # ----------------------------------------------------------
        # Conv4
        # ----------------------------------------------------------

        x = self.conv4(x)
        x = torch.relu(x)

        features[
            "Conv4"
        ] = x.mean(
            dim=-1
        )


        # ----------------------------------------------------------
        # Final pooled representation
        # Conv4 ReLU -> MaxPool -> AdaptiveAvgPool
        # 256-dimensional
        # ----------------------------------------------------------

        x = self.pool(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        features[
            "Final"
        ] = x

        return features


# ======================================================================
# 6. ARCHITECTURE SANITY CHECK
# ======================================================================

temporary_model = ECGCNN(
    NUM_CLASSES
)

parameter_count = sum(
    p.numel()
    for p
    in temporary_model.parameters()
)

print("=" * 80)
print("STEP 6.14 - MULTI-SEED PERFORMANCE + CKA")
print("=" * 80)

print(
    f"Device              : {DEVICE}"
)

print(
    f"Model parameters    : {parameter_count}"
)

print(
    f"Seeds               : {SEEDS}"
)

print(
    f"Methods             : {METHODS}"
)

print(
    f"CKA rounds          : {CKA_ROUNDS}"
)

print(
    f"Test threshold      : {THRESHOLD}"
)

if parameter_count != 224549:

    raise RuntimeError(
        "CNN architecture mismatch."
    )

del temporary_model

cleanup_memory()


# ======================================================================
# 7. CHECK REQUIRED FILES
# ======================================================================

print(
    "\nChecking required files..."
)

required_files = {
    "ptbxl_database.csv":
        DATABASE_FILE,

    "scp_statements.csv":
        SCP_FILE,

    "shared 500 CKA":
        SHARED_500_FILE,
}

for name, path in required_files.items():

    status = (
        "FOUND"
        if path.exists()
        else "MISSING"
    )

    print(
        f"{name:<25}: {status}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"{name} missing:\n{path}"
        )


# ======================================================================
# 8. VERIFY ALL MODEL DIRECTORIES / CHECKPOINTS
# ======================================================================

print(
    "\nVerifying model checkpoints..."
)

for method in METHODS:

    for seed in SEEDS:

        model_dir = MODEL_DIRS[
            method
        ][seed]

        if not model_dir.exists():

            raise FileNotFoundError(
                f"\nMissing model directory:\n"
                f"{model_dir}"
            )


        # Final global checkpoint
        final_global = (
            model_dir
            / "global_round_20.pt"
        )

        if not final_global.exists():

            raise FileNotFoundError(
                f"Missing final global:\n"
                f"{final_global}"
            )


        # Required CKA rounds
        for round_number in CKA_ROUNDS:

            global_path = (
                model_dir
                / (
                    f"global_round_"
                    f"{round_number:02d}.pt"
                )
            )

            if not global_path.exists():

                raise FileNotFoundError(
                    f"Missing:\n"
                    f"{global_path}"
                )


            for client_name in CLIENTS:

                local_path = (
                    model_dir
                    / (
                        f"local_"
                        f"{client_name}_"
                        f"round_"
                        f"{round_number:02d}.pt"
                    )
                )

                if not local_path.exists():

                    raise FileNotFoundError(
                        f"Missing:\n"
                        f"{local_path}"
                    )


        print(
            f"{method:<8} seed "
            f"{seed:<4}: OK"
        )


# ======================================================================
# 9. LOAD DATABASE + DIAGNOSTIC MAPPING
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

if len(database) != 21799:

    raise RuntimeError(
        "Expected 21799 PTB-XL rows."
    )


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


if len(diagnostic_mapping) != 44:

    raise RuntimeError(
        "Expected 44 diagnostic mappings."
    )


# ======================================================================
# 10. TRUE MULTI-LABEL TARGETS
# ======================================================================

def parse_scp_codes(raw):

    if isinstance(
        raw,
        dict,
    ):
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

            class_name = (
                diagnostic_mapping[
                    code
                ]
            )

            if (
                class_name
                in CLASS_TO_INDEX
                and
                class_name
                not in classes
            ):

                classes.append(
                    class_name
                )

    return classes


def make_multihot(classes):

    target = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for class_name in classes:

        target[
            CLASS_TO_INDEX[
                class_name
            ]
        ] = 1.0

    return target


print(
    "Building TRUE multi-label targets..."
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


print(
    f"Usable ECGs          : "
    f"{len(usable_database)}"
)

if len(
    usable_database
) != 21388:

    raise RuntimeError(
        "Expected 21388 usable ECGs."
    )


# ======================================================================
# 11. OFFICIAL HELD-OUT TEST FOLD 10
# ======================================================================

test_df = usable_database[
    usable_database[
        "strat_fold"
    ] == 10
].copy()


print(
    f"Official test ECGs   : "
    f"{len(test_df)}"
)

if len(test_df) != 2158:

    raise RuntimeError(
        "Expected 2158 test ECGs."
    )


test_positive_counts = np.stack(
    test_df[
        "target"
    ].to_numpy()
).sum(
    axis=0
).astype(int)


print(
    "\nTest positive counts:"
)

for class_name, count in zip(
    CLASS_NAMES,
    test_positive_counts,
):

    print(
        f"  {class_name:<5}: "
        f"{count}"
    )


# ======================================================================
# 12. SHARED FIXED 500 VALIDATION ECGs FOR CKA
# ======================================================================

print(
    "\nLoading shared fixed 500 CKA ECGs..."
)

shared_source = pd.read_csv(
    SHARED_500_FILE
)


if "ecg_id" in shared_source.columns:

    shared_ids = (
        shared_source[
            "ecg_id"
        ]
        .astype(int)
        .tolist()
    )

    metadata_by_id = (
        usable_database
        .set_index(
            "ecg_id"
        )
    )

    try:

        shared_df = (
            metadata_by_id
            .loc[
                shared_ids
            ]
            .reset_index()
        )

    except KeyError as error:

        raise RuntimeError(
            "Some shared ECG IDs were not "
            "found in usable PTB-XL."
        ) from error


elif "filename_lr" in shared_source.columns:

    filenames = (
        shared_source[
            "filename_lr"
        ]
        .astype(str)
        .tolist()
    )

    metadata_by_file = (
        usable_database
        .set_index(
            "filename_lr"
        )
    )

    try:

        shared_df = (
            metadata_by_file
            .loc[
                filenames
            ]
            .reset_index()
        )

    except KeyError as error:

        raise RuntimeError(
            "Some shared filenames were not "
            "found in usable PTB-XL."
        ) from error


else:

    raise RuntimeError(
        "Shared 500 CSV must contain "
        "'ecg_id' or 'filename_lr'."
    )


print(
    f"Shared CKA ECGs      : "
    f"{len(shared_df)}"
)

if len(shared_df) != 500:

    raise RuntimeError(
        "Expected exactly 500 shared ECGs."
    )


if not (
    shared_df[
        "strat_fold"
    ] == 9
).all():

    raise RuntimeError(
        "Shared CKA ECGs must all come "
        "from validation fold 9."
    )


if shared_df[
    "ecg_id"
].nunique() != 500:

    raise RuntimeError(
        "Shared CKA ECG IDs are not unique."
    )


print(
    "Shared CKA fold     : 9 only"
)

print(
    "Shared CKA IDs      : unique"
)


# ======================================================================
# 13. DATASET
# ======================================================================

class PTBXLEvaluationDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
    ):

        self.filenames = (
            dataframe[
                "filename_lr"
            ]
            .astype(str)
            .tolist()
        )

        self.targets = np.stack(
            dataframe[
                "target"
            ].to_numpy()
        ).astype(
            np.float32
        )

        self.ecg_ids = (
            dataframe[
                "ecg_id"
            ]
            .astype(int)
            .to_numpy()
        )


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

        signal, _ = wfdb.rdsamp(
            str(
                ROOT
                / filename
            )
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
                f"{signal.shape}: "
                f"{filename}"
            )


        signal = (
            signal
            -
            TRAIN_MEAN[:, None]
        ) / (
            TRAIN_STD[:, None]
        )


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

            int(
                self.ecg_ids[
                    index
                ]
            ),
        )


# ======================================================================
# 14. LOADERS
# ======================================================================

test_dataset = (
    PTBXLEvaluationDataset(
        test_df
    )
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


cka_dataset = (
    PTBXLEvaluationDataset(
        shared_df
    )
)

cka_loader = DataLoader(
    cka_dataset,
    batch_size=CKA_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ======================================================================
# 15. CHECKPOINT LOADER
# ======================================================================

def load_state_dict_safe(
    checkpoint_path,
):

    try:

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    return state


def load_model(
    checkpoint_path,
):

    model = ECGCNN(
        NUM_CLASSES
    )

    state = load_state_dict_safe(
        checkpoint_path
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    del state

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ======================================================================
# 16. FINAL TEST PERFORMANCE
# ======================================================================

def evaluate_model(
    checkpoint_path,
):

    model = load_model(
        checkpoint_path
    )

    logits_list = []
    targets_list = []

    total_bce_sum = 0.0
    total_elements = 0


    with torch.no_grad():

        for x, y, _ in test_loader:

            x = x.to(
                DEVICE,
                dtype=torch.float32,
            )

            y = y.to(
                DEVICE,
                dtype=torch.float32,
            )


            logits = model(x)


            bce_sum = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    reduction="sum",
                )
            )


            total_bce_sum += float(
                bce_sum.item()
            )

            total_elements += int(
                y.numel()
            )


            logits_list.append(
                logits.detach()
                .cpu()
                .numpy()
            )

            targets_list.append(
                y.detach()
                .cpu()
                .numpy()
            )


            del x
            del y
            del logits
            del bce_sum


    logits_all = np.concatenate(
        logits_list,
        axis=0,
    )

    y_true = np.concatenate(
        targets_list,
        axis=0,
    ).astype(
        np.int32
    )


    probabilities = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -logits_all
            )
        )
    )


    y_pred = (
        probabilities
        >=
        THRESHOLD
    ).astype(
        np.int32
    )


    bce_loss = (
        total_bce_sum
        /
        total_elements
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )


    per_class_auc = []

    for class_index in range(
        NUM_CLASSES
    ):

        auc = roc_auc_score(

            y_true[
                :,
                class_index
            ],

            probabilities[
                :,
                class_index
            ],
        )

        per_class_auc.append(
            auc
        )


    per_class_auc = np.array(
        per_class_auc,
        dtype=np.float64,
    )


    macro_auroc = float(
        np.mean(
            per_class_auc
        )
    )


    exact_match = float(
        np.mean(
            np.all(
                y_true
                ==
                y_pred,
                axis=1,
            )
        )
    )


    del model
    del logits_all
    del probabilities
    del y_pred

    cleanup_memory()


    return {

        "loss":
            float(bce_loss),

        "macro_f1":
            float(macro_f1),

        "macro_auroc":
            float(macro_auroc),

        "exact_match":
            float(exact_match),

        "per_class_f1":
            per_class_f1,

        "per_class_auroc":
            per_class_auc,
    }


# ======================================================================
# 17. RUN FINAL TEST PERFORMANCE
# ======================================================================

print("\n" + "=" * 80)
print("FINAL HELD-OUT TEST PERFORMANCE - ROUND 20")
print("=" * 80)


performance_rows = []
per_class_rows = []


for method in METHODS:

    for seed in SEEDS:

        model_dir = MODEL_DIRS[
            method
        ][seed]

        checkpoint_path = (
            model_dir
            / "global_round_20.pt"
        )


        print(
            f"\nEvaluating "
            f"{method} seed {seed}..."
        )


        metrics = evaluate_model(
            checkpoint_path
        )


        print(
            f"  Loss        = "
            f"{metrics['loss']:.6f}"
        )

        print(
            f"  Macro-F1    = "
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            f"  Macro-AUROC = "
            f"{metrics['macro_auroc']:.6f}"
        )

        print(
            f"  Exact Match = "
            f"{metrics['exact_match']:.6f}"
        )


        performance_rows.append(
            {
                "method":
                    method,

                "seed":
                    seed,

                "loss":
                    metrics[
                        "loss"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "macro_auroc":
                    metrics[
                        "macro_auroc"
                    ],

                "exact_match":
                    metrics[
                        "exact_match"
                    ],
            }
        )


        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):

            per_class_rows.append(
                {
                    "method":
                        method,

                    "seed":
                        seed,

                    "class":
                        class_name,

                    "f1":
                        float(
                            metrics[
                                "per_class_f1"
                            ][
                                class_index
                            ]
                        ),

                    "auroc":
                        float(
                            metrics[
                                "per_class_auroc"
                            ][
                                class_index
                            ]
                        ),
                }
            )


# ======================================================================
# 18. PERFORMANCE DATAFRAMES
# ======================================================================

performance_df = pd.DataFrame(
    performance_rows
)

per_class_df = pd.DataFrame(
    per_class_rows
)


performance_df.to_csv(
    OUTPUT_DIR
    / "test_performance_raw_3seeds.csv",
    index=False,
)

per_class_df.to_csv(
    OUTPUT_DIR
    / "test_per_class_raw_3seeds.csv",
    index=False,
)


# ======================================================================
# 19. PERFORMANCE MEAN +/- STD
# ======================================================================

performance_summary_rows = []


for method in METHODS:

    subset = performance_df[
        performance_df[
            "method"
        ] == method
    ]


    for metric in [
        "loss",
        "macro_f1",
        "macro_auroc",
        "exact_match",
    ]:

        values = subset[
            metric
        ].to_numpy(
            dtype=np.float64
        )


        performance_summary_rows.append(
            {
                "method":
                    method,

                "metric":
                    metric,

                "n_seeds":
                    len(values),

                "mean":
                    float(
                        np.mean(values)
                    ),

                "std":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "min":
                    float(
                        np.min(values)
                    ),

                "max":
                    float(
                        np.max(values)
                    ),
            }
        )


performance_summary_df = pd.DataFrame(
    performance_summary_rows
)


performance_summary_df.to_csv(
    OUTPUT_DIR
    / "test_performance_mean_std.csv",
    index=False,
)


# ======================================================================
# 20. PER-CLASS MEAN +/- STD
# ======================================================================

per_class_summary_rows = []


for method in METHODS:

    for class_name in CLASS_NAMES:

        subset = per_class_df[
            (
                per_class_df[
                    "method"
                ] == method
            )
            &
            (
                per_class_df[
                    "class"
                ] == class_name
            )
        ]


        for metric in [
            "f1",
            "auroc",
        ]:

            values = subset[
                metric
            ].to_numpy(
                dtype=np.float64
            )


            per_class_summary_rows.append(
                {
                    "method":
                        method,

                    "class":
                        class_name,

                    "metric":
                        metric,

                    "mean":
                        float(
                            np.mean(values)
                        ),

                    "std":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),
                }
            )


per_class_summary_df = pd.DataFrame(
    per_class_summary_rows
)


per_class_summary_df.to_csv(
    OUTPUT_DIR
    / "test_per_class_mean_std.csv",
    index=False,
)


# ======================================================================
# 21. PAIRED PERFORMANCE DELTAS
# MOON - FEDAVG FOR SAME SEED
# ======================================================================

paired_performance_rows = []


for seed in SEEDS:

    fedavg_row = performance_df[
        (
            performance_df[
                "method"
            ] == "FedAvg"
        )
        &
        (
            performance_df[
                "seed"
            ] == seed
        )
    ].iloc[0]


    moon_row = performance_df[
        (
            performance_df[
                "method"
            ] == "MOON"
        )
        &
        (
            performance_df[
                "seed"
            ] == seed
        )
    ].iloc[0]


    row = {
        "seed":
            seed
    }


    for metric in [
        "loss",
        "macro_f1",
        "macro_auroc",
        "exact_match",
    ]:

        row[
            f"delta_{metric}"
        ] = float(
            moon_row[
                metric
            ]
            -
            fedavg_row[
                metric
            ]
        )


    paired_performance_rows.append(
        row
    )


paired_performance_df = pd.DataFrame(
    paired_performance_rows
)


paired_performance_df.to_csv(
    OUTPUT_DIR
    / "test_paired_MOON_minus_FedAvg.csv",
    index=False,
)


paired_performance_summary_rows = []


for metric in [
    "loss",
    "macro_f1",
    "macro_auroc",
    "exact_match",
]:

    values = paired_performance_df[
        f"delta_{metric}"
    ].to_numpy(
        dtype=np.float64
    )


    paired_performance_summary_rows.append(
        {
            "metric":
                metric,

            "mean_delta":
                float(
                    np.mean(values)
                ),

            "std_delta":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
        }
    )


paired_performance_summary_df = pd.DataFrame(
    paired_performance_summary_rows
)


paired_performance_summary_df.to_csv(
    OUTPUT_DIR
    / "test_paired_delta_mean_std.csv",
    index=False,
)


# ======================================================================
# 22. EXACT LINEAR CKA PROTOCOL
# ======================================================================

def center_matrix(X):

    return (
        X
        -
        X.mean(
            axis=0,
            keepdims=True,
        )
    )


def linear_cka(
    X,
    Y,
    eps=1e-8,
):

    X = center_matrix(X)
    Y = center_matrix(Y)

    numerator = (
        np.linalg.norm(
            X.T @ Y,
            ord="fro",
        )
        ** 2
    )

    denominator = (

        np.linalg.norm(
            X.T @ X,
            ord="fro",
        )

        *

        np.linalg.norm(
            Y.T @ Y,
            ord="fro",
        )
    )

    return float(
        numerator
        /
        (
            denominator
            +
            eps
        )
    )


# ======================================================================
# 23. EXTRACT FEATURES
# ======================================================================

def extract_features(
    checkpoint_path,
):

    model = load_model(
        checkpoint_path
    )


    layer_batches = {
        layer: []
        for layer in LAYERS
    }


    with torch.no_grad():

        for x, _, _ in cka_loader:

            x = x.to(
                DEVICE,
                dtype=torch.float32,
            )


            batch_features = (
                model.extract_cka_features(
                    x
                )
            )


            for layer in LAYERS:

                layer_batches[
                    layer
                ].append(

                    batch_features[
                        layer
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )


            del x
            del batch_features


    result = {}


    for layer in LAYERS:

        result[
            layer
        ] = np.concatenate(
            layer_batches[
                layer
            ],
            axis=0,
        )


        if result[
            layer
        ].shape[0] != 500:

            raise RuntimeError(
                f"{layer}: expected 500 "
                "feature rows."
            )


    del model
    del layer_batches

    cleanup_memory()

    return result


# ======================================================================
# 24. MULTI-SEED CKA
# ======================================================================

print("\n" + "=" * 80)
print("MULTI-SEED CKA ANALYSIS")
print("=" * 80)


cka_pairwise_rows = []
cka_seed_summary_rows = []


for method in METHODS:

    for seed in SEEDS:

        model_dir = MODEL_DIRS[
            method
        ][seed]


        print(
            "\n"
            + "-" * 80
        )

        print(
            f"{method} | seed {seed}"
        )

        print(
            "-" * 80
        )


        for round_number in CKA_ROUNDS:

            print(
                f"\nRound "
                f"{round_number}/20"
            )


            # ------------------------------------------------------
            # GLOBAL FEATURES
            # ------------------------------------------------------

            global_path = (
                model_dir
                / (
                    f"global_round_"
                    f"{round_number:02d}.pt"
                )
            )


            global_features = (
                extract_features(
                    global_path
                )
            )


            # ------------------------------------------------------
            # LOCAL FEATURES
            # ------------------------------------------------------

            local_features = {}


            for client_name in CLIENTS:

                print(
                    f"  Extracting "
                    f"{client_name}...",
                    end="",
                    flush=True,
                )


                local_path = (
                    model_dir
                    / (
                        f"local_"
                        f"{client_name}_"
                        f"round_"
                        f"{round_number:02d}.pt"
                    )
                )


                local_features[
                    client_name
                ] = extract_features(
                    local_path
                )


                print(
                    " OK"
                )


            # ------------------------------------------------------
            # CKA FOR EACH LAYER
            # ------------------------------------------------------

            for layer in LAYERS:

                client_client_values = []
                local_global_values = []


                # --------------------------------------------------
                # CLIENT-CLIENT
                # 10 pairs for 5 hospitals
                # --------------------------------------------------

                for client_a, client_b in combinations(
                    CLIENTS,
                    2,
                ):

                    cka_value = linear_cka(

                        local_features[
                            client_a
                        ][
                            layer
                        ],

                        local_features[
                            client_b
                        ][
                            layer
                        ],
                    )


                    client_client_values.append(
                        cka_value
                    )


                    cka_pairwise_rows.append(
                        {
                            "method":
                                method,

                            "seed":
                                seed,

                            "round":
                                round_number,

                            "layer":
                                layer,

                            "comparison_type":
                                "Client-Client",

                            "model_A":
                                client_a,

                            "model_B":
                                client_b,

                            "cka":
                                cka_value,
                        }
                    )


                # --------------------------------------------------
                # LOCAL-GLOBAL
                # 5 comparisons
                # --------------------------------------------------

                for client_name in CLIENTS:

                    cka_value = linear_cka(

                        local_features[
                            client_name
                        ][
                            layer
                        ],

                        global_features[
                            layer
                        ],
                    )


                    local_global_values.append(
                        cka_value
                    )


                    cka_pairwise_rows.append(
                        {
                            "method":
                                method,

                            "seed":
                                seed,

                            "round":
                                round_number,

                            "layer":
                                layer,

                            "comparison_type":
                                "Local-Global",

                            "model_A":
                                client_name,

                            "model_B":
                                "Global",

                            "cka":
                                cka_value,
                        }
                    )


                cc_mean = float(
                    np.mean(
                        client_client_values
                    )
                )


                lg_mean = float(
                    np.mean(
                        local_global_values
                    )
                )


                cka_seed_summary_rows.append(
                    {
                        "method":
                            method,

                        "seed":
                            seed,

                        "round":
                            round_number,

                        "layer":
                            layer,

                        "client_client_cka":
                            cc_mean,

                        "local_global_cka":
                            lg_mean,
                    }
                )


                if layer == "Final":

                    print(
                        f"  Final CC CKA = "
                        f"{cc_mean:.6f}"
                    )

                    print(
                        f"  Final LG CKA = "
                        f"{lg_mean:.6f}"
                    )


            # ------------------------------------------------------
            # FREE ROUND FEATURES
            # ------------------------------------------------------

            del global_features
            del local_features

            cleanup_memory()


# ======================================================================
# 25. SAVE CKA RAW RESULTS
# ======================================================================

cka_pairwise_df = pd.DataFrame(
    cka_pairwise_rows
)

cka_seed_summary_df = pd.DataFrame(
    cka_seed_summary_rows
)


cka_pairwise_df.to_csv(
    OUTPUT_DIR
    / "CKA_pairwise_raw.csv",
    index=False,
)

cka_seed_summary_df.to_csv(
    OUTPUT_DIR
    / "CKA_by_seed_round_layer.csv",
    index=False,
)


# ======================================================================
# 26. CKA MEAN +/- STD ACROSS THREE SEEDS
# ======================================================================

cka_multiseed_rows = []


for method in METHODS:

    for round_number in CKA_ROUNDS:

        for layer in LAYERS:

            subset = cka_seed_summary_df[
                (
                    cka_seed_summary_df[
                        "method"
                    ] == method
                )
                &
                (
                    cka_seed_summary_df[
                        "round"
                    ] == round_number
                )
                &
                (
                    cka_seed_summary_df[
                        "layer"
                    ] == layer
                )
            ]


            cc_values = subset[
                "client_client_cka"
            ].to_numpy(
                dtype=np.float64
            )

            lg_values = subset[
                "local_global_cka"
            ].to_numpy(
                dtype=np.float64
            )


            if len(
                cc_values
            ) != 3:

                raise RuntimeError(
                    "Expected 3 CKA seeds."
                )


            cka_multiseed_rows.append(
                {
                    "method":
                        method,

                    "round":
                        round_number,

                    "layer":
                        layer,

                    "cc_mean":
                        float(
                            np.mean(
                                cc_values
                            )
                        ),

                    "cc_std":
                        float(
                            np.std(
                                cc_values,
                                ddof=1,
                            )
                        ),

                    "lg_mean":
                        float(
                            np.mean(
                                lg_values
                            )
                        ),

                    "lg_std":
                        float(
                            np.std(
                                lg_values,
                                ddof=1,
                            )
                        ),
                }
            )


cka_multiseed_df = pd.DataFrame(
    cka_multiseed_rows
)


cka_multiseed_df.to_csv(
    OUTPUT_DIR
    / "CKA_mean_std_3seeds.csv",
    index=False,
)


# ======================================================================
# 27. PAIRED CKA DELTAS
# MOON - FEDAVG PER SEED
# ======================================================================

cka_delta_rows = []


for seed in SEEDS:

    for round_number in CKA_ROUNDS:

        for layer in LAYERS:

            fedavg = (
                cka_seed_summary_df[
                    (
                        cka_seed_summary_df[
                            "method"
                        ] == "FedAvg"
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "seed"
                        ] == seed
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "round"
                        ] == round_number
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "layer"
                        ] == layer
                    )
                ]
                .iloc[0]
            )


            moon = (
                cka_seed_summary_df[
                    (
                        cka_seed_summary_df[
                            "method"
                        ] == "MOON"
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "seed"
                        ] == seed
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "round"
                        ] == round_number
                    )
                    &
                    (
                        cka_seed_summary_df[
                            "layer"
                        ] == layer
                    )
                ]
                .iloc[0]
            )


            cka_delta_rows.append(
                {
                    "seed":
                        seed,

                    "round":
                        round_number,

                    "layer":
                        layer,

                    "delta_client_client":
                        float(
                            moon[
                                "client_client_cka"
                            ]
                            -
                            fedavg[
                                "client_client_cka"
                            ]
                        ),

                    "delta_local_global":
                        float(
                            moon[
                                "local_global_cka"
                            ]
                            -
                            fedavg[
                                "local_global_cka"
                            ]
                        ),
                }
            )


cka_delta_df = pd.DataFrame(
    cka_delta_rows
)


cka_delta_df.to_csv(
    OUTPUT_DIR
    / "CKA_paired_MOON_minus_FedAvg.csv",
    index=False,
)


# ======================================================================
# 28. PAIRED CKA DELTA MEAN +/- STD
# ======================================================================

cka_delta_summary_rows = []


for round_number in CKA_ROUNDS:

    for layer in LAYERS:

        subset = cka_delta_df[
            (
                cka_delta_df[
                    "round"
                ] == round_number
            )
            &
            (
                cka_delta_df[
                    "layer"
                ] == layer
            )
        ]


        cc_values = subset[
            "delta_client_client"
        ].to_numpy(
            dtype=np.float64
        )

        lg_values = subset[
            "delta_local_global"
        ].to_numpy(
            dtype=np.float64
        )


        cka_delta_summary_rows.append(
            {
                "round":
                    round_number,

                "layer":
                    layer,

                "delta_cc_mean":
                    float(
                        np.mean(
                            cc_values
                        )
                    ),

                "delta_cc_std":
                    float(
                        np.std(
                            cc_values,
                            ddof=1,
                        )
                    ),

                "delta_lg_mean":
                    float(
                        np.mean(
                            lg_values
                        )
                    ),

                "delta_lg_std":
                    float(
                        np.std(
                            lg_values,
                            ddof=1,
                        )
                    ),
            }
        )


cka_delta_summary_df = pd.DataFrame(
    cka_delta_summary_rows
)


cka_delta_summary_df.to_csv(
    OUTPUT_DIR
    / "CKA_paired_delta_mean_std.csv",
    index=False,
)


# ======================================================================
# 29. ROUND 20 LAYERWISE TABLE
# ======================================================================

round20_layerwise_df = (
    cka_multiseed_df[
        cka_multiseed_df[
            "round"
        ] == 20
    ]
    .copy()
)


round20_layerwise_df.to_csv(
    OUTPUT_DIR
    / "Round20_CKA_layerwise_mean_std.csv",
    index=False,
)


round20_delta_df = (
    cka_delta_summary_df[
        cka_delta_summary_df[
            "round"
        ] == 20
    ]
    .copy()
)


round20_delta_df.to_csv(
    OUTPUT_DIR
    / "Round20_MOON_minus_FedAvg_CKA.csv",
    index=False,
)


# ======================================================================
# 30. FINAL REPRESENTATION TRAJECTORY
# ======================================================================

final_trajectory_df = (
    cka_multiseed_df[
        cka_multiseed_df[
            "layer"
        ] == "Final"
    ]
    .copy()
)


final_trajectory_df.to_csv(
    OUTPUT_DIR
    / "Final_representation_CKA_trajectory.csv",
    index=False,
)


# ======================================================================
# 31. HUMAN-READABLE REPORT
# ======================================================================

REPORT_FILE = (
    OUTPUT_DIR
    / "STEP6_14_SUMMARY.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "=" * 80
        + "\n"
    )

    file.write(
        "STEP 6.14 - MULTI-SEED PERFORMANCE + CKA\n"
    )

    file.write(
        "=" * 80
        + "\n\n"
    )


    file.write(
        "FAIR MULTI-SEED PROTOCOL\n"
    )

    file.write(
        "Seeds: 42 matched, 123, 2026\n"
    )

    file.write(
        "Methods: FedAvg, MOON\n"
    )

    file.write(
        "Final test: official PTB-XL fold 10\n"
    )

    file.write(
        "CKA: fixed 500 ECGs from validation fold 9\n"
    )

    file.write(
        "CKA rounds: 1, 2, 5, 10, 15, 20\n\n"
    )


    file.write(
        "=" * 80
        + "\n"
    )

    file.write(
        "FINAL TEST PERFORMANCE - MEAN +/- STD\n"
    )

    file.write(
        "=" * 80
        + "\n"
    )


    for method in METHODS:

        file.write(
            f"\n{method}\n"
        )


        subset = (
            performance_summary_df[
                performance_summary_df[
                    "method"
                ] == method
            ]
        )


        for _, row in subset.iterrows():

            file.write(
                f"{row['metric']:<15}: "
                f"{row['mean']:.6f} "
                f"+/- "
                f"{row['std']:.6f}\n"
            )


    file.write(
        "\n"
        + "=" * 80
        + "\n"
    )

    file.write(
        "PAIRED PERFORMANCE DELTA "
        "(MOON - FEDAVG)\n"
    )

    file.write(
        "=" * 80
        + "\n"
    )


    for _, row in (
        paired_performance_summary_df
        .iterrows()
    ):

        file.write(
            f"{row['metric']:<15}: "
            f"{row['mean_delta']:+.6f} "
            f"+/- "
            f"{row['std_delta']:.6f}\n"
        )


    file.write(
        "\n"
        + "=" * 80
        + "\n"
    )

    file.write(
        "ROUND 20 CKA - MEAN +/- STD\n"
    )

    file.write(
        "=" * 80
        + "\n"
    )


    for method in METHODS:

        file.write(
            f"\n{method}\n"
        )


        subset = round20_layerwise_df[
            round20_layerwise_df[
                "method"
            ] == method
        ]


        for _, row in subset.iterrows():

            file.write(
                f"{row['layer']:<8} | "
                f"CC "
                f"{row['cc_mean']:.6f} "
                f"+/- "
                f"{row['cc_std']:.6f} | "
                f"LG "
                f"{row['lg_mean']:.6f} "
                f"+/- "
                f"{row['lg_std']:.6f}\n"
            )


    file.write(
        "\n"
        + "=" * 80
        + "\n"
    )

    file.write(
        "ROUND 20 CKA DELTA "
        "(MOON - FEDAVG)\n"
    )

    file.write(
        "=" * 80
        + "\n"
    )


    for _, row in (
        round20_delta_df
        .iterrows()
    ):

        file.write(
            f"{row['layer']:<8} | "
            f"Delta CC "
            f"{row['delta_cc_mean']:+.6f} "
            f"+/- "
            f"{row['delta_cc_std']:.6f} | "
            f"Delta LG "
            f"{row['delta_lg_mean']:+.6f} "
            f"+/- "
            f"{row['delta_lg_std']:.6f}\n"
        )


# ======================================================================
# 32. CONSOLE FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("MULTI-SEED FINAL PERFORMANCE")
print("=" * 80)


for method in METHODS:

    print(
        f"\n{method}"
    )

    subset = (
        performance_summary_df[
            performance_summary_df[
                "method"
            ] == method
        ]
    )


    for _, row in subset.iterrows():

        print(
            f"  {row['metric']:<13}: "
            f"{row['mean']:.6f} "
            f"+/- "
            f"{row['std']:.6f}"
        )


print("\n" + "=" * 80)
print("PAIRED PERFORMANCE DELTA: MOON - FEDAVG")
print("=" * 80)


for _, row in (
    paired_performance_summary_df
    .iterrows()
):

    print(
        f"{row['metric']:<15}: "
        f"{row['mean_delta']:+.6f} "
        f"+/- "
        f"{row['std_delta']:.6f}"
    )


print("\n" + "=" * 80)
print("ROUND 20 LAYERWISE CKA DELTA: MOON - FEDAVG")
print("=" * 80)


for _, row in (
    round20_delta_df
    .iterrows()
):

    print(
        f"{row['layer']:<7} | "
        f"CC "
        f"{row['delta_cc_mean']:+.6f} "
        f"+/- "
        f"{row['delta_cc_std']:.6f} | "
        f"LG "
        f"{row['delta_lg_mean']:+.6f} "
        f"+/- "
        f"{row['delta_lg_std']:.6f}"
    )


# ======================================================================
# 33. OUTPUT LIST
# ======================================================================

print("\n" + "=" * 80)
print("STEP 6.14 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nGenerated files:"
)

generated_files = [
    "test_performance_raw_3seeds.csv",
    "test_performance_mean_std.csv",
    "test_per_class_raw_3seeds.csv",
    "test_per_class_mean_std.csv",
    "test_paired_MOON_minus_FedAvg.csv",
    "test_paired_delta_mean_std.csv",
    "CKA_pairwise_raw.csv",
    "CKA_by_seed_round_layer.csv",
    "CKA_mean_std_3seeds.csv",
    "CKA_paired_MOON_minus_FedAvg.csv",
    "CKA_paired_delta_mean_std.csv",
    "Round20_CKA_layerwise_mean_std.csv",
    "Round20_MOON_minus_FedAvg_CKA.csv",
    "Final_representation_CKA_trajectory.csv",
    "STEP6_14_SUMMARY.txt",
]

for filename in generated_files:

    path = (
        OUTPUT_DIR
        / filename
    )

    status = (
        "OK"
        if path.exists()
        else "MISSING"
    )

    print(
        f" - {filename:<45} "
        f"{status}"
    )


print("\nScientific safeguards:")
print(" - Test fold 10 used only for final Round-20 evaluation.")
print(" - CKA dynamics use validation fold 9, not test data.")
print(" - Same fixed 500 ECGs used for every CKA comparison.")
print(" - Seed 42 uses matched micro-batch implementation.")
print(" - Seeds compared: 42, 123, 2026.")
print(" - FedAvg and MOON use identical N2 partition.")
print(" - Linear CKA protocol identical across all models.")

print("=" * 80)