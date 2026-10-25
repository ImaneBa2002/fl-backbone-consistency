# ======================================================================
# STEP 6.13 - N2 FEDAVG - SEED 123
# MEMORY-SAFE + CRASH-SAFE + GRADIENT ACCUMULATION
# ======================================================================
#
# Scientific protocol:
#
# Dataset            : PTB-XL
# Task               : TRUE multi-label classification
# Classes            : NORM, MI, STTC, CD, HYP
# Train folds        : 1-8
# Partition          : N2 Extreme Non-IID
# Clients            : 5 simulated hospitals
# Communication      : 20 rounds
# Local epochs       : 2
# Effective batch    : 32
# Micro-batch        : 4
# Optimizer          : Adam
# Learning rate      : 0.001
# Loss               : BCEWithLogitsLoss
# Aggregation        : Weighted FedAvg
#
# IMPORTANT:
# MICRO_BATCH_SIZE = 4 is only a memory implementation detail.
# The effective optimization batch remains 32 through gradient accumulation.
#
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
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

SEED = 123

NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 2

# Scientific effective batch size
EFFECTIVE_BATCH_SIZE = 32

# Only 4 ECGs go through the CNN at one time
MICRO_BATCH_SIZE = 4

if EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
    raise ValueError(
        "EFFECTIVE_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE."
    )

ACCUMULATION_STEPS = (
    EFFECTIVE_BATCH_SIZE
    // MICRO_BATCH_SIZE
)

LR = 0.001


CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name
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

PARTITION_FILE = (
    ROOT
    / "step6_5_1_N2_partition"
    / "ptbxl_noniid_N2_5_hospitals.csv"
)

OUTPUT_DIR = (
    ROOT
    / "step6_13_multiseed"
    / "N2_FedAvg_seed123"
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
    / "training_log.csv"
)

ROUND_SUMMARY_FILE = (
    OUTPUT_DIR
    / "round_summary.csv"
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
# 4. REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ======================================================================
# 5. MEMORY CLEANUP
# ======================================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ======================================================================
# 6. SAFE / ATOMIC TORCH SAVE
# ======================================================================

def atomic_torch_save(
    obj,
    path,
):

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
# 7. RANDOM STATE FOR RESUME
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
# 8. CNN
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
            kernel_size=2
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

        logits = self.fc(x)

        return logits


# ======================================================================
# 9. HEADER
# ======================================================================

print("=" * 80)
print("STEP 6.13 - N2 FEDAVG MULTI-SEED")
print("SEED 123 - MEMORY SAFE")
print("=" * 80)

print(f"Device               : {DEVICE}")
print(f"Seed                 : {SEED}")
print(f"Rounds               : {ROUNDS}")
print(f"Local epochs         : {LOCAL_EPOCHS}")
print(f"Effective batch size : {EFFECTIVE_BATCH_SIZE}")
print(f"Micro-batch size     : {MICRO_BATCH_SIZE}")
print(f"Accumulation steps   : {ACCUMULATION_STEPS}")
print(f"Learning rate        : {LR}")


# ======================================================================
# 10. ARCHITECTURE CHECK
# ======================================================================

temporary_model = ECGCNN(
    NUM_CLASSES
)

parameter_count = sum(
    parameter.numel()
    for parameter
    in temporary_model.parameters()
)

print(
    f"\nCNN parameters: "
    f"{parameter_count}"
)

if parameter_count != 224549:

    raise RuntimeError(
        "CNN architecture mismatch. "
        f"Expected 224549, found {parameter_count}."
    )

del temporary_model

cleanup_memory()


# ======================================================================
# 11. CHECK FILES
# ======================================================================

for file_name, path in {

    "ptbxl_database.csv":
        DATABASE_FILE,

    "scp_statements.csv":
        SCP_FILE,

    "N2 partition":
        PARTITION_FILE,

}.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{file_name} not found:\n{path}"
        )


# ======================================================================
# 12. LOAD PTB-XL DATABASE
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
    "Total ECGs:",
    len(database)
)

if len(database) != 21799:

    raise RuntimeError(
        "Expected 21799 PTB-XL rows."
    )


# ======================================================================
# 13. LOAD SCP DIAGNOSTIC MAPPING
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
        "Expected 44 diagnostic SCP mappings."
    )


# ======================================================================
# 14. LABEL FUNCTIONS
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

            cls = diagnostic_mapping[
                code
            ]

            if (
                cls in CLASS_TO_INDEX
                and
                cls not in classes
            ):

                classes.append(
                    cls
                )

    return classes


def make_multihot(classes):

    y = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for cls in classes:

        y[
            CLASS_TO_INDEX[cls]
        ] = 1.0

    return y


# ======================================================================
# 15. EXTRACT TRUE MULTI-LABEL TARGETS
# ======================================================================

print(
    "\nExtracting labels..."
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
# 16. OFFICIAL TRAIN FOLDS 1-8
# ======================================================================

train_database = usable_database[
    usable_database[
        "strat_fold"
    ].isin(
        range(1, 9)
    )
].copy()


print(
    "Train ECGs:",
    len(train_database)
)

if len(train_database) != 17084:

    raise RuntimeError(
        "Expected 17084 training ECGs."
    )


# ======================================================================
# 17. LOAD N2 PARTITION
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
        "Expected 17084 N2 partition rows."
    )


# ======================================================================
# 18. PATIENT LEAKAGE CHECK
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
# 19. TRAIN/PARTITION ECG ID CHECK
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
        "N2 partition does not exactly "
        "match official training ECGs."
    )


# ======================================================================
# 20. MERGE HOSPITAL ASSIGNMENT
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
        "Merged training size mismatch."
    )


# ======================================================================
# 21. HOSPITAL DISTRIBUTION
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
            f"{client_name}: expected "
            f"{expected}, found {actual}."
        )


# ======================================================================
# 22. LIGHTWEIGHT CLIENT STORAGE
# ======================================================================

client_data = {}


for client_name in CLIENTS:

    client_df = train_data[
        train_data[
            "hospital"
        ] == client_name
    ]

    filenames = (
        client_df[
            "filename_lr"
        ]
        .astype(str)
        .tolist()
    )

    targets = np.stack(
        client_df[
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


# Large pandas objects no longer required during training.

del database
del usable_database
del train_database
del train_data
del partition
del patient_hospital_counts
del scp

cleanup_memory()


# ======================================================================
# 23. LAZY PTB-XL DATASET
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

        filename = self.filenames[
            index
        ]

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
        )

        # wfdb: (1000, 12)
        # CNN:  (12, 1000)
        signal = signal.T

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

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        target = self.targets[
            index
        ]

        x = torch.from_numpy(
            signal
        )

        y = torch.from_numpy(
            target
        )

        return x, y


# ======================================================================
# 24. MODEL STATE COPY
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
# 25. GRADIENT SCALING
#
# BCEWithLogitsLoss(reduction="sum") accumulates gradients over
# micro-batches. Before Adam.step(), gradients are divided by:
#
# effective_samples * NUM_CLASSES
#
# This reproduces BCEWithLogitsLoss(reduction="mean") on the original
# effective batch of 32 samples.
# ======================================================================

def normalize_accumulated_gradients(
    model,
    accumulated_samples,
):

    denominator = float(
        accumulated_samples
        *
        NUM_CLASSES
    )

    if denominator <= 0:

        raise RuntimeError(
            "Invalid gradient denominator."
        )

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameter.grad.div_(
                denominator
            )


# ======================================================================
# 26. LOCAL TRAINING
# ======================================================================

def train_local(
    global_state,
    client_name,
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

    # IMPORTANT:
    # DataLoader only loads MICRO_BATCH_SIZE ECGs at once.
    loader = DataLoader(

        dataset,

        batch_size=MICRO_BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=False,

        drop_last=False,
    )


    model = ECGCNN(
        NUM_CLASSES
    )

    model.load_state_dict(
        global_state,
        strict=True,
    )

    model.to(
        DEVICE
    )

    model.train()


    # SUM is necessary for mathematically correct
    # gradient accumulation.
    criterion = nn.BCEWithLogitsLoss(
        reduction="sum"
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LR,
    )


    running_loss_sum = 0.0
    total_samples = 0

    optimizer_steps = 0


    for local_epoch in range(
        LOCAL_EPOCHS
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_samples = 0
        accumulated_micro_batches = 0


        number_of_micro_batches = len(
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


            logits = model(x)


            loss_sum = criterion(
                logits,
                y,
            )


            # Accumulate SUM gradients.
            loss_sum.backward()


            running_loss_sum += float(
                loss_sum.item()
            )

            total_samples += (
                micro_size
            )

            accumulated_samples += (
                micro_size
            )

            accumulated_micro_batches += 1


            # Release activations immediately.
            del logits
            del loss_sum
            del x
            del y


            is_full_effective_batch = (
                accumulated_micro_batches
                ==
                ACCUMULATION_STEPS
            )

            is_last_micro_batch = (
                micro_index
                ==
                number_of_micro_batches - 1
            )


            if (
                is_full_effective_batch
                or
                is_last_micro_batch
            ):

                normalize_accumulated_gradients(

                    model,

                    accumulated_samples,
                )


                optimizer.step()


                optimizer.zero_grad(
                    set_to_none=True
                )


                optimizer_steps += 1


                accumulated_samples = 0
                accumulated_micro_batches = 0


                # Occasional explicit Python garbage collection.
                # Not done every micro-batch because that would
                # unnecessarily slow CPU training.
                if optimizer_steps % 25 == 0:

                    cleanup_memory()


        cleanup_memory()


    if total_samples == 0:

        raise RuntimeError(
            f"{client_name}: "
            "no samples were trained."
        )


    # BCE mean over samples AND five labels.
    mean_loss = (

        running_loss_sum

        /
        (
            total_samples
            *
            NUM_CLASSES
        )
    )


    local_state = state_to_cpu(
        model
    )


    del optimizer
    del criterion
    del loader
    del dataset
    del model

    cleanup_memory()


    return (
        local_state,
        float(mean_loss),
    )


# ======================================================================
# 27. MEMORY-EFFICIENT WEIGHTED FEDAVG
#
# Instead of keeping 5 local models in RAM simultaneously,
# each local model is immediately added to one accumulator.
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


def add_client_to_accumulator(
    accumulator,
    local_state,
    client_size,
):

    weight = float(
        client_size
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


def finish_weighted_fedavg(
    accumulator,
    total_weight,
):

    if total_weight <= 0:

        raise RuntimeError(
            "FedAvg total weight must be > 0."
        )


    new_global_state = {}


    for key, value in accumulator.items():

        new_global_state[
            key
        ] = (

            value
            /
            float(total_weight)
        )


    return new_global_state


# ======================================================================
# 28. CSV LOGS
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
# 29. CONFIGURATION FILE
# ======================================================================

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "STEP 6.13 - N2 FEDAVG MULTI-SEED\n"
    )

    file.write(
        "=" * 70
        + "\n"
    )

    file.write(
        f"Seed: {SEED}\n"
    )

    file.write(
        f"Device: {DEVICE}\n"
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
        f"Gradient accumulation steps: "
        f"{ACCUMULATION_STEPS}\n"
    )

    file.write(
        f"Learning rate: {LR}\n"
    )

    file.write(
        "Optimizer: Adam\n"
    )

    file.write(
        "Loss: BCEWithLogitsLoss\n"
    )

    file.write(
        "Task: TRUE multi-label\n"
    )

    file.write(
        "Classes: NORM, MI, STTC, CD, HYP\n"
    )

    file.write(
        "Aggregation: Weighted FedAvg\n"
    )

    file.write(
        "Partition: N2 extreme Non-IID\n"
    )

    file.write(
        "Official train folds: 1-8\n"
    )

    file.write(
        "Normalization: training-fold statistics only\n"
    )

    file.write(
        "num_workers: 0\n"
    )

    file.write(
        "pin_memory: False\n"
    )

    file.write(
        "Gradient accumulation: enabled\n"
    )

    file.write(
        "Crash-safe resume: enabled\n"
    )


# ======================================================================
# 30. INITIALIZE / RESUME
# ======================================================================

training_rows = []
round_rows = []


if RESUME_FILE.exists():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESUME STATE FOUND"
    )

    print(
        "=" * 80
    )


    checkpoint = torch.load(

        RESUME_FILE,

        map_location="cpu",

        weights_only=False,
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

    current_round_losses = checkpoint[
        "current_round_losses"
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

    else:

        print(
            "All clients of current round "
            "already completed."
        )


    del checkpoint

    cleanup_memory()


else:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STARTING NEW SEED 123 RUN"
    )

    print(
        "=" * 80
    )


    initial_model = ECGCNN(
        NUM_CLASSES
    )


    global_state = state_to_cpu(
        initial_model
    )


    del initial_model

    cleanup_memory()


    # Initial model before any FL round.
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

    current_round_losses = []


    # Resume point before Round 1 / H1.
    atomic_torch_save(

        {
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

            "current_round_losses":
                current_round_losses,

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
# 31. FEDERATED TRAINING
# ======================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STARTING / RESUMING N2 FEDAVG TRAINING"
)

print(
    "=" * 80
)


EXPECTED_TOTAL_TRAIN = sum(
    EXPECTED_CLIENT_SIZES.values()
)


while current_round <= ROUNDS:

    print(
        f"\n========== "
        f"ROUND {current_round}/{ROUNDS} "
        f"=========="
    )


    # ------------------------------------------------------------------
    # CLIENTS
    # ------------------------------------------------------------------

    for client_index in range(

        next_client_index,

        len(CLIENTS),
    ):

        client_name = CLIENTS[
            client_index
        ]

        client_size = (
            client_data[
                client_name
            ]["size"]
        )


        print(
            f"{client_name}: "
            f"{client_size} ECGs"
        )


        cleanup_memory()


        local_state, local_loss = (
            train_local(

                global_state=
                    global_state,

                client_name=
                    client_name,
            )
        )


        print(
            f"    loss = "
            f"{local_loss:.6f}"
        )


        # --------------------------------------------------------------
        # SAVE LOCAL CLIENT MODEL
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # ADD TO FEDAVG
        # --------------------------------------------------------------

        add_client_to_accumulator(

            accumulator,

            local_state,

            client_size,
        )


        accumulated_weight += (
            client_size
        )


        current_round_losses.append(
            local_loss
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

                "local_loss":
                    local_loss,
            }
        )


        # Local model no longer needed in RAM.
        del local_state

        cleanup_memory()


        save_logs(
            training_rows,
            round_rows,
        )


        # --------------------------------------------------------------
        # CRASH-SAFE CHECKPOINT
        #
        # If the computer stops during the next client,
        # rerunning this script resumes from here.
        # --------------------------------------------------------------

        atomic_torch_save(

            {
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

                "current_round_losses":
                    current_round_losses,

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
    # 32. AGGREGATION
    # ==================================================================

    if (
        accumulated_weight
        !=
        EXPECTED_TOTAL_TRAIN
    ):

        raise RuntimeError(
            f"Round {current_round}: "
            f"expected FedAvg weight "
            f"{EXPECTED_TOTAL_TRAIN}, "
            f"found {accumulated_weight}."
        )


    new_global_state = (
        finish_weighted_fedavg(

            accumulator,

            accumulated_weight,
        )
    )


    del global_state

    global_state = (
        new_global_state
    )

    del new_global_state


    mean_client_loss = float(
        np.mean(
            current_round_losses
        )
    )


    print(
        f"Mean client loss = "
        f"{mean_client_loss:.6f}"
    )


    # ==================================================================
    # 33. SAVE GLOBAL ROUND
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


    round_rows.append(
        {
            "seed":
                SEED,

            "round":
                current_round,

            "mean_client_loss":
                mean_client_loss,

            "total_samples":
                accumulated_weight,
        }
    )


    save_logs(
        training_rows,
        round_rows,
    )


    # ==================================================================
    # 34. NEXT ROUND
    # ==================================================================

    del accumulator

    cleanup_memory()


    current_round += 1


    if current_round <= ROUNDS:

        next_client_index = 0

        accumulator = create_accumulator(
            global_state
        )

        accumulated_weight = 0

        current_round_losses = []


        atomic_torch_save(

            {
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

                "current_round_losses":
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
# 35. FINAL MODEL
# ======================================================================

final_model_path = (

    OUTPUT_DIR
    / "global_final_round_20.pt"
)


atomic_torch_save(
    global_state,
    final_model_path,
)


# ======================================================================
# 36. VERIFY ALL CHECKPOINTS
# ======================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "VERIFYING CHECKPOINTS"
)

print(
    "=" * 80
)


missing_files = []


for round_number in range(
    1,
    ROUNDS + 1,
):

    global_file = (

        OUTPUT_DIR

        / (
            f"global_round_"
            f"{round_number:02d}.pt"
        )
    )


    if not global_file.exists():

        missing_files.append(
            global_file
        )


    for client_name in CLIENTS:

        local_file = (

            OUTPUT_DIR

            / (
                f"local_"
                f"{client_name}_"
                f"round_"
                f"{round_number:02d}.pt"
            )
        )


        if not local_file.exists():

            missing_files.append(
                local_file
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
# 37. COMPLETION FILE
# ======================================================================

with open(
    COMPLETION_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "STEP 6.13 N2 FedAvg seed 123 completed successfully.\n"
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
        "Gradient accumulation: enabled\n"
    )

    file.write(
        "Loss: BCEWithLogitsLoss\n"
    )

    file.write(
        "Aggregation: Weighted FedAvg\n"
    )


# Resume state is no longer needed after complete success.

if RESUME_FILE.exists():

    RESUME_FILE.unlink()


del global_state

cleanup_memory()


# ======================================================================
# 38. DONE
# ======================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STEP 6.13 N2 FEDAVG SEED 123 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 80
)

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
    "  True multi-label     : YES"
)

print(
    "  BCEWithLogitsLoss    : YES"
)

print(
    "  Weighted FedAvg      : YES"
)

print(
    "  Memory-safe          : YES"
)

print(
    "  Crash-safe resume    : YES"
)

print(
    "=" * 80
)