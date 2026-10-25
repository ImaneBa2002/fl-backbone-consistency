# ======================================================================
# STEP 6.16
# CENTRALIZED BASELINE - 3 MATCHED SEEDS
# ======================================================================

from pathlib import Path
import ast
import gc
import json
import random

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

SEEDS = [42, 123, 2026]
EPOCHS = 40
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-3
THRESHOLD = 0.5

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATABASE_FILE = ROOT / "ptbxl_database.csv"
SCP_FILE = ROOT / "scp_statements.csv"
OUTPUT_ROOT = ROOT / "step6_16_centralized_3seeds"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
STEP614_DIR = ROOT / "step6_14_multiseed_evaluation_CKA"

TRAIN_MEAN = np.array([
    -0.001747, -0.001521,  0.000225,  0.001617,
    -0.000932, -0.000625,  0.000178, -0.000952,
    -0.001565, -0.001356, -0.000800, -0.002424,
], dtype=np.float32)

TRAIN_STD = np.array([
    0.160010, 0.159230, 0.159972, 0.138148,
    0.138767, 0.137978, 0.234350, 0.334216,
    0.328469, 0.292764, 0.272069, 0.280968,
], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(12, 32, 7, padding=3), nn.GroupNorm(8, 32))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 7, padding=3), nn.GroupNorm(8, 64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.GroupNorm(8, 128))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2), nn.GroupNorm(8, 256))
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


print("=" * 80)
print("STEP 6.16 - CENTRALIZED BASELINE - 3 MATCHED SEEDS")
print("=" * 80)
print(f"Device               : {DEVICE}")
print(f"Seeds                : {SEEDS}")
print(f"Epochs               : {EPOCHS}")
print(f"Effective batch size : {EFFECTIVE_BATCH_SIZE}")
print(f"Micro-batch size     : {MICRO_BATCH_SIZE}")
print(f"Accumulation steps   : {ACCUMULATION_STEPS}")
print(f"Learning rate        : {LEARNING_RATE}")
print(f"Classes              : {CLASS_NAMES}")

_tmp = ECGCNN(NUM_CLASSES)
parameter_count = sum(p.numel() for p in _tmp.parameters())
print(f"CNN parameters       : {parameter_count}")
if parameter_count != 224549:
    raise RuntimeError(f"CNN architecture mismatch: {parameter_count} != 224549")
del _tmp
cleanup_memory()

for name, path in {"ptbxl_database.csv": DATABASE_FILE, "scp_statements.csv": SCP_FILE}.items():
    status = "FOUND" if path.exists() else "MISSING"
    print(f"{name:<25}: {status}")
    if not path.exists():
        raise FileNotFoundError(path)

print("\nLoading PTB-XL metadata...")
database = pd.read_csv(DATABASE_FILE)
database["ecg_id"] = database["ecg_id"].astype(int)
if len(database) != 21799:
    raise RuntimeError(f"Expected 21799 rows, got {len(database)}")

scp = pd.read_csv(SCP_FILE, index_col=0)
diagnostic_mapping = {}
for code, row in scp.iterrows():
    diagnostic_class = row.get("diagnostic_class")
    if pd.notna(diagnostic_class):
        diagnostic_class = str(diagnostic_class).strip()
        if diagnostic_class in CLASS_NAMES:
            diagnostic_mapping[str(code)] = diagnostic_class

print(f"Database rows        : {len(database)}")
print(f"Diagnostic mappings : {len(diagnostic_mapping)}")
if len(diagnostic_mapping) != 44:
    raise RuntimeError(f"Expected 44 diagnostic mappings, got {len(diagnostic_mapping)}")


def parse_scp_codes(raw):
    if isinstance(raw, dict):
        return raw
    if pd.isna(raw):
        return {}
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def extract_classes(raw):
    codes = parse_scp_codes(raw)
    classes = []
    for code in codes.keys():
        code = str(code)
        if code in diagnostic_mapping:
            class_name = diagnostic_mapping[code]
            if class_name in CLASS_TO_INDEX and class_name not in classes:
                classes.append(class_name)
    return classes


def make_multihot(classes):
    target = np.zeros(NUM_CLASSES, dtype=np.float32)
    for class_name in classes:
        target[CLASS_TO_INDEX[class_name]] = 1.0
    return target


print("Building TRUE multi-label targets...")
database["target_classes"] = database["scp_codes"].apply(extract_classes)
database["target"] = database["target_classes"].apply(make_multihot)
usable_database = database[database["target_classes"].apply(len) > 0].copy()

print(f"Usable ECGs          : {len(usable_database)}")
print(f"Excluded ECGs        : {len(database) - len(usable_database)}")
if len(usable_database) != 21388:
    raise RuntimeError(f"Expected 21388 usable ECGs, got {len(usable_database)}")

train_df = usable_database[usable_database["strat_fold"].between(1, 8)].copy()
val_df = usable_database[usable_database["strat_fold"] == 9].copy()
test_df = usable_database[usable_database["strat_fold"] == 10].copy()

print("\nOfficial usable split:")
print(f"Train: {len(train_df)}")
print(f"Val  : {len(val_df)}")
print(f"Test : {len(test_df)}")
if len(train_df) != 17084 or len(val_df) != 2146 or len(test_df) != 2158:
    raise RuntimeError("Official split sizes do not match expected values.")


class PTBXLDataset(Dataset):
    def __init__(self, dataframe):
        self.filenames = dataframe["filename_lr"].astype(str).tolist()
        self.targets = np.stack(dataframe["target"].to_numpy()).astype(np.float32)
        self.ecg_ids = dataframe["ecg_id"].astype(int).to_numpy()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        signal, _ = wfdb.rdsamp(str(ROOT / filename))
        signal = signal.astype(np.float32, copy=False).T
        if signal.shape != (12, 1000):
            raise RuntimeError(f"Unexpected ECG shape {signal.shape}: {filename}")
        signal = (signal - TRAIN_MEAN[:, None]) / TRAIN_STD[:, None]
        signal = np.array(signal, dtype=np.float32, copy=True)
        target = np.array(self.targets[index], dtype=np.float32, copy=True)
        return torch.from_numpy(signal), torch.from_numpy(target), int(self.ecg_ids[index])


train_dataset = PTBXLDataset(train_df)
test_dataset = PTBXLDataset(test_df)


def save_training_state(path, model, optimizer, completed_epoch, history):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": int(completed_epoch),
        "history": history,
    }, path)


def load_training_state(path, model, optimizer):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["completed_epoch"]), list(payload.get("history", []))


def train_one_epoch(model, optimizer, loader):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_task_sum = 0.0
    total_samples = 0
    accumulated_samples = 0
    number_of_batches = len(loader)

    for batch_index, (x, y, _) in enumerate(loader, start=1):
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)
        logits = model(x)
        task_sum = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / NUM_CLASSES
        task_sum.backward()

        batch_samples = int(y.shape[0])
        accumulated_samples += batch_samples
        total_samples += batch_samples
        total_task_sum += float(task_sum.item())

        if accumulated_samples >= EFFECTIVE_BATCH_SIZE or batch_index == number_of_batches:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0

        del x, y, logits, task_sum

    return float(total_task_sum / total_samples)


def evaluate_final_test(model):
    loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model.eval()
    logits_list = []
    targets_list = []
    total_bce_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            logits = model(x)
            bce_sum = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            total_bce_sum += float(bce_sum.item())
            total_elements += int(y.numel())
            logits_list.append(logits.detach().cpu().numpy())
            targets_list.append(y.detach().cpu().numpy())
            del x, y, logits, bce_sum

    logits_all = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0).astype(np.int32)
    probabilities = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred = (probabilities >= THRESHOLD).astype(np.int32)

    loss = total_bce_sum / total_elements
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_auroc = np.array([
        roc_auc_score(y_true[:, i], probabilities[:, i])
        for i in range(NUM_CLASSES)
    ], dtype=float)
    macro_auroc = float(np.mean(per_class_auroc))
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

    return {
        "loss": float(loss),
        "macro_f1": float(macro_f1),
        "macro_auroc": float(macro_auroc),
        "exact_match": float(exact_match),
        "per_class_f1": np.asarray(per_class_f1, dtype=float),
        "per_class_auroc": per_class_auroc,
    }


all_test_rows = []
all_per_class_rows = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"CENTRALIZED SEED {seed}")
    print("=" * 80)

    set_seed(seed)
    seed_dir = OUTPUT_ROOT / f"Centralized_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    resume_file = seed_dir / "resume_state.pt"
    final_model_file = seed_dir / "centralized_epoch_40.pt"
    history_file = seed_dir / "training_history.csv"
    result_file = seed_dir / "final_test_metrics.json"

    model = ECGCNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    completed_epoch = 0
    history = []

    if resume_file.exists() and not final_model_file.exists():
        completed_epoch, history = load_training_state(resume_file, model, optimizer)
        model.to(DEVICE)
        print(f"Resuming from completed epoch {completed_epoch}")
    elif final_model_file.exists():
        print("Final model already exists. Training will be skipped.")
        try:
            state = torch.load(final_model_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(final_model_file, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(DEVICE)
        completed_epoch = EPOCHS

    if completed_epoch < EPOCHS:
        for epoch in range(completed_epoch + 1, EPOCHS + 1):
            generator = torch.Generator()
            generator.manual_seed(seed * 1000 + epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=MICRO_BATCH_SIZE,
                shuffle=True,
                generator=generator,
                num_workers=0,
                pin_memory=False,
            )

            mean_loss = train_one_epoch(model, optimizer, train_loader)
            history.append({
                "seed": seed,
                "epoch": epoch,
                "train_bce_loss": mean_loss,
            })
            pd.DataFrame(history).to_csv(history_file, index=False)
            save_training_state(resume_file, model, optimizer, epoch, history)

            if epoch in [1, 2, 5, 10, 20, 30, 40]:
                torch.save(model.state_dict(), seed_dir / f"centralized_epoch_{epoch:02d}.pt")

            print(f"Epoch {epoch:02d}/{EPOCHS} | train BCE = {mean_loss:.6f}")
            del train_loader
            cleanup_memory()

    torch.save(model.state_dict(), final_model_file)
    print(f"Final model saved: {final_model_file.name}")

    if result_file.exists():
        print("Final test metrics already exist; loading them.")
        with open(result_file, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        metrics = {
            "loss": float(saved_metrics["loss"]),
            "macro_f1": float(saved_metrics["macro_f1"]),
            "macro_auroc": float(saved_metrics["macro_auroc"]),
            "exact_match": float(saved_metrics["exact_match"]),
            "per_class_f1": np.asarray(saved_metrics["per_class_f1"], dtype=float),
            "per_class_auroc": np.asarray(saved_metrics["per_class_auroc"], dtype=float),
        }
    else:
        print("Evaluating held-out test fold 10 once...")
        metrics = evaluate_final_test(model)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "loss": metrics["loss"],
                "macro_f1": metrics["macro_f1"],
                "macro_auroc": metrics["macro_auroc"],
                "exact_match": metrics["exact_match"],
                "per_class_f1": metrics["per_class_f1"].tolist(),
                "per_class_auroc": metrics["per_class_auroc"].tolist(),
            }, f, indent=2)

    print(f"Test Loss        = {metrics['loss']:.6f}")
    print(f"Test Macro-F1    = {metrics['macro_f1']:.6f}")
    print(f"Test Macro-AUROC = {metrics['macro_auroc']:.6f}")
    print(f"Test Exact Match = {metrics['exact_match']:.6f}")

    all_test_rows.append({
        "method": "Centralized",
        "seed": seed,
        "loss": metrics["loss"],
        "macro_f1": metrics["macro_f1"],
        "macro_auroc": metrics["macro_auroc"],
        "exact_match": metrics["exact_match"],
    })

    for class_index, class_name in enumerate(CLASS_NAMES):
        all_per_class_rows.append({
            "method": "Centralized",
            "seed": seed,
            "class": class_name,
            "f1": float(metrics["per_class_f1"][class_index]),
            "auroc": float(metrics["per_class_auroc"][class_index]),
        })

    del model, optimizer
    cleanup_memory()


test_results_df = pd.DataFrame(all_test_rows)
per_class_results_df = pd.DataFrame(all_per_class_rows)

test_results_df.to_csv(OUTPUT_ROOT / "centralized_test_performance_raw_3seeds.csv", index=False)
per_class_results_df.to_csv(OUTPUT_ROOT / "centralized_test_per_class_raw_3seeds.csv", index=False)

summary_rows = []
for metric in ["loss", "macro_f1", "macro_auroc", "exact_match"]:
    values = test_results_df[metric].to_numpy(dtype=float)
    summary_rows.append({
        "method": "Centralized",
        "metric": metric,
        "n_seeds": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_ROOT / "centralized_test_performance_mean_std.csv", index=False)

per_class_summary_rows = []
for class_name in CLASS_NAMES:
    subset = per_class_results_df[per_class_results_df["class"] == class_name]
    for metric in ["f1", "auroc"]:
        values = subset[metric].to_numpy(dtype=float)
        per_class_summary_rows.append({
            "method": "Centralized",
            "class": class_name,
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        })

per_class_summary_df = pd.DataFrame(per_class_summary_rows)
per_class_summary_df.to_csv(OUTPUT_ROOT / "centralized_test_per_class_mean_std.csv", index=False)

fed_file = STEP614_DIR / "test_performance_mean_std.csv"
if fed_file.exists():
    federated_summary = pd.read_csv(fed_file)
    final_three_way = pd.concat([summary_df, federated_summary], ignore_index=True)
    final_three_way["method"] = pd.Categorical(
        final_three_way["method"],
        categories=["Centralized", "FedAvg", "MOON"],
        ordered=True,
    )
    final_three_way = final_three_way.sort_values(["metric", "method"])
    final_three_way.to_csv(
        OUTPUT_ROOT / "final_performance_Centralized_FedAvg_MOON.csv",
        index=False,
    )
    print("\nCreated final 3-way performance table.")
else:
    print("\nStep 6.14 summary not found. Centralized results were still saved correctly.")

report_file = OUTPUT_ROOT / "STEP6_16_SUMMARY.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 6.16 - CENTRALIZED BASELINE - 3 MATCHED SEEDS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Seeds: 42, 123, 2026\n")
    f.write("Epochs: 40\n")
    f.write("Effective batch: 32\n")
    f.write("Micro-batch: 4\n")
    f.write("Accumulation: 8\n")
    f.write("Optimizer: Adam\n")
    f.write("Learning rate: 0.001\n")
    f.write("True multi-label: YES\n")
    f.write("Official test fold 10 used only after final epoch.\n\n")
    f.write("FINAL CENTRALIZED TEST PERFORMANCE - MEAN +/- STD\n")
    for _, row in summary_df.iterrows():
        f.write(f"{row['metric']:<15}: {row['mean']:.6f} +/- {row['std']:.6f}\n")

print("\n" + "=" * 80)
print("CENTRALIZED FINAL PERFORMANCE - 3 SEEDS")
print("=" * 80)
for _, row in summary_df.iterrows():
    print(f"{row['metric']:<15}: {row['mean']:.6f} +/- {row['std']:.6f}")

print("\nGenerated:")
print(" - centralized_test_performance_raw_3seeds.csv")
print(" - centralized_test_performance_mean_std.csv")
print(" - centralized_test_per_class_raw_3seeds.csv")
print(" - centralized_test_per_class_mean_std.csv")
print(" - STEP6_16_SUMMARY.txt")
if (OUTPUT_ROOT / "final_performance_Centralized_FedAvg_MOON.csv").exists():
    print(" - final_performance_Centralized_FedAvg_MOON.csv")

print("\nScientific safeguards:")
print(" - Same CNN architecture: 224,549 parameters.")
print(" - Same true multi-label PTB-XL target.")
print(" - Same train-only normalization.")
print(" - Same seeds: 42, 123, 2026.")
print(" - Same effective batch: 32.")
print(" - Same sample-exposure budget: 40 passes.")
print(" - Test fold 10 evaluated only after final epoch.")
print(" - mean +/- std uses sample std (ddof=1).")

print("\nSTEP 6.16 COMPLETED SUCCESSFULLY")
print("=" * 80)
