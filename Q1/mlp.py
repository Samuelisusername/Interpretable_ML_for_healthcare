import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc
# from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision





# /home/nbalke/jupyter/bin


# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = "pre-processed-HEART.csv"
TARGET     = "HeartDisease"
TEST_SIZE  = 0.2
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=[TARGET]).values.astype("float32")
y = df[TARGET].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def to_tensors(X, y):
    return TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))

train_loader = DataLoader(to_tensors(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(to_tensors(X_test,  y_test),  batch_size=BATCH_SIZE)

# ── Model ────────────────────────────────────────────────────────────────────
in_features = X_train.shape[1]

# MLP(in_channels, hidden_channels) — last element of hidden_channels is the output size
model = MLP(
    in_channels=in_features,
    hidden_channels=[64, 32, 16, 1],
    activation_layer=nn.ReLU,
    dropout=0.2,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

print("now training", flush=True)
# ── Training ─────────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3}/{EPOCHS}  loss: {total_loss / len(train_loader):.4f}", flush=True)

# ── Evaluation ───────────────────────────────────────────────────────────────
model.eval()
all_logits, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(DEVICE)).cpu()
        all_logits.append(logits)
        all_labels.append(y_batch)

logits = torch.cat(all_logits)          # raw logits
probs  = torch.sigmoid(logits)          # probabilities
preds  = (probs >= 0.5).float()         # binary predictions
labels = torch.cat(all_labels)          # ground truth

auroc = roc_auc_score(labels.numpy(), probs.numpy())

precision, recall, _ = precision_recall_curve(labels.numpy(), probs.numpy())
auprc = auc(recall, precision)
f1     = f1_score(labels.numpy(), preds.numpy())
bal_acc = balanced_accuracy_score(labels.numpy(), preds.numpy())

metrics = (
    "── Test Metrics ──────────────────────\n"
    f"  F1 Score            : {f1:.4f}\n"
    f"  AUROC               : {auroc:.4f}\n"
    f"  AUPRC               : {auprc:.4f}\n"
    f"  Balanced Accuracy   : {bal_acc:.4f}\n"
)

print(metrics)

with open("mlp_metrics.txt", "w") as f:
    f.write(metrics)

