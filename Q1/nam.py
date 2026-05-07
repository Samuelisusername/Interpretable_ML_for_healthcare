from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
ENCODED_CSV = ROOT / "heart_encoded.csv"
RAW_CSV = ROOT / "heart.csv"
TARGET = "HeartDisease"
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
HIDDEN_UNITS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_heart_data(input_file: Path, output_file: Path | None = None) -> pd.DataFrame:
	df = pd.read_csv(input_file)

	df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
	df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
	df["ST_Slope"] = df["ST_Slope"].map({"Down": 0, "Flat": 1, "Up": 2})
	df = pd.get_dummies(df, columns=["ChestPainType", "RestingECG"], dtype=int)

	if output_file is not None:
		df.to_csv(output_file, index=False)

	return df


def load_data() -> pd.DataFrame:
	if ENCODED_CSV.exists():
		return pd.read_csv(ENCODED_CSV)
	if RAW_CSV.exists():
		return encode_heart_data(RAW_CSV, ENCODED_CSV)
	raise FileNotFoundError("Could not find heart_encoded.csv or heart.csv.")


def make_loaders(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
	train_dataset = TensorDataset(
		torch.tensor(X_train, dtype=torch.float32),
		torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
	)
	test_dataset = TensorDataset(
		torch.tensor(X_test, dtype=torch.float32),
		torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
	)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
	return train_loader, test_loader


class FeatureNet(nn.Module):
	def __init__(self, hidden_units: int = HIDDEN_UNITS):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(1, hidden_units),
			nn.ReLU(),
			nn.Linear(hidden_units, hidden_units),
			nn.ReLU(),
			nn.Linear(hidden_units, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)


class NAM(nn.Module):
	def __init__(self, n_features: int, hidden_units: int = HIDDEN_UNITS):
		super().__init__()
		self.feature_nets = nn.ModuleList([FeatureNet(hidden_units) for _ in range(n_features)])
		self.bias = nn.Parameter(torch.zeros(1))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		outputs = [net(x[:, i : i + 1]) for i, net in enumerate(self.feature_nets)]
		return torch.sum(torch.cat(outputs, dim=1), dim=1, keepdim=True) + self.bias

	def feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
		outputs = [net(x[:, i : i + 1]) for i, net in enumerate(self.feature_nets)]
		return torch.cat(outputs, dim=1)


def train(model: NAM, train_loader: DataLoader) -> None:
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	criterion = nn.BCEWithLogitsLoss()

	for epoch in range(1, EPOCHS + 1):
		model.train()
		total_loss = 0.0

		for X_batch, y_batch in train_loader:
			X_batch = X_batch.to(DEVICE)
			y_batch = y_batch.to(DEVICE)

			optimizer.zero_grad()
			loss = criterion(model(X_batch), y_batch)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		if epoch == 1 or epoch % 20 == 0:
			print(f"Epoch {epoch:>3}/{EPOCHS}  loss: {total_loss / len(train_loader):.4f}")


def evaluate(model: NAM, test_loader: DataLoader):
	model.eval()
	all_logits = []
	all_labels = []

	with torch.no_grad():
		for X_batch, y_batch in test_loader:
			logits = model(X_batch.to(DEVICE)).cpu()
			all_logits.append(logits)
			all_labels.append(y_batch)

	logits = torch.cat(all_logits)
	labels = torch.cat(all_labels)
	probs = torch.sigmoid(logits)
	preds = (probs >= 0.5).float()

	auroc = roc_auc_score(labels.numpy(), probs.numpy())
	precision, recall, _ = precision_recall_curve(labels.numpy(), probs.numpy())
	auprc = auc(recall, precision)
	f1 = f1_score(labels.numpy(), preds.numpy())
	balanced_acc = balanced_accuracy_score(labels.numpy(), preds.numpy())
	return f1, balanced_acc, auroc, auprc


def feature_grid(values: np.ndarray) -> np.ndarray:
	unique_values = np.unique(values)
	if len(unique_values) <= 8:
		return unique_values.astype(float)

	lower, upper = np.percentile(values, [1, 99])
	return np.linspace(lower, upper, 200)


def plot_feature_curves(model: NAM, scaler: StandardScaler, X_train_raw: np.ndarray, feature_names: list[str]) -> None:
	model.eval()
	n_features = X_train_raw.shape[1]
	n_cols = 4
	n_rows = int(np.ceil(n_features / n_cols))

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), sharey=True)
	axes = np.array(axes).reshape(-1)

	feature_means = scaler.mean_
	feature_scales = scaler.scale_
	baseline = np.zeros((1, n_features), dtype=np.float32)
	curve_data: list[tuple[np.ndarray, list[float]]] = []

	for index, name in enumerate(feature_names):
		x_values = feature_grid(X_train_raw[:, index])
		y_values = []

		for raw_value in x_values:
			sample = baseline.copy()
			sample[0, index] = (raw_value - feature_means[index]) / feature_scales[index]
			tensor_sample = torch.tensor(sample, dtype=torch.float32, device=DEVICE)
			with torch.no_grad():
				contribution = model.feature_contributions(tensor_sample)[0, index].item()
			y_values.append(contribution)

		curve_data.append((x_values, y_values))

	all_y = np.concatenate([np.asarray(y_vals, dtype=np.float32) for _, y_vals in curve_data])
	y_min = float(all_y.min())
	y_max = float(all_y.max())
	y_pad = 0.05 * max(y_max - y_min, 1e-6)

	for index, name in enumerate(feature_names):
		x_values, y_values = curve_data[index]

		axes[index].plot(x_values, y_values, color="#1f77b4", linewidth=2)
		axes[index].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
		axes[index].set_ylim(y_min - y_pad, y_max + y_pad)
		axes[index].set_title(name, fontsize=20)
		if index % n_cols == 0:
			axes[index].set_ylabel("contribution", fontsize=16)
		else:
			axes[index].set_ylabel("")
			axes[index].tick_params(axis="y", labelleft=False)
		axes[index].tick_params(axis="both", labelsize=12)

	for axis in axes[n_features:]:
		axis.remove()

	fig.tight_layout()
	out_path = ROOT / "nam_feature_curves.png"
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved: {out_path.name}")


def main() -> None:
	torch.manual_seed(42)
	np.random.seed(42)

	df = load_data()
	feature_names = [column for column in df.columns if column != TARGET]

	X = df[feature_names].to_numpy(dtype=np.float32)
	y = df[TARGET].to_numpy(dtype=np.float32)

	X_train_raw, X_test_raw, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=42,
		stratify=y,
	)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train_raw)
	X_test = scaler.transform(X_test_raw)

	train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)

	model = NAM(n_features=X_train.shape[1]).to(DEVICE)
	train(model, train_loader)

	f1, balanced_acc, auroc, auprc = evaluate(model, test_loader)
	print(f"F1 score: {f1:.4f}")
	print(f"Balanced accuracy: {balanced_acc:.4f}")
	print(f"AUROC: {auroc:.4f}")
	print(f"AUPRC: {auprc:.4f}")

	plot_feature_curves(model, scaler, X_train_raw, feature_names)


if __name__ == "__main__":
	main()
