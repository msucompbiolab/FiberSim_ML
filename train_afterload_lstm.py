import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_FILE = "data_afterload/afterload_lstm_sequences.csv"
FEATURE_COLUMNS = [
    "time",
    "m_kinetics_isotype_1_scheme_2_transition_2_parameter_1",
    "m_length",
    "m_length_prev",
]
RAW_FEATURE_COLUMNS = [
    "time",
    "m_kinetics_isotype_1_scheme_2_transition_2_parameter_1",
    "m_length",
]
TARGET_COLUMN = "m_force"
SEQUENCE_ID_COLUMN = "sequence_id"
TIME_STEP_COLUMN = "time_step"


def load_sequence_arrays(
    csv_path: str = SEQUENCE_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ODE-generated afterload trajectories and return:
    - x_sequences: [n_sequences, seq_len, 4]
    - y_sequences: [n_sequences, seq_len, 1]
    - sequence_ids: [n_sequences]
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Sequence file not found: {csv_path}")

    df = pd.read_csv(csv_file)

    required_columns = [SEQUENCE_ID_COLUMN, TIME_STEP_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")

    numeric_columns = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    df = df.sort_values([SEQUENCE_ID_COLUMN, TIME_STEP_COLUMN]).reset_index(drop=True)

    x_sequences: list[np.ndarray] = []
    y_sequences: list[np.ndarray] = []
    sequence_ids: list[int] = []
    expected_sequence_length: int | None = None

    for sequence_id, sequence_df in df.groupby(SEQUENCE_ID_COLUMN, sort=True):
        sequence_df = sequence_df.copy()
        sequence_df["m_length_prev"] = sequence_df["m_length"].shift(1)
        sequence_df["m_length_prev"] = sequence_df["m_length_prev"].fillna(sequence_df["m_length"])

        x_seq = sequence_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_seq = sequence_df[[TARGET_COLUMN]].to_numpy(dtype=float)

        if expected_sequence_length is None:
            expected_sequence_length = x_seq.shape[0]
        elif x_seq.shape[0] != expected_sequence_length:
            raise ValueError(
                "All sequences must have the same length for this training script. "
                f"Expected {expected_sequence_length}, got {x_seq.shape[0]} for sequence {sequence_id}."
            )

        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
        sequence_ids.append(sequence_id)

    if not x_sequences:
        raise ValueError(f"No valid sequences found in {csv_path}")

    return (
        np.stack(x_sequences),
        np.stack(y_sequences),
        np.asarray(sequence_ids),
    )


def get_train_eval_indices(
    n_sequences: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_sequences < 2:
        raise ValueError("Need at least 2 sequences for a train/eval split.")

    n_train = max(1, int(n_sequences * train_ratio))
    n_train = min(n_train, n_sequences - 1)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_sequences)
    return indices[:n_train], indices[n_train:]


def scale_train_eval_sequences(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, MinMaxScaler]:
    n_features = x_train.shape[-1]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_2d = x_train.reshape(-1, n_features)
    x_eval_2d = x_eval.reshape(-1, n_features)
    y_train_2d = y_train.reshape(-1, 1)
    y_eval_2d = y_eval.reshape(-1, 1)

    x_train_scaled = x_scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_eval_scaled = x_scaler.transform(x_eval_2d).reshape(x_eval.shape)
    y_train_scaled = y_scaler.fit_transform(y_train_2d).reshape(y_train.shape)
    y_eval_scaled = y_scaler.transform(y_eval_2d).reshape(y_eval.shape)

    return (
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
        torch.tensor(x_eval_scaled, dtype=torch.float32),
        torch.tensor(y_eval_scaled, dtype=torch.float32),
        x_scaler,
        y_scaler,
    )


class AfterloadLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.readout(output)


def plot_loss_curves(
    train_losses: list[float],
    eval_losses: list[float],
    output_path: str = "afterload_lstm_loss_curve.png",
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, eval_losses, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Afterload LSTM Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_eval_predictions(
    x_eval_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "afterload_lstm_eval_predictions.png",
    max_sequences: int = 4,
) -> None:
    n_plot = min(max_sequences, y_true.shape[0])
    if n_plot == 0:
        return

    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]

    time_values = x_eval_raw[:, :, 0]
    kinetics_values = x_eval_raw[:, 0, 1]
    length_values = x_eval_raw[:, 0, 2]

    for i in range(n_plot):
        axes[i].plot(time_values[i], y_true[i, :, 0], label="Actual m_force")
        axes[i].plot(time_values[i], y_pred[i, :, 0], "--", label="Predicted m_force")
        axes[i].set_ylabel("m_force")
        axes[i].set_title(
            f"Eval sequence {i + 1} | "
            f"m_kinetics={kinetics_values[i]:.4g}, m_length={length_values[i]:.4g}"
        )
        axes[i].legend()

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    x_scaler: MinMaxScaler,
    y_scaler: MinMaxScaler,
    checkpoint_path: str,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    train_ratio: float,
    split_seed: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        checkpoint_path,
    )


def train_lstm_afterload(
    csv_path: str = SEQUENCE_FILE,
    epochs: int = 150,
    batch_size: int = 8,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    train_ratio: float = 0.8,
    split_seed: int = 42,
    checkpoint_path: str = "afterload_lstm_model.pt",
    loss_plot_path: str = "afterload_lstm_loss_curve.png",
    eval_plot_path: str = "afterload_lstm_eval_predictions.png",
) -> None:
    device = torch.device("cpu")
    print("Training LSTM on CPU.")

    x_sequences, y_sequences, sequence_ids = load_sequence_arrays(csv_path=csv_path)
    print(
        f"Loaded {x_sequences.shape[0]} sequences | "
        f"sequence length {x_sequences.shape[1]} | features {x_sequences.shape[2]}"
    )

    train_idx, eval_idx = get_train_eval_indices(
        n_sequences=x_sequences.shape[0],
        train_ratio=train_ratio,
        seed=split_seed,
    )
    x_train_raw = x_sequences[train_idx]
    y_train_raw = y_sequences[train_idx]
    x_eval_raw = x_sequences[eval_idx]
    y_eval_raw = y_sequences[eval_idx]

    x_train, y_train, x_eval, y_eval, x_scaler, y_scaler = scale_train_eval_sequences(
        x_train_raw,
        y_train_raw,
        x_eval_raw,
        y_eval_raw,
    )

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AfterloadLSTM(
        input_size=x_sequences.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    eval_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(dataset) if len(dataset) > 0 else 0.0
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            eval_pred = model(x_eval.to(device))
            eval_loss = criterion(eval_pred, y_eval.to(device)).item()
        eval_losses.append(eval_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss:.6f}"
            )

    model.eval()
    with torch.no_grad():
        eval_pred_scaled = model(x_eval.to(device)).cpu().numpy()

    y_eval_pred = y_scaler.inverse_transform(eval_pred_scaled.reshape(-1, 1)).reshape(y_eval_raw.shape)
    final_eval_mse = np.mean((y_eval_pred - y_eval_raw) ** 2)
    print(f"Final validation loss (unscaled MSE): {final_eval_mse:.6f}")
    print(f"Validation sequence IDs: {sequence_ids[eval_idx].tolist()}")

    plot_loss_curves(
        train_losses=train_losses,
        eval_losses=eval_losses,
        output_path=loss_plot_path,
    )
    plot_eval_predictions(
        x_eval_raw=x_eval_raw,
        y_true=y_eval_raw,
        y_pred=y_eval_pred,
        output_path=eval_plot_path,
    )
    save_checkpoint(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        checkpoint_path=checkpoint_path,
        train_indices=train_idx,
        eval_indices=eval_idx,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    print(
        f"Saved checkpoint to {checkpoint_path}, "
        f"loss plot to {loss_plot_path}, "
        f"and evaluation plot to {eval_plot_path}"
    )


if __name__ == "__main__":
    train_lstm_afterload()
