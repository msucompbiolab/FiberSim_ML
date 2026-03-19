import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from readdata import read_tab_delimited_data


def build_training_arrays(
    feature_file_path: str = "data2",
    target_file_path: str = "data2",
    target_start_col_1based: int = 4,
    target_step: int = 3,
    time_downsample_factor: int = 5,
    max_feature_rows: int | None = 900,
    max_target_part: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build arrays from one or more target txt files and matching metadata files:
    - parameter feature: last numeric metadata column from summary_n_vars_1_part_<n>_metadata.txt
    - time values: first column from the target txt data
    - target series: columns 4, 7, 10, ... from each target txt file
    Returns:
    - x_params: shape [n_samples, 1]
    - y_series: shape [n_samples, n_time]
    - t_values: shape [n_time]
    """
    def get_part_number(path: Path) -> int | None:
        suffix = path.stem.rsplit("_part_", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            return int(suffix[1])
        return None

    def sort_key(path: Path) -> tuple[int, str]:
        part_number = get_part_number(path)
        if part_number is not None:
            return part_number, path.name
        return float("inf"), path.name

    def extract_target_series(target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        t_values_local = (
            pd.to_numeric(target_df.iloc[:, 0], errors="coerce").to_numpy().astype(float)
        )

        target_start_col_0based = target_start_col_1based - 1
        target_cols_local = list(range(target_start_col_0based, target_df.shape[1], target_step))
        y_raw_local = (
            target_df.iloc[:, target_cols_local]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
            .astype(float)
        )

        if y_raw_local.shape[0] == t_values_local.shape[0]:
            return y_raw_local.T, t_values_local
        if y_raw_local.shape[1] == t_values_local.shape[0]:
            return y_raw_local, t_values_local

        raise ValueError(
            f"Time length mismatch: time has {t_values_local.shape[0]} rows, "
            f"target matrix shape is {y_raw_local.shape}."
        )

    def get_metadata_file(target_file: Path, metadata_root: Path) -> Path:
        metadata_file = metadata_root / f"{target_file.stem}_metadata.txt"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata file not found for {target_file.name}: {metadata_file}")
        return metadata_file

    def extract_feature_array(metadata_df: pd.DataFrame, metadata_file: Path) -> np.ndarray:
        excluded_cols = {"sim_number", "sample_id", "sim_file"}
        feature_cols = [col for col in metadata_df.columns if col not in excluded_cols]
        if not feature_cols:
            raise ValueError(f"No feature columns found in metadata file: {metadata_file}")

        feature_values = pd.to_numeric(metadata_df[feature_cols[-1]], errors="coerce")
        return feature_values.to_numpy().reshape(-1, 1).astype(float)

    feature_path = Path(feature_file_path)
    target_path = Path(target_file_path)

    if target_path.is_dir():
        target_files = sorted(
            [path for path in target_path.glob("*.txt") if not path.name.endswith("_metadata.txt")],
            key=sort_key,
        )
        if max_target_part is not None:
            target_files = [
                path
                for path in target_files
                if (get_part_number(path) is not None and get_part_number(path) <= max_target_part)
            ]
    elif target_path.is_file():
        target_files = [target_path]
    else:
        raise FileNotFoundError(f"Target path not found: {target_file_path}")

    if not target_files:
        raise FileNotFoundError(f"No .txt target files found in: {target_file_path}")

    if not feature_path.is_dir():
        raise FileNotFoundError(f"Feature metadata directory not found: {feature_file_path}")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    t_values: np.ndarray | None = None

    for target_file in target_files:
        target_df = read_tab_delimited_data(str(target_file))
        y_part, t_part = extract_target_series(target_df)
        metadata_file = get_metadata_file(target_file, feature_path)
        metadata_df = read_tab_delimited_data(str(metadata_file))
        x_part = extract_feature_array(metadata_df, metadata_file)
        print(
            f"{target_file.name}: {t_part.shape[0]} timepoints, "
            f"{y_part.shape[0]} data series"
        )

        if t_values is None:
            t_values = t_part
        elif t_values.shape != t_part.shape or not np.allclose(t_values, t_part, equal_nan=True):
            raise ValueError(
                f"Time values in {target_file} do not match the earlier target files."
            )

        if x_part.shape[0] != y_part.shape[0]:
            raise ValueError(
                f"Metadata mismatch in {metadata_file.name}: features have {x_part.shape[0]} rows, "
                f"targets have {y_part.shape[0]} series."
            )

        x_parts.append(x_part)
        y_parts.append(y_part)

    x_params = np.vstack(x_parts)
    y_series = np.vstack(y_parts)
    assert t_values is not None

    if max_feature_rows is not None:
        x_params = x_params[:max_feature_rows]
        y_series = y_series[:max_feature_rows]

    if x_params.shape[0] != y_series.shape[0]:
        raise ValueError(
            f"Row mismatch: features have {x_params.shape[0]} rows, targets have {y_series.shape[0]} rows."
        )

    valid_sample_mask = np.isfinite(x_params).all(axis=1) & np.isfinite(y_series).all(axis=1)
    x_params = x_params[valid_sample_mask]
    y_series = y_series[valid_sample_mask]

    valid_time_mask = np.isfinite(t_values)
    t_values = t_values[valid_time_mask]
    y_series = y_series[:, valid_time_mask]

    if time_downsample_factor < 1:
        raise ValueError("time_downsample_factor must be >= 1.")
    if time_downsample_factor > 1:
        t_values = t_values[::time_downsample_factor]
        y_series = y_series[:, ::time_downsample_factor]

    return x_params, y_series, t_values


def split_train_eval(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sample-wise arrays into train/eval subsets."""
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples to perform an 80/20 split.")

    n_samples = x.shape[0]
    n_train = max(1, int(n_samples * train_ratio))
    n_train = min(n_train, n_samples - 1)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    train_idx = indices[:n_train]
    eval_idx = indices[n_train:]
    return x[train_idx], y[train_idx], x[eval_idx], y[eval_idx]


def build_point_dataset(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    t_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sample-wise series to point-wise supervised data:
    input features: [parameter, time]
    target: scalar value at that time
    """
    n_samples = x_samples.shape[0]
    n_time = t_values.shape[0]

    x_param = np.repeat(x_samples, n_time, axis=0)
    x_time = np.tile(t_values, n_samples).reshape(-1, 1)
    x_points = np.hstack([x_param, x_time])
    y_points = y_samples.reshape(-1, 1)

    return x_points, y_points


def scale_train_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, MinMaxScaler]:
    """Fit MinMaxScaler on training data and transform train/eval."""
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    x_eval_scaled = x_scaler.transform(x_eval)
    y_eval_scaled = y_scaler.transform(y_eval)

    return (
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
        torch.tensor(x_eval_scaled, dtype=torch.float32),
        torch.tensor(y_eval_scaled, dtype=torch.float32),
        x_scaler,
        y_scaler,
    )


class TimeSeriesRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def plot_eval_predictions(
    t_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "evaluation_timeseries_plot.png",
    max_series: int = 5,
) -> None:
    """Plot predicted vs actual time series for evaluation samples."""
    n_plot = min(max_series, y_true.shape[0])
    if n_plot == 0:
        return

    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        axes[i].plot(t_values, y_true[i], label="Actual")
        axes[i].plot(t_values, y_pred[i], "--", label="Predicted")
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"Evaluation sample {i + 1}")
        axes[i].legend()

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(
    train_losses: list[float],
    eval_losses: list[float],
    output_path: str = "loss_curve.png",
) -> None:
    """Plot training and evaluation loss across epochs."""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, eval_losses, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_cpu(
    feature_file_path: str = "data2",
    target_file_path: str = "data2",
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    time_downsample_factor: int = 5,
) -> None:

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    #else:
    device = torch.device("cpu")
    print("No GPU available. Training will run on CPU.")

    x_params, y_series, t_values = build_training_arrays(
        feature_file_path=feature_file_path,
        target_file_path=target_file_path,
        time_downsample_factor=time_downsample_factor,
    )

    x_train_s, y_train_s, x_eval_s, y_eval_s = split_train_eval(
        x_params, y_series, train_ratio=0.8, seed=42
    )
    print(
        f"Train samples: {x_train_s.shape[0]} | Eval samples: {x_eval_s.shape[0]} | "
        f"Time points: {t_values.shape[0]}"
    )

    x_train_points, y_train_points = build_point_dataset(x_train_s, y_train_s, t_values)
    x_eval_points, y_eval_points = build_point_dataset(x_eval_s, y_eval_s, t_values)

    x_train, y_train, x_eval, y_eval, _, y_scaler = scale_train_eval(
        x_train_points, y_train_points, x_eval_points, y_eval_points
    )

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TimeSeriesRegressor().to(device)
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
            eval_pred_epoch = model(x_eval.to(device))
            eval_loss_epoch = criterion(eval_pred_epoch, y_eval.to(device)).item()
        eval_losses.append(eval_loss_epoch)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss_epoch:.6f}"
            )

    model.eval()
    with torch.no_grad():
        eval_pred = model(x_eval.to(device))
        eval_loss = criterion(eval_pred, y_eval.to(device)).item()
    print(f"Evaluation Loss (MSE): {eval_loss:.6f}")

    y_true_flat = y_scaler.inverse_transform(y_eval.cpu().numpy()).reshape(-1)
    y_pred_flat = y_scaler.inverse_transform(eval_pred.cpu().numpy()).reshape(-1)

    n_eval_samples = y_eval_s.shape[0]
    n_time = t_values.shape[0]
    y_true_series = y_true_flat.reshape(n_eval_samples, n_time)
    y_pred_series = y_pred_flat.reshape(n_eval_samples, n_time)

    plot_eval_predictions(
        t_values=t_values,
        y_true=y_true_series,
        y_pred=y_pred_series,
        output_path="evaluation_timeseries_plot.png",
        max_series=5,
    )
    plot_loss_curves(
        train_losses=train_losses,
        eval_losses=eval_losses,
        output_path="loss_curve.png",
    )

    torch.save(model.state_dict(), "timeseries_model_cpu.pt")
    print(
        "Training complete. Model saved to timeseries_model_cpu.pt, "
        "plots saved to evaluation_timeseries_plot.png and loss_curve.png"
    )


if __name__ == "__main__":
    train_cpu(
        feature_file_path="data2",
        target_file_path="data2",
        epochs=1000,
        time_downsample_factor=50,
    )
