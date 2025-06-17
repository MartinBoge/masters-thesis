import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from analysis.data.gold.utils import GoldConfig


def load_and_split_data(detrend: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Reads forecasting dataset and returns train, validation, and test dataframes.
    """
    df = pl.read_parquet(f"{GoldConfig.SAVE_DIR}/forecasting_dataset.parquet")

    total_rows = len(df)

    n_test = round(total_rows * 0.15)
    n_validation = round(total_rows * 0.15)
    n_train = total_rows - n_test - n_validation

    df_train = df.slice(0, n_train)  # Start at 0, take n_train rows
    df_validation = df.slice(n_train, n_validation)  # Start at n_train, take n_validation rows
    df_test = df.slice(n_train + n_validation, n_test)  # Start at n_train+n_validation, take n_test rows

    print("-------- Data Splits --------")
    print(f"Original:   {total_rows:,} rows")
    print(f"Train:      {n_train:,} rows ({n_train / total_rows:.0%})")
    print(f"Validation:  {n_validation:,} rows ({n_validation / total_rows:.0%})")
    print(f"Test:        {n_test:,} rows ({n_test / total_rows:.0%})")

    assert n_train == df_train.shape[0]
    assert n_validation == df_validation.shape[0]
    assert n_test == df_test.shape[0]

    if detrend:
        df_train, df_validation, df_test = detrend_data(df_train, df_validation, df_test)

    return df_train, df_validation, df_test


def detrend_data(
    df_train: pl.DataFrame, df_validation: pl.DataFrame, df_test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Detrends the target variable based on the training data and applies
    the same transformation to validation and test sets.
    """
    # Get the target values as numpy arrays
    y_train = df_train["y"].to_numpy()
    X_train = df_train["t_hours_since_start"].to_numpy().reshape(-1, 1)

    # Fit linear regression on training data only
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate trend using the model fitted on training data
    train_trend = model.predict(df_train["t_hours_since_start"].to_numpy().reshape(-1, 1))
    val_trend = model.predict(df_validation["t_hours_since_start"].to_numpy().reshape(-1, 1))
    test_trend = model.predict(df_test["t_hours_since_start"].to_numpy().reshape(-1, 1))

    # Add detrended values to each dataframe
    df_train = df_train.with_columns((pl.col("y") - pl.Series(train_trend)).alias("y_detrended"))

    df_validation = df_validation.with_columns((pl.col("y") - pl.Series(val_trend)).alias("y_detrended"))

    df_test = df_test.with_columns((pl.col("y") - pl.Series(test_trend)).alias("y_detrended"))

    print("Data detrended using train trend")

    return df_train, df_validation, df_test


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray, verbose: bool = True) -> dict:
    if predictions.shape != actuals.shape:
        raise ValueError("Arrays for predictions and actuals must be same shape")

    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
        actuals = actuals.flatten()

    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

    if verbose:
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    return metrics


def plot_residuals(predictions: np.ndarray, actuals: np.ndarray, title: str, save_name: str | None = None) -> None:
    residuals = actuals - predictions

    if len(residuals.shape) == 2:
        residuals = residuals.flatten()

    # Plot the residuals over time
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label="Residuals", color="blue")
    plt.axhline(0, color="black", linewidth=1)  # Line at 0
    # plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.legend()
    if save_name:
        plt.savefig(f"{save_name}_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="orange", edgecolor="black", alpha=0.7)
    # plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    if save_name:
        plt.savefig(f"{save_name}_frequency.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions_vs_actuals(predictions: np.ndarray, actuals: np.ndarray, num_samples: int = 3) -> None:
    """
    Plot predictions vs actuals for a few randomly selected samples
    Works with both 2D arrays from naive model and 3D arrays from LSTM
    """
    plt.figure(figsize=(15, 10))

    # Randomly select indices (without replacement)
    total_samples = predictions.shape[0]
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    for idx, i in enumerate(random_indices):
        plt.subplot(num_samples, 1, idx + 1)

        if len(predictions.shape) == 3:
            # LSTM case: 3D arrays [batch, seq, features]
            pred_sample = predictions[i, :, 0]
            act_sample = actuals[i, :, 0]
        else:
            # Naive model case: 2D arrays [samples, horizon]
            pred_sample = predictions[i]
            act_sample = actuals[i]

        plt.plot(act_sample, label="Actual", color="blue")
        plt.plot(pred_sample, label="Predicted", color="red", linestyle="--")
        # plt.title(f"Predictions vs Actuals - Sample {i + 1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actuals_v2(
    predictions: np.ndarray, actuals: np.ndarray, num_samples: int = 3, seq_len: int = 24, save_name: str | None = None
) -> None:
    """
    Plot predictions vs actuals for randomly selected samples
    Works with flat arrays from inverse_transform
    """
    plt.figure(figsize=(15, 10))

    # Calculate total number of complete sequences
    total_sequences = len(predictions) // seq_len

    # Randomly select sequence indices (without replacement)
    random_indices = random.sample(range(total_sequences), min(num_samples, total_sequences))

    for idx, seq_idx in enumerate(random_indices):
        plt.subplot(num_samples, 1, idx + 1)

        # Extract the sequence from flattened array
        start_idx = seq_idx * seq_len
        end_idx = start_idx + seq_len

        pred_sample = predictions[start_idx:end_idx]
        act_sample = actuals[start_idx:end_idx]

        plt.plot(act_sample, label="Actual", color="blue")
        plt.plot(pred_sample, label="Predicted", color="red", linestyle="--")
        # plt.title(f"Predictions vs Actuals - Sample {seq_idx + 1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions_vs_actuals_v3(predictions: np.ndarray, actuals: np.ndarray, num_samples: int = 3) -> None:
    """
    Plot predictions vs actuals for a few randomly selected samples
    Works with both 1D, 2D, and 3D arrays.
    """
    if predictions.shape != actuals.shape:
        raise ValueError("Arrays for predictions and actuals must be same shape")

    total_samples = predictions.shape[0]
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    plt.figure(figsize=(15, 10))

    for idx, i in enumerate(random_indices):
        plt.subplot(num_samples, 1, idx + 1)

        if len(predictions.shape) == 1:
            # For 1D arrays, there's only one point to plot (not a time series)
            pred_sample = predictions[i]
            act_sample = actuals[i]
            # Plot as point or bar since it's a single value
            plt.bar([0, 1], [act_sample, pred_sample], color=["blue", "red"])
            plt.xticks([0, 1], ["Actual", "Predicted"])
            plt.ylabel("Value")
        elif len(predictions.shape) == 2:
            # For 2D arrays, we plot each sample as a time series
            pred_sample = predictions[i]
            act_sample = actuals[i]
            x = range(len(pred_sample))
            plt.plot(x, act_sample, label="Actual", color="blue")
            plt.plot(x, pred_sample, label="Predicted", color="red", linestyle="--")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.legend()

        elif len(predictions.shape) == 3:
            # For 3D arrays, we plot the data with multiple channels
            # Option 1: Plot just the first channel
            pred_sample = predictions[i, :, 0]
            act_sample = actuals[i, :, 0]
            plt.plot(act_sample, label="Actual (Channel 0)", color="blue")
            plt.plot(pred_sample, label="Predicted (Channel 0)", color="red", linestyle="--")

            # Option 2: You could also add visualization for additional channels
            # if predictions.shape[2] > 1:
            #     for c in range(1, min(3, predictions.shape[2])):  # Plot up to 3 channels
            #         plt.plot(predictions[i, :, c], label=f"Predicted (Ch {c})",
            #                 color=f"C{c+1}", linestyle="--")
            #         plt.plot(actuals[i, :, c], label=f"Actual (Ch {c})",
            #                 color=f"C{c+1}")

            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
        else:
            raise ValueError("Input arrays have invalid shape")

        plt.plot(act_sample, label="Actual", color="blue")
        plt.plot(pred_sample, label="Predicted", color="red", linestyle="--")
        # plt.title(f"Predictions vs Actuals - Sample {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
