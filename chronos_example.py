"""
Energy price forecasting with Chronos (zero-shot time-series learning).

Example adapted from: https://github.com/amazon-science/chronos-forecasting

Requirements:
    pip install 'pandas[pyarrow]' chronos-forecasting matplotlib
    (GPU optional; uses CPU if CUDA is not available)

Install Chronos:
    pip install chronos-forecasting
"""

import os

import pandas as pd
from chronos import Chronos2Pipeline

# --- Configuration ---
CHRONOS_MODEL = "amazon/chronos-2"
TRAIN_URL = "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
TEST_URL = "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
PREDICTION_LENGTH = 24
CONTEXT_TAIL_LEN = 256
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
ID_COLUMN = "id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"


def main() -> None:
    # Use GPU if available, otherwise CPU
    device = "cuda" if _cuda_available() else "cpu"
    print(f"Training Chronos model on {device}...")
    
    pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)

    # Load historical target and covariates
    context_df = pd.read_parquet(TRAIN_URL)  # The training dataset
    test_df = pd.read_parquet(TEST_URL)      # The test dataset
    future_df = test_df.drop(columns=TARGET_COLUMN) # The test dataset without the target column
    
    print("\nFirst few rows of the context dataframe:")
    print(context_df.head())
    
    print("\nFirst few rows of the test dataframe:")
    print(test_df.head())
    
    print("\nFirst few rows of the future dataframe:")
    print(future_df.head())
    
    # Zero-shot forecast WITH COVARIATES
    pred_df = pipeline.predict_df(
        context_df,                             # The training dataset
        future_df=future_df,                    # The test dataset without the target column
        prediction_length=PREDICTION_LENGTH,    # The number of steps to forecast
        quantile_levels=QUANTILE_LEVELS,         # The quantiles for the probabilistic forecast
        id_column=ID_COLUMN,                    # The column identifying different time series
        timestamp_column=TIMESTAMP_COLUMN,       # The column with datetime information
        target=TARGET_COLUMN,                    # The column with the target values to predict
    )

    # Plot: last context window, forecast, and ground truth
    ts_context = context_df.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN].tail(CONTEXT_TAIL_LEN)
    ts_pred = pred_df.set_index(TIMESTAMP_COLUMN)
    ts_ground_truth = test_df.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]

    try:
        import matplotlib.pyplot as plt

        IMAGE_DIR = "images"
        os.makedirs(IMAGE_DIR, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        ts_context.plot(ax=ax, label="Historical", color="xkcd:azure")
        ts_ground_truth.plot(ax=ax, label="Ground truth", color="xkcd:grass green")
        ts_pred["predictions"].plot(ax=ax, label="Forecast", color="xkcd:violet")
        ax.fill_between(
            ts_pred.index,
            ts_pred["0.1"],
            ts_pred["0.9"],
            alpha=0.4,
            label="80% prediction interval",
            color="xkcd:light lavender",
        )
        ax.set_title("Energy price forecast (Chronos zero-shot)")
        ax.set_xlabel("Timestamp")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, "chronos_forecast.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nImage saved in '{IMAGE_DIR}/chronos_forecast.png'")
    except ImportError:
        print("\n(Install matplotlib to generate the plot.)")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
