import os

import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

MODEL_NAME = "dolphin-fin-segmentation"

# Наші можливі назви Mask mAP50-95 у MLflow.
# Спочатку беремо explicit validation metric.
METRIC_CANDIDATES = [
    "val_metrics/mAP50-95_M_",
    "metrics/mAP50-95_M_",
    "val_metrics/mAP50-95_M",
    "metrics/mAP50-95_M",
]


def get_metric(run):
    for metric_name in METRIC_CANDIDATES:
        if metric_name in run.data.metrics:
            return metric_name, run.data.metrics[metric_name]

    return None, None


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Знаходимо версії за aliases
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger",
    )

    print(f"Champion:   version {champion.version}")
    print(f"Challenger: version {challenger.version}")

    # 2. Знаходимо training runs цих моделей
    champion_run = client.get_run(champion.run_id)
    challenger_run = client.get_run(challenger.run_id)

    # 3. Беремо Mask mAP50-95
    champion_metric_name, champion_metric = get_metric(
        champion_run
    )

    challenger_metric_name, challenger_metric = get_metric(
        challenger_run
    )

    if champion_metric is None:
        print("\nChampion metrics available:")
        for key, value in champion_run.data.metrics.items():
            print(f"  {key}: {value}")

        raise ValueError(
            "Mask mAP50-95 metric was not found for champion."
        )

    if challenger_metric is None:
        print("\nChallenger metrics available:")
        for key, value in challenger_run.data.metrics.items():
            print(f"  {key}: {value}")

        raise ValueError(
            "Mask mAP50-95 metric was not found for challenger."
        )

    print(
        f"\nChampion metric "
        f"({champion_metric_name}): "
        f"{champion_metric:.6f}"
    )

    print(
        f"Challenger metric "
        f"({challenger_metric_name}): "
        f"{challenger_metric:.6f}"
    )

    # 4. Реальне порівняння
    if challenger_metric > champion_metric:
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            challenger.version,
        )

        print(
            f"\nPROMOTED: version {challenger.version} "
            f"is the new champion "
            f"({challenger_metric:.6f} > "
            f"{champion_metric:.6f})"
        )

    else:
        print(
            f"\nNOT PROMOTED: version {champion.version} "
            f"remains champion "
            f"({champion_metric:.6f} >= "
            f"{challenger_metric:.6f})"
        )


if __name__ == "__main__":
    main()