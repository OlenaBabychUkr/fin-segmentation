import os
import re
from pathlib import Path
from datetime import datetime

import mlflow
from ultralytics import YOLO

import pandas as pd
import mlflow.pyfunc
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "dolphin-fin-segmentation")

MODEL_PATH = os.getenv("MODEL_PATH", "/weights/yolov8n-seg.pt")
DATA_YAML = os.getenv("DATA_YAML", "/data/data.yaml")
DATASET_VERSION = os.getenv("DATASET_VERSION", "roboflow-seg-v1")

EPOCHS = int(os.getenv("EPOCHS", "20"))
IMGSZ = int(os.getenv("IMGSZ", "432"))
BATCH = int(os.getenv("BATCH", "4"))
DEVICE = os.getenv("DEVICE", "cpu")
WORKERS = int(os.getenv("WORKERS", "2"))

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/runs"))


def _safe_metric_name(name: str) -> str:
    """Keep MLflow metric names readable and valid."""
    return re.sub(r"[^a-zA-Z0-9_./ -]", "_", name)


def _numeric_metrics(metrics_obj) -> dict[str, float]:
    """Extract numeric metrics from an Ultralytics metrics object."""
    raw = getattr(metrics_obj, "results_dict", None)
    if raw is None and isinstance(metrics_obj, dict):
        raw = metrics_obj
    if not raw:
        return {}

    out = {}
    for key, value in raw.items():
        try:
            out[_safe_metric_name(str(key))] = float(value)
        except (TypeError, ValueError):
            continue
    return out

class YOLOSegmentationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        """
        model_input must contain an 'image_path' column.
        """
        image_paths = model_input["image_path"].tolist()

        results = self.model.predict(
            source=image_paths,
            verbose=False,
        )

        predictions = []

        for result in results:
            predictions.append({
                "boxes": (
                    result.boxes.xyxy.cpu().numpy().tolist()
                    if result.boxes is not None
                    else []
                ),
                "scores": (
                    result.boxes.conf.cpu().numpy().tolist()
                    if result.boxes is not None
                    else []
                ),
                "masks": (
                    result.masks.xy
                    if result.masks is not None
                    else []
                ),
            })

        return predictions

def get_promotion_metric(run):
    candidates = [
        "val_metrics/mAP50-95_M_",
        "metrics/mAP50-95_M_",
        "val_metrics/mAP50-95_M",
        "metrics/mAP50-95_M",
    ]

    for key in candidates:
        if key in run.data.metrics:
            return key, run.data.metrics[key]

    return None, None

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"yolov8n-seg-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model": Path(MODEL_PATH).name,
                "task": "instance_segmentation",
                "dataset_version": DATASET_VERSION,
                "data_yaml": DATA_YAML,
                "epochs": EPOCHS,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "device": DEVICE,
                "workers": WORKERS,
            }
        )

        model = YOLO(MODEL_PATH)

        train_metrics = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )

        # Log final training/validation metrics returned by Ultralytics.
        metrics = _numeric_metrics(train_metrics)
        if metrics:
            mlflow.log_metrics(metrics)

        # Run an explicit validation pass on the best weights when available.
        save_dir = RUNS_DIR / run_name
        best_pt = save_dir / "weights" / "best.pt"
        model_for_val = YOLO(str(best_pt if best_pt.exists() else MODEL_PATH))
        val_metrics_obj = model_for_val.val(
            data=DATA_YAML,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
        )

        val_metrics = {
            f"val_{k}": v for k, v in _numeric_metrics(val_metrics_obj).items()
        }
        if val_metrics:
            mlflow.log_metrics(val_metrics)

        # Save the complete Ultralytics run directory:
        # best.pt, last.pt, results.csv, results.png, confusion matrices, args.yaml, etc.
        if save_dir.exists():
            # 1. Зберігаємо всі Ultralytics artifacts
            mlflow.log_artifacts(
                str(save_dir),
                artifact_path="ultralytics_run"
            )

            # 2. Шлях до best.pt
            best_pt = save_dir / "weights" / "best.pt"

            if not best_pt.exists():
                raise FileNotFoundError(
                    f"Best checkpoint not found: {best_pt}"
                )

            # 3. Логуємо модель у MLflow
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=YOLOSegmentationModel(),
                artifacts={
                    "weights": str(best_pt)
                },
                pip_requirements=[
                    "mlflow==3.15.1",
                    "ultralytics>=8.3,<9",
                    "pandas",
                ],
            )

            # 4. Реєструємо її в Model Registry
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="dolphin-fin-segmentation",
            )

            client = MlflowClient()

            MODEL_REGISTRY_NAME = "dolphin-fin-segmentation"
            new_version = registered_model.version

            # New model becomes challenger
            client.set_registered_model_alias(
                MODEL_REGISTRY_NAME,
                "challenger",
                new_version,
            )

            print(f"challenger -> version {new_version}")


            # Get current champion
            try:
                champion_version = client.get_model_version_by_alias(
                    MODEL_REGISTRY_NAME,
                    "champion",
                )

                champion_run = client.get_run(
                    champion_version.run_id
                )

                champion_metric_name, champion_metric = (
                    get_promotion_metric(champion_run)
                )

            except Exception:
                champion_version = None
                champion_metric_name = None
                champion_metric = None


            # Get challenger metric
            challenger_run = client.get_run(
                registered_model.run_id
            )

            challenger_metric_name, challenger_metric = (
                get_promotion_metric(challenger_run)
            )


            print(
                f"Champion metric ({champion_metric_name}): "
                f"{champion_metric}"
            )

            print(
                f"Challenger metric ({challenger_metric_name}): "
                f"{challenger_metric}"
            )


            # Promotion logic
            if champion_version is None:
                client.set_registered_model_alias(
                    MODEL_REGISTRY_NAME,
                    "champion",
                    new_version,
                )

                client.delete_registered_model_alias(
                    MODEL_REGISTRY_NAME,
                    "challenger",
                )

                print(
                    f"No existing champion. "
                    f"Version {new_version} promoted to champion."
                )

            elif str(champion_version.version) == str(new_version):
                print(
                    f"Version {new_version} is already champion. "
                    "Skipping comparison."
                )

            elif (
                challenger_metric is not None
                and champion_metric is not None
                and challenger_metric > champion_metric
            ):
                client.set_registered_model_alias(
                    MODEL_REGISTRY_NAME,
                    "champion",
                    new_version,
                )

                client.delete_registered_model_alias(
                    MODEL_REGISTRY_NAME,
                    "challenger",
                )

                print(
                    f"Version {new_version} promoted to champion "
                    f"({challenger_metric:.4f} > {champion_metric:.4f})"
                )

            else:
                print(
                    f"Version {new_version} remains challenger. "
                    f"Champion stays at version "
                    f"{champion_version.version}."
                )

            print(
                f"Registered model version: "
                f"{registered_model.version}"
            )            
        mlflow.set_tags(
            {
                "project": "dolphin-fin-segmentation",
                "framework": "ultralytics",
                "dataset_version": DATASET_VERSION,
                "model_stage": "training",
            }
        )

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
        print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"Ultralytics output: {save_dir}")


if __name__ == "__main__":
    main()
