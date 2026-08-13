# MLflow training service

Files:
- `train.py` — YOLOv8 segmentation training + MLflow tracking.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — training image.
- `docker-compose.training-snippet.yml` — paste the `training:` service under `services:` in the existing compose file.

Expected existing project paths:
- `training/dataset_yolo/data.yaml`
- `weights/yolov8n-seg.pt`

Before training:
1. Ensure the MLflow server is running.
2. Ensure MinIO contains the bucket `mlflow-artifacts`.
3. Ensure MLflow was started with `--artifacts-destination s3://mlflow-artifacts`.
4. Add the training service from the compose snippet.

Build:
    docker compose build training

Run:
    docker compose run --rm training

Open MLflow:
    http://localhost:5000

The run logs:
- hyperparameters and dataset version
- final Ultralytics metrics
- an explicit validation pass
- the full Ultralytics run directory, including `best.pt`, `last.pt`, plots and CSV outputs

Artifacts are uploaded through the MLflow tracking server. With `--artifacts-destination s3://mlflow-artifacts`,
the server proxies artifact storage to MinIO, so the training container itself does not need MinIO credentials.


MLflow 3.5+ host validation:
- The training container calls `http://mlflow:5000`.
- Add `--allowed-hosts "localhost:*,127.0.0.1:*,mlflow:*"` to the existing MLflow server command.
- See `mlflow-compose-patch.yml`.
