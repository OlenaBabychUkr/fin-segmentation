# Dolphin Fin Instance Segmentation — MLOps Baseline

ML inference service and data pipeline for instance segmentation of dolphin
dorsal fins. Built according to the high-level design document.

> **⚠️ Placeholder model notice**
> The current inference service uses a **pretrained YOLOv8n-seg (COCO)** model
> as a baseline to demonstrate API contracts and data flow. It is NOT trained on
> dolphin fins and will not detect them specifically.
> Replace `MODEL_PATH` with a custom checkpoint after Assignment 2/3.

---

## Project structure

```
dolphin-segmentation/
├── inference/                   # Part 1 — FastAPI inference service
│   ├── app/
│   │   ├── main.py              # FastAPI routes (/predict, /health)
│   │   ├── inference.py         # YOLO inference engine + mask crop logic
│   │   └── schemas.py           # Pydantic request/response models
│   ├── requirements.txt
│   ├── Dockerfile               # Multi-stage build
│   └── .dockerignore
│
├── airflow/                     # Part 2 — Data pipeline
│   ├── dags/
│   │   └── sync_dolphin_dataset.py   # DAG: Drive → validate → MinIO → manifest
│   ├── init_db.sql              # Creates dataset_uploads table in Postgres
│   └── README_connections.md   # How to configure the Google Drive credential
│
├── scripts/
│   ├── demo_request.py          # Python demo (download test image + call API)
│   ├── demo_request.sh          # curl demo
│   └── visualize_masks.py       # Overlay masks on image for manual review (design §5.5)
│
├── weights/                     # Mount custom .pt checkpoints here (gitignored)
├── docker-compose.yml           # Full stack: inference + MinIO + Airflow + Postgres
├── .env.example                 # Copy to .env and fill in credentials
├── .gitignore                   # Excludes .env, weights, raw data, credentials
└── README.md
```

---

## Part 1 — Inference API

### API contract

| Field | Role | How it's produced |
|---|---|---|
| `mask.points` | **PRIMARY output** — polygon contour | Directly from YOLO segmentation model |
| `bbox` | Derived metadata | `cv2.boundingRect()` of mask contour — NOT used for cropping |
| `crop_base64` | Mask-shaped fin silhouette | Binary mask → alpha channel → tight crop to bbox bounds |
| `crop_type` | Always `"mask_shaped"` | Explicit marker — NOT a rectangular bbox crop |

This distinction is the core motivation for instance segmentation over object
detection (design §2): bbox crops capture water, body, neighbouring dolphins;
mask crops yield only the fin silhouette.

### Build & run (Docker)

```bash
# 1. Copy and edit environment config
cp .env.example .env           # edit MinIO/Postgres/Airflow credentials

# 2. Start inference + MinIO only (no Airflow):
docker-compose up inference minio minio-init

# 3. Full stack including Airflow:
docker-compose up
```

The inference API is available at `http://localhost:8000`.

### Using a custom-trained checkpoint

After training your dolphin-fin model (Assignment 2/3):

1. Place the `.pt` file in `weights/dolphin_fin_yolo_seg_v1.pt`
2. In `docker-compose.yml`, uncomment and update:
   ```yaml
   environment:
     MODEL_PATH: /weights/dolphin_fin_yolo_seg_v1.pt
     MODEL_NAME: dolphin_fin_yolo_seg_v1
   ```
3. Rebuild: `docker-compose up --build inference`

The `model_info.is_placeholder` field in responses will remain `true` until
you change it in `inference/app/main.py`.

---

## Part 1 — Demo & testing

### Prerequisites (local, outside Docker)
```bash
pip install requests Pillow opencv-python numpy
```

### Option A: Python demo
```bash
# Download a free test image + send to API:
python scripts/demo_request.py --download-test-image

# Your own image:
python scripts/demo_request.py --image path/to/dolphin.jpg --conf 0.3
```

### Option B: curl
```bash
bash scripts/demo_request.sh              # auto-downloads test image
bash scripts/demo_request.sh dolphin.jpg  # your image
```

### Option C: raw curl (multipart upload)
```bash
curl -X POST http://localhost:8000/predict/upload \
     -F "file=@dolphin.jpg;type=image/jpeg" \
     -F "conf_threshold=0.25" \
     -F "include_crop=true"
```

### Option D: JSON / base64
```bash
B64=$(base64 -w0 dolphin.jpg)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"image_base64\": \"$B64\", \"image_id\": \"test-01\", \"conf_threshold\": 0.25}"
```

### Visualize masks (manual review — design §5.5)
```bash
# From saved API response:
python scripts/visualize_masks.py --image dolphin.jpg \
    --response example_response.json --out viz.jpg

# Live (call API + draw in one step):
python scripts/visualize_masks.py --image dolphin.jpg --live --out viz.jpg
```

### Example response (abbreviated)
```json
{
  "image_id": "test-01",
  "num_fins": 1,
  "inference_time_ms": 142.5,
  "image_width": 640,
  "image_height": 427,
  "model_info": {
    "model_name": "yolov8n-seg.pt",
    "is_placeholder": true,
    "note": "Pretrained COCO baseline. Replace MODEL_PATH ..."
  },
  "fins": [
    {
      "instance_id": 0,
      "confidence": 0.712,
      "mask": {
        "points": [312.1, 88.4, 318.9, 85.2, "..."],
        "width": 640,
        "height": 427
      },
      "bbox": {
        "x": 280, "y": 72, "width": 95, "height": 60
      },
      "crop_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
      "crop_type": "mask_shaped"
    }
  ]
}
```

---

## Part 2 — Data pipeline (MinIO + Airflow)

### MinIO buckets

| Bucket | Contents |
|---|---|
| `dolphin-fins-raw` | Raw input photos (no annotations) |
| `dolphin-fins-annotated` | Photos + YOLO-seg `.txt` annotation files |
| `dolphin-fins-manifests` | JSON manifests per dataset version |

Object path pattern: `<YYYY-MM-DD>/<filename>` — date = DAG execution date.

MinIO web console: `http://localhost:9001` (credentials from `.env`)

### Airflow DAG: `sync_dolphin_dataset`

Schedule: `@daily` + manual trigger.

```
t1_list_drive_files
      │
      ▼
t2_validate_files       ← image integrity check + phash dedup (design §5.3)
      │
      ▼
t3_upload_to_minio      ← partitioned by date → dataset_version
      │
      ▼
t4_write_manifest       ← JSON manifest in MinIO + row in dataset_uploads table
```

Airflow UI: `http://localhost:8080` (user: `admin`, password: `admin`)

### Configure Google Drive credential

**See `airflow/README_connections.md` for full instructions.**  
Short version:
1. Create a GCP service account with Drive read-only access.
2. Share the Drive folder with the service account email.
3. In Airflow UI: Admin → Connections → add `google_drive_default` with the
   service account JSON pasted into the "Keyfile JSON" field.
4. Set `GOOGLE_DRIVE_FOLDER_ID` in `.env`.

⚠️ Do NOT put the service account JSON in `.env` or in code.

### Reproducibility link to training (design §4.3)

```python
# Training script pattern:
DATASET_VERSION = "2024-07-10"   # pin to a specific DAG run

# 1. Read the manifest to verify the snapshot:
manifest = minio.get_object("dolphin-fins-manifests", f"{DATASET_VERSION}/manifest.json")

# 2. Download annotated data for that version:
objects = minio.list_objects("dolphin-fins-annotated", prefix=f"{DATASET_VERSION}/")
for obj in objects:
    minio.fget_object("dolphin-fins-annotated", obj.object_name, f"./data/{obj.object_name}")

# 3. Train YOLO on ./data/<DATASET_VERSION>/
# → Experiment is fully reproducible: same version = same data every time.
```

---

## Checklist

### ✅ Done (this iteration)
- [x] **Part 1**: FastAPI inference service with YOLOv8n-seg pretrained baseline
- [x] Mask polygon as PRIMARY output, bbox as derived metadata
- [x] Mask-shaped crop (alpha-masked PNG, not bbox crop) — design §2
- [x] Pydantic schemas with explicit `crop_type: "mask_shaped"` field
- [x] Multi-stage Dockerfile + `.dockerignore`
- [x] `docker-compose.yml` (inference + MinIO + Airflow + Postgres)
- [x] **Part 2**: MinIO buckets with date-partitioned versioning
- [x] Airflow DAG `sync_dolphin_dataset` (4 tasks, `@daily`)
- [x] Google Drive integration (service account via Airflow Connection — NOT in code)
- [x] phash-based near-duplicate detection (design §5.3)
- [x] Dataset manifest + `dataset_uploads` Postgres table for reproducibility
- [x] Demo scripts: Python + curl + mask visualizer
- [x] `.gitignore` excluding credentials, weights, raw data
- [x] All placeholder locations clearly marked with ⚠️ comments

### 🔜 Next iterations (out of scope per design)
- [ ] **Custom model training** (Assignment 2/3): train YOLOv8-seg on the
      dolphin-fin dataset exported from Roboflow, then set `MODEL_PATH`
- [ ] **Re-ID model** (out of scope — design explicitly defers): similarity
      search / individual dolphin identification from fin crops
- [ ] **Online / real-time API**: design §out-of-scope; current batch mode is sufficient
- [ ] **W&B experiment tracking** integration with training loop
- [ ] **Quality scoring** of crops (blur detection, occlusion filter — design §5.4)
- [ ] **Roboflow webhook** to auto-trigger Airflow DAG on new annotation exports
- [ ] GPU support in Docker (`--gpus all`, CUDA base image)
- [ ] Model versioning beyond the single checkpoint (MLflow Model Registry)
- [ ] Automated integration tests for the inference pipeline
