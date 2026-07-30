"""
Airflow DAG: sync_dolphin_dataset
==================================

Syncs dolphin fin images + YOLO-seg annotations from a private Google Drive
folder into MinIO buckets, validates each file, deduplicates via perceptual
hash, and writes a dataset-version manifest.

Tasks:
  t1_list_drive_files   — list new files in the Drive folder
  t2_validate_files     — validate images + annotations, phash dedup
  t3_upload_to_minio    — upload valid pairs to MinIO
  t4_write_manifest     — write JSON manifest + insert into dataset_uploads table

Reproducibility (design §4.3):
  Each DAG run produces a dataset_version (YYYY-MM-DD__<run_id>) stored in the
  manifest. Training scripts consume a specific dataset_version from MinIO,
  guaranteeing the exact data snapshot used for each experiment.

Schedule: daily (@daily) + manual trigger.

# ⚠️ CREDENTIALS — NEVER IN CODE (design §Privacy):
#   Google Drive:  add Airflow Connection "google_drive_default"
#                  (see airflow/README_connections.md)
#   MinIO keys:    read from env vars MINIO_ACCESS_KEY / MINIO_SECRET_KEY
#   Postgres DSN:  managed by Airflow AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

# Google Drive folder containing raw photos and Roboflow-exported annotations.
# ⚠️ Set this in Airflow Variables UI (Admin → Variables) — not in code.
DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")

MINIO_ENDPOINT    = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
# ⚠️ No defaults for credentials — fail fast at import time if not configured
MINIO_ACCESS_KEY  = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY  = os.environ.get("MINIO_SECRET_KEY", "")
if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise EnvironmentError(
        "MINIO_ACCESS_KEY and MINIO_SECRET_KEY env vars are required. "
        "Set them in .env (never in code)."
    )

BUCKET_RAW        = "dolphin-fins-raw"
BUCKET_ANNOTATED  = "dolphin-fins-annotated"
BUCKET_MANIFESTS  = "dolphin-fins-manifests"

PHASH_THRESHOLD   = 10   # Hamming distance; images with dist ≤ this are near-dupes
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Helper: MinIO client
# ---------------------------------------------------------------------------

def _minio_client():
    from minio import Minio  # type: ignore
    return Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://"),
    )


# ---------------------------------------------------------------------------
# Helper: Google Drive client (via service account Airflow Connection)
# ---------------------------------------------------------------------------

def _drive_service():
    """
    Build a Google Drive API service using credentials from the Airflow
    connection 'google_drive_default'.

    ⚠️ The service-account JSON key must be configured in the Airflow
    Connection UI (NOT hardcoded or in .env).  See README_connections.md.
    """
    import json as _json
    from google.oauth2 import service_account  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from airflow.hooks.base import BaseHook

    conn = BaseHook.get_connection("google_drive_default")
    # The service account JSON is stored in conn.extra (as a JSON string)
    extra = _json.loads(conn.extra or "{}")
    keyfile_dict = extra.get("keyfile_dict") or extra  # support both formats

    creds = service_account.Credentials.from_service_account_info(
        keyfile_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Task 1: list new files in Drive folder
# ---------------------------------------------------------------------------

def t1_list_drive_files(**context) -> list[dict]:
    """
    List all image + annotation files in the Drive folder.

    Returns a list of dicts:
      { "file_id": str, "name": str, "mime_type": str, "size": int }

    Only files modified since the last successful DAG run are fetched
    (stored as Airflow Variable 'drive_last_sync_ts').
    """
    if not DRIVE_FOLDER_ID:
        raise ValueError(
            "GOOGLE_DRIVE_FOLDER_ID is not set. "
            "Set it in docker-compose .env or Airflow Variables."
        )

    service = _drive_service()

    last_sync = Variable.get("drive_last_sync_ts", default_var="1970-01-01T00:00:00Z")
    query = (
        f"'{DRIVE_FOLDER_ID}' in parents "
        f"and trashed = false "
        f"and modifiedTime > '{last_sync}'"
    )

    files: list[dict] = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                pageSize=200,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    logger.info("Drive folder contains %d new/modified file(s)", len(files))
    context["ti"].xcom_push(key="drive_files", value=files)
    return files


# ---------------------------------------------------------------------------
# Task 2: validate + dedup
# ---------------------------------------------------------------------------

def t2_validate_files(**context):
    """
    For each image in the Drive file list:
      1. Check it has a paired annotation file (same stem, .txt extension).
      2. Download image bytes and validate (not corrupted, decodable).
      3. Compute perceptual hash (phash) via imagehash library.
      4. Skip near-duplicate images (Hamming dist ≤ PHASH_THRESHOLD).

    Pushes to XCom:
      "valid_pairs": list of { "image_file": dict, "label_file": dict,
                               "phash": str }
    """
    import imagehash  # type: ignore
    from PIL import Image as PILImage  # type: ignore
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore

    service = _drive_service()
    drive_files: list[dict] = context["ti"].xcom_pull(key="drive_files", task_ids="t1_list_drive_files")

    if not drive_files:
        logger.info("No new files to validate.")
        context["ti"].xcom_push(key="valid_pairs", value=[])
        context["ti"].xcom_push(key="phash_dupes_removed", value=0)
        return

    # Build lookup: stem → file dict
    by_stem: dict[str, dict[str, Any]] = {}
    for f in drive_files:
        name = f["name"]
        stem, ext = os.path.splitext(name)
        by_stem.setdefault(stem, {})[ext.lower()] = f

    seen_hashes: list[Any] = []
    valid_pairs: list[dict] = []
    dupes_removed = 0

    for stem, ext_map in by_stem.items():
        # Must have both image and annotation
        image_ext = next(
            (e for e in SUPPORTED_EXTS if e in ext_map), None
        )
        if not image_ext or ".txt" not in ext_map:
            logger.debug("Skipping %s — missing image or .txt annotation", stem)
            continue

        img_file = ext_map[image_ext]

        # Download image to validate + compute phash
        try:
            request = service.files().get_media(fileId=img_file["id"])
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            pil_img = PILImage.open(buf).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to download/decode image %s: %s", stem, exc)
            continue

        # Perceptual hash deduplication (design §5.3)
        phash = imagehash.phash(pil_img)
        is_dupe = any(abs(phash - seen) <= PHASH_THRESHOLD for seen in seen_hashes)
        if is_dupe:
            logger.info("Near-duplicate detected, skipping: %s (phash=%s)", stem, phash)
            dupes_removed += 1
            continue

        seen_hashes.append(phash)
        valid_pairs.append({
            "image_file": img_file,
            "label_file": ext_map[".txt"],
            "phash": str(phash),
        })

    logger.info(
        "Validation complete: %d valid pairs, %d dupes removed",
        len(valid_pairs), dupes_removed,
    )
    context["ti"].xcom_push(key="valid_pairs", value=valid_pairs)
    context["ti"].xcom_push(key="phash_dupes_removed", value=dupes_removed)


# ---------------------------------------------------------------------------
# Task 3: upload to MinIO
# ---------------------------------------------------------------------------

def t3_upload_to_minio(**context):
    """
    Download valid image+annotation pairs from Drive and upload to MinIO.

    MinIO object paths use date-based partitioning for dataset versioning:
      dolphin-fins-annotated/<dataset_version>/<filename>

    where dataset_version = execution_date in YYYY-MM-DD format.
    """
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore

    valid_pairs: list[dict] = context["ti"].xcom_pull(
        key="valid_pairs", task_ids="t2_validate_files"
    )

    if not valid_pairs:
        logger.info("No valid pairs to upload.")
        context["ti"].xcom_push(key="uploaded_count", value=0)
        return

    service = _drive_service()
    minio = _minio_client()
    dataset_version = context["ds"]  # execution date YYYY-MM-DD

    def _download_bytes(file_id: str) -> bytes:
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    uploaded = 0
    for pair in valid_pairs:
        img_file   = pair["image_file"]
        label_file = pair["label_file"]

        try:
            img_bytes   = _download_bytes(img_file["id"])
            label_bytes = _download_bytes(label_file["id"])
        except Exception as exc:
            logger.error("Failed to download %s: %s", img_file["name"], exc)
            continue

        img_key   = f"{dataset_version}/{img_file['name']}"
        label_key = f"{dataset_version}/{label_file['name']}"

        for bucket, key, data in [
            (BUCKET_ANNOTATED, img_key,   img_bytes),
            (BUCKET_ANNOTATED, label_key, label_bytes),
            (BUCKET_RAW,       f"{dataset_version}/{img_file['name']}", img_bytes),
        ]:
            minio.put_object(
                bucket,
                key,
                io.BytesIO(data),
                length=len(data),
            )

        uploaded += 1

    logger.info("Uploaded %d image+annotation pairs to MinIO", uploaded)
    context["ti"].xcom_push(key="uploaded_count", value=uploaded)
    context["ti"].xcom_push(key="dataset_version", value=dataset_version)


# ---------------------------------------------------------------------------
# Task 4: write manifest
# ---------------------------------------------------------------------------

def t4_write_manifest(**context):
    """
    Write a JSON manifest to MinIO and record metadata in the Postgres
    dataset_uploads table.

    The manifest ties together the dataset_version with exact file counts,
    enabling training scripts to reference a reproducible data snapshot
    (design §4.3 reproducibility requirement).

    Training usage pattern:
      1. List manifests in dolphin-fins-manifests/<dataset_version>/manifest.json
      2. Download annotated data from dolphin-fins-annotated/<dataset_version>/
      3. Train with those exact files — experiment is fully reproducible.
    """
    import psycopg2  # type: ignore

    ti = context["ti"]
    dataset_version  = ti.xcom_pull(key="dataset_version",    task_ids="t3_upload_to_minio") or context["ds"]
    uploaded_count   = ti.xcom_pull(key="uploaded_count",     task_ids="t3_upload_to_minio") or 0
    dupes_removed    = ti.xcom_pull(key="phash_dupes_removed", task_ids="t2_validate_files")  or 0
    run_id           = context["run_id"]
    ts               = datetime.utcnow().isoformat()

    manifest = {
        "dataset_version":    dataset_version,
        "run_id":             run_id,
        "timestamp_utc":      ts,
        "files_uploaded":     uploaded_count,
        "dupes_removed":      dupes_removed,
        "source":             "google_drive",
        "bucket_raw":         BUCKET_RAW,
        "bucket_annotated":   BUCKET_ANNOTATED,
        "annotation_format":  "yolo_segmentation",  # Roboflow export format
        "note": (
            "Training scripts should reference dataset_version to ensure "
            "reproducibility (design §4.3)."
        ),
    }

    # Upload manifest to MinIO
    minio = _minio_client()
    manifest_key  = f"{dataset_version}/manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    minio.put_object(
        BUCKET_MANIFESTS,
        manifest_key,
        io.BytesIO(manifest_bytes),
        length=len(manifest_bytes),
        content_type="application/json",
    )
    logger.info("Manifest written to minio://%s/%s", BUCKET_MANIFESTS, manifest_key)

    # Insert row into dataset_uploads Postgres table.
    # Use SQLAlchemy's own URL parser — avoids fragile regex that breaks on
    # special characters or encoded passwords in the DSN string.
    db_url = os.environ.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    if not db_url:
        raise EnvironmentError(
            "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN is not set — "
            "cannot record dataset manifest metadata."
        )

    from sqlalchemy.engine import make_url  # type: ignore
    url = make_url(db_url)
    conn_pg = psycopg2.connect(
        host=url.host,
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        dbname=url.database,
    )
    try:
        with conn_pg, conn_pg.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_uploads
                  (dataset_version, run_id, timestamp_utc,
                   files_uploaded, dupes_removed, manifest_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_version) DO UPDATE
                  SET files_uploaded = EXCLUDED.files_uploaded,
                      timestamp_utc  = EXCLUDED.timestamp_utc;
                """,
                (dataset_version, run_id, ts,
                 uploaded_count, dupes_removed,
                 f"minio://{BUCKET_MANIFESTS}/{manifest_key}"),
            )
        logger.info("Manifest metadata recorded in dataset_uploads table.")
    finally:
        conn_pg.close()

    # Update last-sync timestamp so t1 only fetches new files next run
    Variable.set("drive_last_sync_ts", ts)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="sync_dolphin_dataset",
    default_args=default_args,
    description=(
        "Sync dolphin fin images + YOLO-seg annotations from Google Drive "
        "to MinIO, with phash dedup and dataset versioning."
    ),
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["dolphin", "data-pipeline", "mlops"],
) as dag:

    task1 = PythonOperator(
        task_id="t1_list_drive_files",
        python_callable=t1_list_drive_files,
    )

    task2 = PythonOperator(
        task_id="t2_validate_files",
        python_callable=t2_validate_files,
    )

    task3 = PythonOperator(
        task_id="t3_upload_to_minio",
        python_callable=t3_upload_to_minio,
    )

    task4 = PythonOperator(
        task_id="t4_write_manifest",
        python_callable=t4_write_manifest,
    )

    task1 >> task2 >> task3 >> task4
