-- Dataset upload metadata table
-- Created alongside Airflow's own metadata DB in the same Postgres instance.
-- Used by the sync_dolphin_dataset DAG (task t4_write_manifest) to record
-- each dataset version's provenance for reproducibility (design §4.3).

CREATE TABLE IF NOT EXISTS dataset_uploads (
    id               SERIAL PRIMARY KEY,
    dataset_version  TEXT        NOT NULL UNIQUE,   -- e.g. "2024-07-10"
    run_id           TEXT        NOT NULL,           -- Airflow run_id
    timestamp_utc    TIMESTAMPTZ NOT NULL,
    files_uploaded   INTEGER     NOT NULL DEFAULT 0,
    dupes_removed    INTEGER     NOT NULL DEFAULT 0,
    manifest_path    TEXT,                           -- minio://bucket/key
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE dataset_uploads IS
  'One row per DAG run. Training scripts join on dataset_version to get the '
  'exact MinIO path for reproducible experiment data snapshots (design §4.3).';
