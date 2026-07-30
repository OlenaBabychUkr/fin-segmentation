# Setting up Airflow Connections for the DAG

## ⚠️ Privacy note
Credentials for Google Drive must NEVER be committed to git or placed in
`.env` / docker-compose.  Configure them exclusively via the Airflow UI.

---

## Google Drive Connection (`google_drive_default`)

The DAG `sync_dolphin_dataset` reads from a private Google Drive folder using
a **service account** with read-only access.

### Step 1 — Create a Google Cloud service account

1. Go to [Google Cloud Console](https://console.cloud.google.com/) →
   IAM & Admin → Service Accounts.
2. Create a new service account (e.g., `dolphin-data-reader`).
3. Under "Keys", click **Add Key → Create new key → JSON**.
   Download the `.json` file — keep it safe.
4. In **Google Drive**, right-click the folder containing the dolphin photos
   and click **Share**. Add the service account email
   (`dolphin-data-reader@<project>.iam.gserviceaccount.com`) with **Viewer**
   access.

### Step 2 — Register the connection in Airflow

1. Open Airflow UI at `http://localhost:8080` (admin / admin).
2. **Admin → Connections → Add a new record**.
3. Fill in:
   | Field          | Value                                   |
   |----------------|-----------------------------------------|
   | Connection Id  | `google_drive_default`                  |
   | Connection Type| `Google Cloud` (or `Generic`)           |
   | Keyfile JSON   | Paste the **entire contents** of the `.json` service account key |
4. Click **Save**.

The DAG reads this via `BaseHook.get_connection("google_drive_default")` and
parses the JSON from `conn.extra`.

### Step 3 — Set the Drive folder ID

Either:
- Set `GOOGLE_DRIVE_FOLDER_ID` in `.env` (docker-compose picks it up), **OR**
- Go to Airflow UI → **Admin → Variables** → Add key `google_drive_folder_id`
  with the folder ID as value, and update the DAG to use
  `Variable.get("google_drive_folder_id")`.

The folder ID is the last part of the Google Drive URL:
```
https://drive.google.com/drive/folders/<FOLDER_ID_HERE>
```

---

## MinIO connection (no Airflow Connection needed)

MinIO credentials are passed as environment variables directly:
```
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
```
These are set in `.env` (which is gitignored).
