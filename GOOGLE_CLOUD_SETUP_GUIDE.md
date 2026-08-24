# 🚀 Google Cloud & Vertex AI Onboarding Guide for CineQA

This guide provides step-by-step instructions for teammates to connect to our shared Google Cloud Project (`project-aefe3ba2-ab8b-478a-82d`) and authenticate AI models (Gemini 2.5 on Vertex AI) locally on Windows, macOS, or Linux.

---

## 📋 Shared Project Details

| Parameter | Value |
| :--- | :--- |
| **GCP Project ID** | `project-aefe3ba2-ab8b-478a-82d` |
| **Default Location/Region** | `us-central1` |
| **Default Model** | `gemini-2.5-flash` (or `gemini-2.5-pro`) |
| **Required IAM Role** | `Vertex AI User` / `Vertex AI Administrator` |

---

## 🔑 Method 1: Application Default Credentials (ADC) — Recommended

This is the cleanest and most secure method. It uses your personal Google account that was granted access to the project without managing raw JSON key files.

### Step 1: Install Google Cloud CLI (`gcloud`)
If you don't have `gcloud` installed:
- **Windows**: Download and run the [Google Cloud CLI Installer](https://cloud.google.com/sdk/docs/install#windows).
- **macOS (Homebrew)**: `brew install --cask google-cloud-sdk`
- **Linux**: `curl https://sdk.cloud.google.com | bash`

Verify installation:
```bash
gcloud --version
```

### Step 2: Authenticate Your Google Account
Run the following commands in your terminal:

```bash
# 1. Login to your Google account in browser
gcloud auth login

# 2. Authenticate Application Default Credentials (ADC) for Python SDK
gcloud auth application-default login

# 3. Set the default active project
gcloud config set project project-aefe3ba2-ab8b-478a-82d

# 4. Set the quota project to prevent quota warning errors
gcloud auth application-default set-quota-project project-aefe3ba2-ab8b-478a-82d
```

---

## 🔑 Method 2: Service Account Key File (`.json`) — Alternative for CI/CD

If you prefer using a persistent service account JSON key:

1. Go to the [Google Cloud Console - Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts?project=project-aefe3ba2-ab8b-478a-82d).
2. Click on the CineQA service account (or create one with the `Vertex AI User` role).
3. Navigate to the **Keys** tab ➔ **Add Key** ➔ **Create new key** ➔ Select **JSON** ➔ Download the file.
4. Save the file locally (e.g. `C:\secrets\gcp_key.json` or `~/.gcp/gcp_key.json`).
5. Set the environment variable in your terminal or `.env`:
   - **Windows PowerShell**:
     ```powershell
     $env:GOOGLE_APPLICATION_CREDENTIALS="C:\secrets\gcp_key.json"
     $env:GOOGLE_CLOUD_PROJECT="project-aefe3ba2-ab8b-478a-82d"
     $env:GOOGLE_CLOUD_LOCATION="us-central1"
     ```
   - **Linux / macOS**:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp_key.json"
     export GOOGLE_CLOUD_PROJECT="project-aefe3ba2-ab8b-478a-82d"
     export GOOGLE_CLOUD_LOCATION="us-central1"
     ```

---

## ⚙️ Environment Configuration (`.env`)

In the root of the repository (`cineqa_agent/`), create or update your `.env` file:

```ini
# Google Cloud Vertex AI Configuration
GOOGLE_CLOUD_PROJECT=project-aefe3ba2-ab8b-478a-82d
GOOGLE_CLOUD_LOCATION=us-central1
DEFAULT_GEMINI_MODEL=gemini-2.5-flash

# Telemetry & Monitoring Ports
PROMETHEUS_PORT=8000
METRICS_EXPORT_INTERVAL=5.0
```

---

## 🧪 Step 3: Verify Python Connection

Run this quick test script to verify your connection to Vertex AI:

```python
# test_connection.py
from google import genai

PROJECT_ID = "project-aefe3ba2-ab8b-478a-82d"
LOCATION = "us-central1"

print(f"Connecting to Vertex AI (Project: {PROJECT_ID}, Location: {LOCATION})...")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello CineQA! Confirm Vertex AI connectivity in one sentence."
)

print("\n[SUCCESS] Vertex AI Response:")
print(response.text)
```

Run the script:
```bash
python test_connection.py
```

Expected output:
```text
Connecting to Vertex AI (Project: project-aefe3ba2-ab8b-478a-82d, Location: us-central1)...

[SUCCESS] Vertex AI Response:
Hello CineQA! Vertex AI connectivity is active and confirmed.
```

---

## 🎬 Step 4: Launch CineQA Studio UI

Once authenticated, launch the full interactive studio:

```bash
# Navigate to repository root
cd cineqa_agent

# Install dependencies if you haven't already
pip install -r requirements.txt

# Start Streamlit studio
python -m streamlit run ui/app.py
```

Open your browser at `http://localhost:8501`.

---

## ❓ Troubleshooting & Common Errors

### 1. `PermissionDenied: 403 Vertex AI API has not been used... or it is disabled`
* **Fix**: Ensure the project owner has enabled the Vertex AI API (`aiplatform.googleapis.com`) on the GCP project, or run:
  ```bash
  gcloud services enable aiplatform.googleapis.com --project=project-aefe3ba2-ab8b-478a-82d
  ```

### 2. `UserWarning: Your application has authenticated using end user credentials... without a quota project`
* **Fix**: Run:
  ```bash
  gcloud auth application-default set-quota-project project-aefe3ba2-ab8b-478a-82d
  ```

### 3. `PermissionDenied: 403 User does not have permission 'aiplatform.endpoints.predict'`
* **Fix**: Ask the project admin to ensure your Google account has the **Vertex AI User** (`roles/aiplatform.user`) role in the GCP IAM console.
