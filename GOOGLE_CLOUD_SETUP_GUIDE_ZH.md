# 🚀 Google Cloud Vertex AI (Gemini 2.5) 隊友連線操作指南

這份指南專門教你如何連線到我們共用的 Google Cloud 專案（`project-aefe3ba2-ab8b-478a-82d`），並在你的電腦本機透過 Python 呼叫 Vertex AI 的 Gemini 2.5 模型。

---

## 📋 專案基本資訊

* **GCP 專案 ID**：`project-aefe3ba2-ab8b-478a-82d`
* **預設區域（Location）**：`us-central1`
* **可用模型**：`gemini-2.5-flash`（速度快、成本低）或 `gemini-2.5-pro`（複雜推理）
* **必要 IAM 權限**：`Vertex AI User`（隊長已將你的 Google 帳號加入專案）

---

## 🔑 方法一：個人帳號授權登入（ADC 模式）—— ⭐️ 強烈推薦

這是最簡單、最安全的方法，**不需要手動下載任何金鑰檔案（避免金鑰洩漏）**。

### 步驟 1：安裝 Google Cloud CLI (`gcloud`)
如果電腦尚未安裝 `gcloud`：
* **Windows**：下載並安裝 ➔ [Google Cloud CLI 官方安裝包](https://cloud.google.com/sdk/docs/install#windows)。
* **macOS**：打開終端機執行 `brew install --cask google-cloud-sdk`
* **Linux**：打開終端機執行 `curl https://sdk.cloud.google.com | bash`

安裝完成後，打開終端機（PowerShell 或 Terminal）確認安裝成功：
```bash
gcloud --version
```

---

### 步驟 2：終端機執行 4 行授權指令
在終端機中依序執行以下 4 行指令：

```bash
# 1. 登入你的 Google 帳號（瀏覽器會跳出登入視窗，點擊允許）
gcloud auth login

# 2. 授權 Python SDK 使用本機預設憑證 (ADC)
gcloud auth application-default login

# 3. 綁定共用專案 ID
gcloud config set project project-aefe3ba2-ab8b-478a-82d

# 4. 設定配額專案（消除 Quota Warning 警告）
gcloud auth application-default set-quota-project project-aefe3ba2-ab8b-478a-82d
```

---

## 🔑 方法二：下載 Service Account JSON 金鑰（備用方案）

如果你不想在瀏覽器登入，或是在無法開啟瀏覽器的環境（如伺服器/容器）：

1. 登入 [Google Cloud Console 服務帳戶頁面](https://console.cloud.google.com/iam-admin/serviceaccounts?project=project-aefe3ba2-ab8b-478a-82d)。
2. 點擊你的服務帳號（Service Account）。
3. 進入 **金鑰（Keys）** 標籤 ➔ 點擊 **新增金鑰（Add Key）** ➔ **建立新的金鑰（Create new key）** ➔ 選擇 **JSON** ➔ 下載存檔。
4. 將 `.json` 檔案放在本機（例如 `C:\secrets\gcp_key.json`，**切勿上傳到 Git**）。
5. 在終端機設定環境變數：
   * **Windows PowerShell**：
     ```powershell
     $env:GOOGLE_APPLICATION_CREDENTIALS="C:\secrets\gcp_key.json"
     $env:GOOGLE_CLOUD_PROJECT="project-aefe3ba2-ab8b-478a-82d"
     $env:GOOGLE_CLOUD_LOCATION="us-central1"
     ```
   * **macOS / Linux**：
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp_key.json"
     export GOOGLE_CLOUD_PROJECT="project-aefe3ba2-ab8b-478a-82d"
     export GOOGLE_CLOUD_LOCATION="us-central1"
     ```

---

## 🧪 步驟 3：安裝 Python 套件並測試連線

### 1. 安裝官方 Google GenAI SDK
```bash
pip install google-genai
```

### 2. 建立測試腳本 `test_gemini.py`
建立一個簡單的 Python 檔案 `test_gemini.py`，內容如下：

```python
from google import genai

# 初始化 Vertex AI 客戶端
client = genai.Client(
    vertexai=True,
    project="project-aefe3ba2-ab8b-478a-82d",
    location="us-central1"
)

print("正在發送測試請求給 Gemini 2.5...")

# 呼叫模型
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="請用一句話確認 Vertex AI 連線成功！"
)

print("\n[連線成功！模型回覆]：")
print(response.text)
```

### 3. 執行測試
```bash
python test_gemini.py
```

看到模型回覆即代表你的開發環境已經成功連上 Google Cloud Vertex AI，可以開始自由調用 Gemini 進行開發了！

---

## ❓ 常見錯誤排查（FAQ）

### 1. `403 PermissionDenied: Vertex AI API has not been used...`
* **解法**：在終端機執行啟用 API：
  ```bash
  gcloud services enable aiplatform.googleapis.com --project=project-aefe3ba2-ab8b-478a-82d
  ```

### 2. `UserWarning: ... authenticated ... without a quota project`
* **解法**：執行以下指令即可消除警告：
  ```bash
  gcloud auth application-default set-quota-project project-aefe3ba2-ab8b-478a-82d
  ```

### 3. `403 PermissionDenied: User does not have permission 'aiplatform.endpoints.predict'`
* **解法**：請聯繫隊長在 GCP 後台確認你的 Google 帳號已被賦予 **Vertex AI 使用者（Vertex AI User）** 角色。
