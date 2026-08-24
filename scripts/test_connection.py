import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from google import genai
from config.settings import settings

def test_connectivity():
    print("=" * 70)
    print("🚀 CineQA - Google Cloud Vertex AI Connectivity Test")
    print("=" * 70)
    print(f"• GCP Project:  {settings.GOOGLE_CLOUD_PROJECT}")
    print(f"• GCP Location: {settings.GOOGLE_CLOUD_LOCATION}")
    print(f"• Model Target: {settings.DEFAULT_GEMINI_MODEL}")
    print("-" * 70)
    print("Sending ping prompt to Gemini on Vertex AI...")

    try:
        client = settings.get_genai_client()
        response = client.models.generate_content(
            model=settings.DEFAULT_GEMINI_MODEL,
            contents="Hello CineQA! Confirm Vertex AI connectivity in one sentence."
        )
        print("\n[SUCCESS] Vertex AI responded successfully:")
        print(f"💬 '{response.text.strip()}'")
        print("=" * 70)
        print("🎉 You are fully authenticated and ready to run CineQA!")
    except Exception as e:
        print("\n[ERROR] Connectivity verification failed:")
        print(e)
        print("\n💡 Tip: Run 'gcloud auth application-default login' and ensure your account has 'Vertex AI User' role.")
        sys.exit(1)

if __name__ == "__main__":
    test_connectivity()
