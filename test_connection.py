import sys
import os

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

def main():
    print("=" * 60)
    print("🎬 CineQA-Agent: Google Cloud & Gemini Connection Test")
    print("=" * 60)
    print(f"Project ID : {settings.GOOGLE_CLOUD_PROJECT}")
    print(f"Location   : {settings.GOOGLE_CLOUD_LOCATION}")
    print(f"Model      : {settings.DEFAULT_GEMINI_MODEL}")
    print("-" * 60)

    try:
        client = settings.get_genai_client()
        print("Sending test request to Gemini 2.0...")
        
        response = client.models.generate_content(
            model=settings.DEFAULT_GEMINI_MODEL,
            contents="Say hello and confirm Google Cloud Vertex AI connection is working!"
        )
        
        print("\n[SUCCESS] Connection verified! Gemini response:")
        print(f">>> {response.text.strip()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Run: gcloud auth application-default login")
        print("2. Run: gcloud services enable aiplatform.googleapis.com --project " + settings.GOOGLE_CLOUD_PROJECT)
        print("3. Or set GEMINI_API_KEY in .env file")
        print("=" * 60)

if __name__ == "__main__":
    main()
