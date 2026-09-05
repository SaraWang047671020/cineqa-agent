from google import genai
from config.settings import settings

def get_embedding(text: str) -> list:
    try:
        client = settings.get_genai_client()
        result = client.models.embed_content(
            model='text-embedding-004',
            contents=text,
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"[Embedding Error]: {e}")
        # Return a zero vector of size 768 (standard Gemini embedding size) as fallback
        return [0.0] * 768
