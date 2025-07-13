import requests

# Configuration
API_URL = "http://localhost:8000/synthesize"


def test_emotion_tts(text: str,
                     emotion: str = "neutral",
                     language: str = "en",
                     audio_path: str = None):
    """Test de synthèse avec émotion"""
    data = {
        "text": text,
        "emotion": emotion,
        "language": language,
        "audio_path": audio_path
    }

    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        filename = f"output_{emotion}.wav"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio sauvegardé : {filename}")
    else:
        print(f"❌ Erreur : {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Tests avec différentes émotions
    test_emotion_tts(
        text="Ca va bien et toi ?",
        emotion="neutral",
        language="fr"
    )