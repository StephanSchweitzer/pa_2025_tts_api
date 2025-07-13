from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
import torchaudio
import io
import os
from typing import Optional
from emotion_model import ValenceArousalXTTS



# === CONFIGURATION ===
ADAPTER_PATH = "models/adapters/model.pth"
MODEL_DIR = "./models/xtts_v2"
DEFAULT_REFERENCE_AUDIO = "voices/adr/test1.wav"
# BUCKET_MODEL = "models/xtts_v2"
# BUCKET_ADAPTER = "models/adapters/model.pth"
# BUCKET_OUTPUT = os.getenv("GCS_BUCKET_NAME", "your-models-bucket")
# BUCKET_REFERENCE_AUDIO = "voices/adr/test1.wav"


# === PYDANTIC MODEL ===
class EmotionTTSRequest(BaseModel):
    text: str = Field(..., description="Texte à synthétiser")
    emotion: str = Field(..., description="Émotion: happy, sad, angry, neutral")
    language: str = Field(default="fr", description="""Langue ("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja")""")


# === FASTAPI APP ===
app = FastAPI(title="VoiceCast generation audio", version="1.0.0")

# === GLOBAL VARIABLES ===
model = None
storage_client = None

# Émotions prédéfinies
EMOTIONS = {
    "happy": (0.9, 0.7),
    "sad": (0.1, 0.1),
    "angry": (0.1, 0.9),
    "neutral": (0.5, 0.5),
}


# === STARTUP EVENT ===
@app.on_event("startup")
async def startup_event():
    global model
    print("Chargement du modèle...")

    model = ValenceArousalXTTS(local_model_dir=MODEL_DIR)

    if torch.cuda.is_available():
        model = model.cuda()
        print("Modèle chargé sur GPU")
    else:
        print("Modèle chargé sur CPU")

    model.load_valence_arousal_adapter(ADAPTER_PATH)
    print("✅ Modèle chargé avec succès")


# === HELPER FUNCTION ===
def audio_to_bytes(audio_tensor: torch.Tensor) -> bytes:
    """
    Convertit un tensor audio en bytes WAV
    """
    if not isinstance(audio_tensor, torch.Tensor):
        audio_tensor = torch.from_numpy(audio_tensor)

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.cpu(), 24000, format="wav")
    buffer.seek(0)
    return buffer.getvalue()


# === MAIN ENDPOINT ===
@app.post("/synthesize")
async def synthesize_speech(request: EmotionTTSRequest):
    """Synthétise la parole avec une émotion"""
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    if request.emotion not in EMOTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Émotion non reconnue : '{request.emotion}'. Disponibles: {list(EMOTIONS.keys())}"
        )

    if request.language not in ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]:
        raise HTTPException(
            status_code=400,
            detail=f"""Langue non reconnue : '{request.language}'. Disponibles: "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"."""
        )

    if request.audio_path is None:
        request.audio_path = DEFAULT_REFERENCE_AUDIO

    valence, arousal = EMOTIONS[request.emotion]


    try:
        # Génération de l'audio
        audio_output = model.inference_with_valence_arousal(
            text=request.text,
            language=request.language,
            audio_path=DEFAULT_REFERENCE_AUDIO,
            valence=valence,
            arousal=arousal,
            temperature=0.75,
            length_penalty=1.5,
            repetition_penalty=2.0
        )

        # Extraire le tensor audio
        if isinstance(audio_output, dict) and 'wav' in audio_output:
            audio = audio_output['wav']
        else:
            audio = audio_output

        # Convertir en bytes
        audio_bytes = audio_to_bytes(audio)

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=synthesized_speech.wav"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la synthèse: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
