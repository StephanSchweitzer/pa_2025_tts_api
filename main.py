import logging
import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import torch
import uvicorn

from config import Config
from models import HealthResponse
from tts_utils.emotion_model import ValenceArousalXTTS
from tts_utils.audio_processor import AudioProcessor, cleanup_temp_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
audio_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, audio_processor

    logger.info("Starting up application...")

    config_errors = Config.validate_paths()
    if config_errors:
        for error in config_errors:
            logger.error(error)
        raise RuntimeError(f"Configuration validation failed: {config_errors}")

    audio_processor = AudioProcessor(
        target_sample_rate=Config.SAMPLE_RATE,
        max_duration=30.0,  # 30 seconds max
        min_duration=0.5  # 0.5 seconds min
    )
    logger.info("✅ Audio processor initialized")

    try:
        logger.info("Loading XTTS model...")
        model = ValenceArousalXTTS(local_model_dir=Config.MODEL_DIR)

        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")

        model.load_valence_arousal_adapter(Config.ADAPTER_PATH)
        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    yield

    logger.info("Shutting down application...")
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Voice Synthesis API",
    description="Text-to-Speech API with valence and arousal control using reference audio",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        sample_rate=Config.SAMPLE_RATE,
        supported_languages=Config.SUPPORTED_LANGUAGES
    )


@app.post("/synthesize")
async def synthesize_speech_with_reference(
        text: str = Form(...),
        valence: float = Form(...),
        arousal: float = Form(...),
        reference_audio: UploadFile = File(...),
        language: str = Form("en"),
        temperature: float = Form(0.75),
        length_penalty: float = Form(1.0),
        repetition_penalty: float = Form(2.0),
        top_p: float = Form(0.8)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if audio_processor is None:
        raise HTTPException(status_code=503, detail="Audio processor not initialized")

    if not (0 <= valence <= 1.0):
        raise HTTPException(status_code=400, detail="Valence must be between -1.0 and 1.0")
    if not (0 <= arousal <= 1.0):
        raise HTTPException(status_code=400, detail="Arousal must be between -1.0 and 1.0")
    if language not in Config.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: '{language}'. Supported: {Config.SUPPORTED_LANGUAGES}"
        )
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    temp_audio_path = None
    try:
        logger.info(f"Synthesizing: '{text[:50]}...' with valence={valence}, arousal={arousal}")

        audio_content = await reference_audio.read()

        audio_tensor, sample_rate = audio_processor.process_uploaded_audio(
            audio_bytes=audio_content,
            content_type=reference_audio.content_type,  # Can be None - processor will handle it
            filename=reference_audio.filename
        )

        try:
            audio_output = model.inference_with_valence_arousal(
                text=text,
                language=language,
                audio_tensor=audio_tensor,
                valence=valence,
                arousal=arousal,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_p=top_p
            )
        except (TypeError, AttributeError) as e:
            logger.info("Model requires file path, creating temporary file")

            temp_audio_path = audio_processor.save_tensor_as_temp_file(
                audio_tensor, sample_rate
            )

            audio_output = model.inference_with_valence_arousal(
                text=text,
                language=language,
                audio_path=temp_audio_path,
                valence=valence,
                arousal=arousal,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_p=top_p
            )

        if isinstance(audio_output, dict) and 'wav' in audio_output:
            output_audio = audio_output['wav']
        else:
            output_audio = audio_output

        audio_bytes = AudioProcessor.tensor_to_bytes(output_audio, Config.SAMPLE_RATE)

        logger.info("✅ Audio synthesis completed successfully")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesized_speech.wav",
                "Content-Length": str(len(audio_bytes))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    finally:
        if temp_audio_path:
            cleanup_temp_file(temp_audio_path)


@app.get("/")
async def root():
    return {
        "message": "Voice Synthesis API",
        "version": "1.0.0",
        "description": "Text-to-Speech with emotional control using reference audio",
        "endpoints": {
            "health": "/health - Check API health status",
            "synthesize": "/synthesize - Generate speech with reference audio"
        },
        "supported_languages": Config.SUPPORTED_LANGUAGES
    }


if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)