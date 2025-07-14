import os


class Config:
    ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapters/model.pth")
    MODEL_DIR = os.getenv("MODEL_DIR", "./models/xtts_v2")

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    SAMPLE_RATE = 24000

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ]

    @classmethod
    def validate_paths(cls):
        errors = []
        if not os.path.exists(cls.MODEL_DIR):
            errors.append(f"Model directory not found: {cls.MODEL_DIR}")
        if not os.path.exists(cls.ADAPTER_PATH):
            errors.append(f"Adapter file not found: {cls.ADAPTER_PATH}")
        return errors