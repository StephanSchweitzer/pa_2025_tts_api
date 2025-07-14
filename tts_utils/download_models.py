from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    directories = [
        Path("../models/xtts_v2"),
        Path("../models/adapters")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_base_xtts():
    logger.info("Downloading base XTTS v2 model...")
    try:
        models_dir = Path("../models/xtts_v2")
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=str(models_dir),
        )
        logger.info(f"✅ Base XTTS model downloaded successfully to: {models_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to download base XTTS model: {e}")
        raise


def download_voicecast_adapter():
    logger.info("Downloading voicecast emotional adapter...")
    try:
        adapter_dir = Path("../models/adapters")
        snapshot_download(
            repo_id="StephanSchweitzer/pa_2025_finetuned_emotion_xtts_v2",
            local_dir=str(adapter_dir),
            resume_download=True
        )
        logger.info(f"Voicecast adapter downloaded successfully to: {adapter_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to download voicecast adapter: {e}")
        raise


def verify_downloads():
    logger.info("Verifying downloads...")

    base_model_path = Path("../models/xtts_v2")
    if base_model_path.exists() and any(base_model_path.iterdir()):
        logger.info(f"Base model found in {base_model_path}")
    else:
        logger.error(f"Base model not found in {base_model_path}")
        return False

    adapter_path = Path("../models/adapters")
    if adapter_path.exists() and any(adapter_path.iterdir()):
        logger.info(f"Adapter found in {adapter_path}")
        files = [f.name for f in adapter_path.iterdir()]
        logger.info(f"Adapter files: {files}")
    else:
        logger.error(f"Adapter not found in {adapter_path}")
        return False

    return True


def main():
    logger.info("Starting model download process...")

    try:
        create_directories()

        download_base_xtts()
        download_voicecast_adapter()

        if verify_downloads():
            logger.info("All models downloaded and verified successfully!")
            return True
        else:
            logger.error("Download verification failed")
            return False

    except Exception as e:
        logger.error(f"Download process failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)