import logging
import io
import tempfile
import os
import mimetypes
from typing import Tuple, Dict, Union, Optional
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AudioProcessor:
    SUPPORTED_FORMATS = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'audio/vorbis',
        'audio/m4a', 'audio/aac',
        'audio/webm'
    }

    def __init__(self, target_sample_rate: int = 22050,
                 max_duration: float = 30.0,
                 min_duration: float = 0.5):

        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration

    def infer_content_type(self, content_type: Optional[str], filename: Optional[str] = None) -> str:

        if content_type and content_type.lower() != 'none' and content_type in self.SUPPORTED_FORMATS:
            return content_type

        # Try to infer from filename
        if filename:
            inferred_type, _ = mimetypes.guess_type(filename)
            if inferred_type and inferred_type in self.SUPPORTED_FORMATS:
                logger.info(f"Inferred content type '{inferred_type}' from filename '{filename}'")
                return inferred_type

        # Try to infer from file extension directly
        if filename:
            ext = os.path.splitext(filename.lower())[1]
            extension_map = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.m4a': 'audio/m4a',
                '.aac': 'audio/aac',
                '.webm': 'audio/webm'
            }

            if ext in extension_map:
                inferred_type = extension_map[ext]
                logger.info(f"Inferred content type '{inferred_type}' from extension '{ext}'")
                return inferred_type

        logger.warning(f"Could not determine content type for file '{filename}', defaulting to audio/wav")
        return 'audio/wav'

    def validate_audio_format(self, content_type: str, filename: str = None) -> None:
        if content_type not in self.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{content_type}' for file '{filename}'. "
                       f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

    def get_audio_info(self, audio_bytes: bytes, filename: str = None) -> Dict:
        try:
            audio_buffer = io.BytesIO(audio_bytes)

            with sf.SoundFile(audio_buffer) as f:
                info = {
                    'duration': len(f) / f.samplerate,
                    'sample_rate': f.samplerate,
                    'channels': f.channels,
                    'format': f.format,
                    'subtype': f.subtype,
                    'frames': len(f),
                    'size_bytes': len(audio_bytes)
                }

            logger.info(f"Audio info for {filename}: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to read audio info for {filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted audio file '{filename}': {str(e)}"
            )

    def validate_audio_constraints(self, audio_info: Dict, filename: str = None) -> None:
        duration = audio_info['duration']

        if duration > self.max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.1f}s (maximum: {self.max_duration}s) for file '{filename}'"
            )

        if duration < self.min_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short: {duration:.1f}s (minimum: {self.min_duration}s) for file '{filename}'"
            )

        if audio_info['sample_rate'] < 8000:
            raise HTTPException(
                status_code=400,
                detail=f"Sample rate too low: {audio_info['sample_rate']}Hz (minimum: 8000Hz) for file '{filename}'"
            )

    def load_and_preprocess_audio(self, audio_bytes: bytes, filename: str = None) -> Tuple[torch.Tensor, int]:
        try:
            audio_buffer = io.BytesIO(audio_bytes)

            audio_array, sr = librosa.load(
                audio_buffer,
                sr=self.target_sample_rate,
                mono=True,
                dtype=np.float32
            )

            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95

            audio_tensor = torch.from_numpy(audio_array)

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            logger.info(f"Processed audio '{filename}': shape={audio_tensor.shape}, sr={sr}")

            return audio_tensor, sr

        except Exception as e:
            logger.error(f"Failed to process audio '{filename}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Audio processing failed for '{filename}': {str(e)}"
            )

    def process_uploaded_audio(self, audio_bytes: bytes, content_type: Optional[str] = None,
                             filename: Optional[str] = None) -> Tuple[torch.Tensor, int]:
        inferred_content_type = self.infer_content_type(content_type, filename)

        self.validate_audio_format(inferred_content_type, filename)

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail=f"Audio file '{filename}' is empty")

        audio_info = self.get_audio_info(audio_bytes, filename)
        self.validate_audio_constraints(audio_info, filename)

        return self.load_and_preprocess_audio(audio_bytes, filename)

    def save_tensor_as_temp_file(self, audio_tensor: torch.Tensor, sample_rate: int = None) -> str:
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        torchaudio.save(temp_path, audio_tensor.cpu(), sample_rate, format="wav")

        logger.info(f"Saved audio tensor to temporary file: {temp_path}")
        return temp_path

    @staticmethod
    def tensor_to_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_tensor)

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.cpu(), sample_rate, format="wav")
        buffer.seek(0)

        return buffer.getvalue()


def cleanup_temp_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")