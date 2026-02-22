import whisper
import logging
from pathlib import Path
import os

LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

logger = logging.getLogger(__name__)
transcription_model = None

def load_model():
    if LITE_MODE:
        logger.info("LITE_MODE is enabled. Skipping Whisper model loading.")
        return

    global transcription_model
    if transcription_model is None:
        try:
            logger.info("Loading 'base.en' Whisper model for transcription...")
            transcription_model = whisper.load_model("base.en")
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            transcription_model = None

def generate_transcript(audio_path_str: str) -> str:
    if transcription_model is None:
        logger.warning("Transcription model is not available. Cannot generate transcript.")
        return None

    try:
        audio_path = Path(audio_path_str)
        logger.info(f"Starting transcription for: {audio_path.name}")

        result = transcription_model.transcribe(audio_path_str, verbose=False)

        vtt_path = audio_path.with_suffix('.vtt')

        from whisper.utils import get_writer
        writer = get_writer("vtt", str(vtt_path.parent))
        writer(result, str(audio_path.name))
        
        logger.info(f"Transcription complete. VTT file saved to: {vtt_path}")
        return str(vtt_path)

    except Exception as e:
        logger.error(f"An error occurred during transcription for {audio_path_str}: {e}", exc_info=True)
        return None