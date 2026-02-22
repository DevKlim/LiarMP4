# my_vision_process.py (Stub for HF Spaces / Lite Mode)
import logging

logger = logging.getLogger(__name__)

# Dummy client
client = None

def process_vision_info(messages, return_video_kwargs=False, client=None):
    """
    Stub function to prevent ImportErrors in API-only mode.
    If this is called, it means LITE_MODE logic failed or was bypassed.
    """
    logger.warning("process_vision_info called in LITE/API environment. Returning empty placeholders.")
    if return_video_kwargs:
        return None, None, {"fps": [0]}
    return None, None