from pathlib import Path
import base64

def encode_raw(path):
    """Send original JPEG bytes — no re-encode, no second lossy pass."""
    return base64.b64encode(Path(path).read_bytes()).decode()