import hashlib
from functools import lru_cache

# Helper function to calc md5 sum (unique fingerprint of text)
@lru_cache(maxsize=None)
def md5sum(text: str) -> str:
    text_md5 = hashlib.md5(text.encode('utf-8'))
    return text_md5.hexdigest()

def harmonic_mean(x: float, y: float) -> float:
    return ((2 * x * y) / (x + y))