import hashlib
from functools import lru_cache
from pathlib import Path
import os

# Helper function to calc md5 sum (unique fingerprint of text)
@lru_cache(maxsize=None)
def md5sum(text: str) -> str:
    text_md5 = hashlib.md5(text.encode('utf-8'))
    return text_md5.hexdigest()

def harmonic_mean(x: float, y: float) -> float:
    return ((2 * x * y) / (x + y))

def get_data_dir() -> Path:
    """ 
        The High Performance Cluser has an
        environment variable `$DATA` set that points to the data partition.
        We evaluate this environment variable, if it doesn't exist, we just use
        the current working directory
    """
    data_dir = os.environ.get('DATA')
    if not data_dir:
        return Path.cwd()
    return Path(data_dir)