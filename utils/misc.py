import hashlib

# Helper function to calc md5 sum (unique fingerprint of text)
def md5sum(text: str) -> str:
    text_md5 = hashlib.md5(text.encode('utf-8'))
    return text_md5.hexdigest()