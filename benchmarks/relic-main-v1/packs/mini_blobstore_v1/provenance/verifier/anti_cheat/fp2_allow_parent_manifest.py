from pathlib import Path

path = Path("blobstore/manifests.py")
text = path.read_text()
needle = 'raise ValueError("invalid manifest name")'
if needle not in text:
    raise SystemExit("name validation not found")
path.write_text(text.replace(needle, "return", 1))
