from pathlib import Path

path = Path("blobstore/uploads.py")
text = path.read_text()
needle = "if previous is not None and previous != object_id:"
if needle not in text:
    raise SystemExit("part conflict guard not found")
path.write_text(text.replace(needle, "if False:", 1))
