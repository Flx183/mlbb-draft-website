
import re

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def pagename_to_filename(pagename: str) -> str:
    return pagename.replace("/", "_")