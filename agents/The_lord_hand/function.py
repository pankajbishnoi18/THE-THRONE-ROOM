from pathlib import Path
from dotenv import load_dotenv


def read_file(path):
    return Path(path).read_text(encoding="utf-8")


