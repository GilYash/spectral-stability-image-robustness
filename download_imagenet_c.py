from pathlib import Path
import urllib.request
import tarfile

BASE_URL = "https://zenodo.org/records/2235448/files"

FILES = [
    "noise.tar",
    "blur.tar",
    "weather.tar",
    "digital.tar",
    "extra.tar",   # optional
]

DATA_DIR = Path("data/imagenet-c")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILES:
    archive_path = DATA_DIR / filename
    url = f"{BASE_URL}/{filename}?download=1"

    if not archive_path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print(f"{filename} already downloaded.")

    print(f"Extracting {filename}...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(DATA_DIR)

print("Done.")