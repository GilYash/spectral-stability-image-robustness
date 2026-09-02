from pathlib import Path
import urllib.request
import tarfile

BASE_URL = "https://zenodo.org/records/2235448/files"

FILES = [
    "noise.tar",
    "blur.tar",
    "weather.tar",
    "digital.tar",
    "extra.tar",
]

DATA_DIR = Path("data/imagenet-c")
ARCHIVE_DIR = DATA_DIR.parent
DATA_DIR.mkdir(parents=True, exist_ok=True)


def archive_needs_extraction(archive_path):
    with tarfile.open(archive_path) as tar:
        files = [member for member in tar.getmembers() if member.isfile()]

    for member in files:
        parts = Path(member.name).parts
        if parts and parts[0] == DATA_DIR.name:
            parts = parts[1:]

        if not parts or not (DATA_DIR.joinpath(*parts)).is_file():
            return True

    return False


for filename in FILES:
    archive_path = ARCHIVE_DIR / filename
    url = f"{BASE_URL}/{filename}?download=1"

    if not archive_path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print(f"{filename} already downloaded.")

    if archive_needs_extraction(archive_path):
        print(f"Extracting {filename}...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(DATA_DIR)
    else:
        print(f"{filename} already extracted.")

print("Done.")