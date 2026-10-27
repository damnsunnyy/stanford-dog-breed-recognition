"""Download and prepare the official Stanford Dogs images and split lists."""
from pathlib import Path
from urllib.request import Request, urlopen
import tarfile
import time


ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "stanford_dogs_dataset"
URLS = {
    "images.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
}
ARCHIVE_DIR = ROOT / ".stanford_dogs_downloads"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract an archive while rejecting paths that escape the destination."""
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    archive.extractall(destination)


def download(url: str, destination: Path) -> None:
    """Download with retries and resume a partial file when the server allows it."""
    for attempt in range(1, 6):
        offset = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                # If Range was ignored, restart instead of appending a duplicate archive.
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with destination.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            return
        except Exception as error:
            if attempt == 5:
                raise RuntimeError(
                    f"Could not download {url}. A partial file is saved at {destination}; "
                    "rerun this script to retry/resume, or download the archive in a browser "
                    "and place it at that path."
                ) from error
            print(f"Download interrupted ({error}); retrying ({attempt}/5)...", flush=True)
            time.sleep(min(attempt * 3, 12))


def main() -> None:
    images = DATASET_DIR / "images" / "Images"
    train_list = DATASET_DIR / "train_list.mat"
    test_list = DATASET_DIR / "test_list.mat"
    if images.is_dir() and train_list.is_file() and test_list.is_file():
        print("Stanford Dogs data is already prepared.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        archive_path = ARCHIVE_DIR / filename
        valid_archive = archive_path.is_file() and tarfile.is_tarfile(archive_path)
        if not valid_archive:
            print(f"Downloading {filename} from Stanford (this may take a while)...", flush=True)
            download(url, archive_path)
        print(f"Extracting {filename}...", flush=True)
        with tarfile.open(archive_path) as archive:
            target = DATASET_DIR / "images" if filename == "images.tar" else DATASET_DIR
            target.mkdir(parents=True, exist_ok=True)
            safe_extract(archive, target)

    # lists.tar contains a `lists/` directory; the training and evaluation
    # scripts use the two official split files directly from the dataset root.
    lists_dir = DATASET_DIR / "lists"
    for name in ("train_list.mat", "test_list.mat"):
        source = lists_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Expected {name} in the extracted Stanford lists archive")
        source.replace(DATASET_DIR / name)
    print(f"Dataset ready at {DATASET_DIR}")


if __name__ == "__main__":
    main()
