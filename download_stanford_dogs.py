"""Download and prepare the official Stanford Dogs images and split lists."""
from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import tempfile


ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "stanford_dogs_dataset"
URLS = {
    "images.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
}


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract an archive while rejecting paths that escape the destination."""
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    archive.extractall(destination)


def main() -> None:
    images = DATASET_DIR / "images" / "Images"
    train_list = DATASET_DIR / "train_list.mat"
    test_list = DATASET_DIR / "test_list.mat"
    if images.is_dir() and train_list.is_file() and test_list.is_file():
        print("Stanford Dogs data is already prepared.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stanford-dogs-") as temp:
        temp_dir = Path(temp)
        for filename, url in URLS.items():
            archive_path = temp_dir / filename
            print(f"Downloading {filename} from Stanford (this may take a while)...", flush=True)
            urlretrieve(url, archive_path)
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
