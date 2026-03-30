"""
Download model checkpoints from Google Drive.

Usage:
    python download_checkpoints.py

The checkpoints zip is hosted on Google Drive. This script downloads
and extracts it into the checkpoints/ directory.
"""
import os
import sys
import zipfile
import tarfile
import shutil
import requests

# Google Drive file ID extracted from the sharing URL
GDRIVE_FILE_ID = "1RsW05ER-jj5cf_G3lgjPq_qJA9n_5G3l"
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DOWNLOAD_PATH = os.path.join(CHECKPOINTS_DIR, "_download_tmp")


def download_from_gdrive(file_id: str, destination: str):
    """Download a file from Google Drive, handling the virus scan confirmation page."""
    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download", "confirm": "t"}

    print(f"  Downloading from Google Drive (file ID: {file_id})...")
    response = session.get(base_url, params=params, stream=True)

    # Handle confirmation token for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params["confirm"] = value
            response = session.get(base_url, params=params, stream=True)
            break

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 32768

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  {mb:.1f} MB downloaded...", end="", flush=True)

    print()
    print(f"  Download complete: {destination}")


def extract_archive(archive_path: str, extract_to: str):
    """Extract zip or tar.gz archive."""
    print(f"  Extracting to {extract_to}...")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
    else:
        # Might be a single .pt file
        print(f"  Not an archive, treating as single checkpoint file.")
        dest = os.path.join(extract_to, "best_model.pt")
        shutil.copy2(archive_path, dest)
        return

    # If extraction created a single subdirectory, move contents up
    items = os.listdir(extract_to)
    non_tmp = [i for i in items if not i.startswith("_download")]
    if len(non_tmp) == 1 and os.path.isdir(os.path.join(extract_to, non_tmp[0])):
        subdir = os.path.join(extract_to, non_tmp[0])
        for item in os.listdir(subdir):
            src = os.path.join(subdir, item)
            dst = os.path.join(extract_to, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(subdir)


def checkpoints_exist() -> bool:
    """Check if at least one checkpoint file exists."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return False

    for root, dirs, files in os.walk(CHECKPOINTS_DIR):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pkl") or f.endswith(".joblib"):
                return True
    return False


def list_checkpoints():
    """List all checkpoint files."""
    found = []
    for root, dirs, files in os.walk(CHECKPOINTS_DIR):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pkl") or f.endswith(".joblib"):
                rel = os.path.relpath(os.path.join(root, f), CHECKPOINTS_DIR)
                size_mb = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
                found.append((rel, size_mb))
    return found


def ensure_checkpoints():
    """Download checkpoints if they don't exist."""
    if checkpoints_exist():
        print("Checkpoints already present:")
        for path, size in list_checkpoints():
            print(f"  {path} ({size:.1f} MB)")
        return True

    print("=" * 50)
    print("  Downloading model checkpoints...")
    print("=" * 50)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    archive_path = DOWNLOAD_PATH

    try:
        download_from_gdrive(GDRIVE_FILE_ID, archive_path)
        extract_archive(archive_path, CHECKPOINTS_DIR)

        # Cleanup temp file
        if os.path.exists(archive_path):
            os.remove(archive_path)

        if checkpoints_exist():
            print("\nCheckpoints ready:")
            for path, size in list_checkpoints():
                print(f"  {path} ({size:.1f} MB)")
            return True
        else:
            print("\nERROR: No checkpoint files found after extraction.")
            return False

    except Exception as e:
        print(f"\nERROR downloading checkpoints: {e}")
        print("Please download manually from:")
        print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        print(f"Extract into: {CHECKPOINTS_DIR}/")
        return False


if __name__ == "__main__":
    success = ensure_checkpoints()
    sys.exit(0 if success else 1)
