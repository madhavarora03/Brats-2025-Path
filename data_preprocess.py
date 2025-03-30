import os
import zipfile
from typing import List

from PIL import Image, UnidentifiedImageError
from colorama import Fore, Style
from tqdm.auto import tqdm

ZIP_PATH = "data/BraTS2024-Path-Challenge-TrainingData.zip"
UNZIP_DIR = "data"


def delete_corrupt_images(directory: str) -> List[str]:
    """
    Recursively checks and deletes corrupt or truncated image files in a directory.

    Args:
        directory (str): Path to the root directory to check.

    Returns:
        list: List of deleted corrupted files (if any).
    """
    deleted_files = []
    all_images = []

    # Collect all image file paths first for tqdm
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                all_images.append(os.path.join(root, file))

    print(f"{Fore.CYAN}Scanning {len(all_images)} images...{Style.RESET_ALL}")

    for file_path in tqdm(all_images, desc="Checking images", unit="file", colour="green"):
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check for metadata corruption
            with Image.open(file_path) as img:
                img.load()  # Check for truncation
        except (UnidentifiedImageError, OSError) as e:
            print(
                f"\n{Fore.RED}Corrupt or truncated image found:{Style.RESET_ALL} {file_path}. {Fore.RED}Error:{Style.RESET_ALL} {e}")
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"{Fore.GREEN}Deleted:{Style.RESET_ALL} {file_path}")
            except Exception as del_error:
                print(f"{Fore.YELLOW}Failed to delete{Style.RESET_ALL} {file_path}: {del_error}")

    if not deleted_files:
        print(f"{Fore.BLUE}No corrupted or truncated images found.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.CYAN}Deleted {len(deleted_files)} corrupted or truncated images.{Style.RESET_ALL}\n")

    return deleted_files

if __name__ == "__main__":
    if os.path.exists(ZIP_PATH):
        print("Unzipping the dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(UNZIP_DIR)
        print(f"Data unzipped to {UNZIP_DIR}")

        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            print(f"Removed the zip file: {ZIP_PATH}")
    else:
        print(f"Zip file not found: {ZIP_PATH}")

    delete_corrupt_images(UNZIP_DIR)
