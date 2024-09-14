import time
from pathlib import Path

import os
import fire
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from manga_ocr import MangaOcr


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(mocr, img_or_path, write_to):
    t0 = time.time()
    text = mocr(img_or_path)
    t1 = time.time()

    print(f"took {t1 - t0:0.02f}s: {text}")

    if write_to == "clipboard":
        os.system(f"termux-clipboard-set '{text}'")
    else:
        write_to = Path(write_to)
        if write_to.suffix != ".txt":
            raise ValueError('write_to must be either "clipboard" or a path to a text file')

        with write_to.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def get_path_key(path):
    return path, path.lstat().st_mtime


def run(
    read_from="clipboard",
    write_to="clipboard",
    pretrained_model_name_or_path="kha-white/manga-ocr-base",
    delay_secs=1.0,
):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be either saved to system clipboard, or appended to a text file.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", or a path to a text file.
    :param pretrained_model_name_or_path: Path to a trained model, either local or from Transformers' model hub.
    :param verbose: If True, unhides all warnings.
    :param delay_secs: How often to check for new images, in seconds.
    """

    mocr = MangaOcr(pretrained_model_name_or_path)
    os.system("termux-vibrate -d 70 & sleep 0.15 && termux-vibrate -d 70 &")

    read_from = Path(read_from)
    if not read_from.is_dir():
        os.makedirs(read_from)

    print(f"Reading from directory {read_from}")

    old_paths = set()
    for path in read_from.iterdir():
        old_paths.add(get_path_key(path))

    while True:
        for path in read_from.iterdir():
            path_key = get_path_key(path)

            if path_key in old_paths:
                continue

            old_paths.add(path_key)

            if str(path).startswith(".pending-"):
                continue

            try:
                img = Image.open(path)
                img.load()
            except (UnidentifiedImageError, OSError):
                pass
                # print(f"skipping {path}")
            else:
                process_and_write_results(mocr, img, write_to)
                time.sleep(delay_secs * 5)

        time.sleep(delay_secs)


if __name__ == "__main__":
    fire.Fire(run)
