import os
from PIL import Image


def resize_images(input_dir: str, output_dir: str, size: int):
    assert os.path.isdir(input_dir)
    for dir in os.scandir(input_dir):
        if not dir.is_dir():
            continue
        output_path = os.path.join(output_dir, dir.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        images = os.listdir(dir.path)
        n_images = len(images)
        for idx, image in enumerate(images):
            with open(os.path.join(dir.path, image), "r+b") as f:
                with Image.open(f) as img:
                    img = img.resize(size, Image.ANTIALIAS)
                    img.save(os.path.join(output_path, image), img.format)
            if (idx + 1) % 1000 == 0:
                print(
                    "[{}/{}] Resized the images and saved into '{}'.".format(
                        idx + 1, n_images, output_path
                    )
                )
