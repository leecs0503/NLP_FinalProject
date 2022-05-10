import os
from PIL import Image
from src.preprocess import image_preprocess


def test_resize_image():
    image_dir = os.path.join(".", "tests", "preprocess", "image_example")
    original_image_dir = os.path.join(image_dir, "original")
    result_image_dir = os.path.join(image_dir, "result")
    image_size = 224

    image_preprocess.resize_images(
        original_image_dir, result_image_dir, [image_size, image_size]
    )

    for dir in os.scandir(result_image_dir):
        images = os.listdir(dir.path)
        for image in images:
            with open(os.path.join(dir.path, image), "r+b") as f:
                with Image.open(f) as img:
                    assert img.size == (image_size, image_size)
