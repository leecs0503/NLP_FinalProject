import os
from PIL import Image


def resize_images(input_dir, output_dir, size):
    assert os.path.isdir(input_dir)
    assert os.path.isdir(output_dir)
    assert size > 0
    for dir in os.scandir(input_dir):
        if not dir.is_dir():
            continue
        if not os.path.exists(output_dir + '/' + dir.name):
            os.makedirs(output_dir + '/' + dir.name)
        images = os.listdir(dir.path)
        n_images = len(images)
        for idx, image in enumerate(images):
            with open(os.path.join(dir.path, image), 'r+b') as f:
                with Image.open(f) as img:
                    img = img.resize(size, Image.ANTIALIAS)
                    img.save(os.path.join(output_dir + '/' + dir.name, image), img.format)
            if (idx + 1) % 1000 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(idx + 1, n_images, output_dir + '/' + dir.name))

