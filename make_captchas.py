import os
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import random


def make_captcha(n_digits: int, font: ImageFont) -> None:
    i = random.randint(0, 10 ** n_digits)
    target = f"{i:06}"

    dirname = "./data/grayscale-digits" + target
    if os.path.exists(dirname):
        return make_captcha(n_digits, font)

    img = Image.new("RGB", (200, 80), (0, 0, 0))
    drawing = ImageDraw.Draw(img)
    drawing.text((10, 10), target, fill=(255, 255, 255), font=font)

    os.makedirs(dirname)
    img.save(dirname + "/0.png")


def main(n_images: int, n_digits: int, path_to_font: str) -> None:
    font = ImageFont.truetype(path_to_font, 48)
    for _ in range(n_images):
        make_captcha(n_digits, font)


if __name__ == "__main__":
    n_images = 10_000
    n_digits = 6
    path_to_font = "./lib/Comic Sans MS.ttf"
    main(n_images, n_digits, path_to_font)
