import os
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import random


def make_captcha(dirname: str, n_digits: int, font: ImageFont) -> None:
    i = random.randint(0, 10 ** n_digits)
    target = f"{i:06}"

    subdirname = dirname + target
    if os.path.exists(subdirname):
        return make_captcha(dirname, n_digits, font)

    img = Image.new("RGB", (200, 80), (0, 0, 0))
    drawing = ImageDraw.Draw(img)
    drawing.text((10, 10), target, fill=(255, 255, 255), font=font)

    os.makedirs(subdirname)
    img.save(subdirname + "/0.png")


def main(
    train_dirname: str,
    test_dirname: str,
    n_train_images: int,
    n_test_images: int,
    n_digits: int,
    path_to_font: str,
) -> None:
    font = ImageFont.truetype(path_to_font, 48)
    for _ in range(n_train_images):
        make_captcha(train_dirname, n_digits, font)
    for _ in range(n_test_images):
        make_captcha(test_dirname, n_digits, font)


if __name__ == "__main__":
    train_dirname = "./data/grayscale-digits-train/"
    test_dirname = "./data/grayscale-digits-test/"
    n_train_images = 10_000
    n_test_images = 1000
    n_digits = 6
    path_to_font = "./lib/Comic Sans MS.ttf"
    main(
        train_dirname,
        test_dirname,
        n_train_images,
        n_test_images,
        n_digits,
        path_to_font,
    )
