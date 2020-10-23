#!/usr/bin/env python3

import os
import sys
import numpy as np
import glob
import argparse
from random import randint, choice
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps  # type: ignore


# def random_color() -> Tuple:
def random_color() -> (int, int, int):
    return (randint(0, 245), randint(0, 245), randint(0, 245))


def make_captcha(
    dirname: str, n_digits: int, font_file: str, random_noise: float
) -> None:

    font = ImageFont.truetype(font_file, font_size)

    i = randint(0, 10 ** n_digits)
    target = f"{i:06}"

    subdirname = dirname + target
    if os.path.exists(subdirname):
        return make_captcha(dirname, n_digits, font, random_noise)

    # img = Image.new("RGB", (width, height), (0, 0, 0))
    img = Image.new("RGB", (width, height), "black")

    drawing = ImageDraw.Draw(img)
    if random_noise > 0.0:
        add_noise(drawing, random_noise)

    text_color = random_color()
    drawing.text((10, 10), target, fill=text_color, font=font)

    if debug:
        img.show()
        sys.exit()
    else:
        os.makedirs(subdirname)
        img.save(subdirname + "/0.png")


def add_noise(draw, random_noise: float) -> None:
    # TODO: dont hard code ...
    cmin = int(randint(50, 70) * random_noise)
    cmax = int(randint(90, 120) * random_noise)
    for _ in range(cmin, cmax):
        diam = randint(5, 11)
        x, y = randint(0, width), randint(0, height)
        draw.ellipse([x, y, x + diam, y + diam], fill=random_color())


def main(
    train_dirname: str,
    test_dirname: str,
    m_train_images: int,
    n_test_images: int,
    n_digits: int,
    font_files: list,
    random_noise: int,
) -> None:

    for _ in range(m_train_images):

        make_captcha(train_dirname, n_digits, choice(font_files), random_noise)
    for _ in range(n_test_images):
        make_captcha(test_dirname, n_digits, choice(font_files), random_noise)


if __name__ == "__main__":
    global debug
    width = 200
    height = 80
    train_dirname = "./data/rgb-digits-train/"
    test_dirname = "./data/rgb-digits-test/"
    m_train_images = 10_000

    n_test_images = 1000
    n_digits = 6
    path_to_deafult_font = "./lib/fonts/Comic Sans MS.ttf"
    font_files = [path_to_deafult_font]
    path_glob_to_fonts = "./lib/fonts/*.ttf"
    random_noise = 0.0
    font_size = 44


    parser = argparse.ArgumentParser(
        description="make captcha training data for releasa"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug mode: Generate only one image. Dont save only show and exit.",
    )
    parser.add_argument(
        "-n",
        "--n_test_images",
        type=int,
        default=n_test_images,
        help="number of test captchas to create",
    )
    parser.add_argument(
        "-m",
        "--m_train_images",
        type=int,
        default=m_train_images,

        help="number of train captchas to create",
    )
    parser.add_argument(
        "-r",
        "--random-noise",
        type=float,
        default=random_noise,
        help="add random noise to image. n >= 0; (~0.1-10)",
    )
    parser.add_argument(
        "-f",
        "--font-randomize",
        action="store_true",
        help="randomized fonts for each captcha",
    )

    args = parser.parse_args()
    debug = args.debug
    if args.font_randomize:
        font_files = glob.glob(path_glob_to_fonts)

    main(
        train_dirname,
        test_dirname,
        args.m_train_images,

        args.n_test_images,
        n_digits,
        font_files,
        args.random_noise,
    )

