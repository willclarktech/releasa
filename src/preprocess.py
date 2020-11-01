import cv2  # type: ignore
import glob
from PIL import Image, ImageFilter, ImageOps  # type: ignore

if __name__ == "__main__":
    global debug

    input_dir = "./data/rgb-digits-train-source/"
    path_glob_to_imgs = input_dir + "**/0.png"
    img_paths = glob.glob(path_glob_to_imgs)

    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.filter(ImageFilter.FIND_EDGES)
        img = ImageOps.grayscale(img)
        img.show()
        # img = cv2.imread(img_path)

        # converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # converted[:,:,0] = 255
        # cv2.imwrite("xxx.png", converted)
        raise Exception("xxx")
