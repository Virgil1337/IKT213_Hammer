from pathlib import Path

import cv2
import numpy as np


def padding(image, border_width):
    return cv2.copyMakeBorder(
        image, border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT,
    )


def crop(image, x_0, x_1, y_0, y_1):
    height, width = image.shape[:2]
    return image[y_0:y_1, x_0:x_1]


def resize(image, width, height):
    return cv2.resize(image, (width, height))


def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                emptyPictureArray[y, x, channel] = image[y, x, channel]
    return emptyPictureArray


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hue_shifted(image, emptyPictureArray, hue):
    shifted = image.astype(np.int32) + hue
    emptyPictureArray[:] = np.clip(shifted, 0, 255).astype(np.uint8)
    return emptyPictureArray


def smoothing(image):
    return cv2.GaussianBlur(
        image, (15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT
    )


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)


def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "iris.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    height, width, channels = image.shape
    print(f"Original: width={width}, height={height}, channels={channels}")
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    shift_destination = np.zeros((height, width, 3), dtype=np.uint8)

    results = {
        "01_padding.png": padding(image, 100),
        "02_crop.png": crop(image, 200, width - 130, 200, height - 130),
        "03_resize.png": resize(image, 200, 200),
        "04_copy.png": copy(image, emptyPictureArray),
        "05_grayscale.png": grayscale(image),
        "06_hsv_channels.png": hsv(image),
        "07_color_shift.png": hue_shifted(image, shift_destination, 50),
        "08_smoothing.png": smoothing(image),
        "09_rotation_180.png": rotation(image, 180),
        "10_rotation_90_clockwise.png": rotation(image, 90),
    }
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    for filename, result in results.items():
        path = results_dir / filename
        print(f"Saved {filename}: shape={result.shape}")
    print(f"Finished. Open the images in: {results_dir}")


if __name__ == "__main__":
    main()
