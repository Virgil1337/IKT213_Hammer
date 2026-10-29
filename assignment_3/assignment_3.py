from pathlib import Path

import cv2
import numpy as np


def grayscale(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sobel_edge_detection(image):
    blurred = cv2.GaussianBlur(grayscale(image), (3, 3), sigmaX=0)
    edges = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)
    return cv2.convertScaleAbs(edges)


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred = cv2.GaussianBlur(grayscale(image), (3, 3), sigmaX=0)
    return cv2.Canny(blurred, threshold_1, threshold_2)


def template_match(image, template):
    gray_image = grayscale(image)
    gray_template = grayscale(template)
    height, width = gray_template.shape
    scores = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    y_positions, x_positions = np.where(scores >= 0.9)

    if image.ndim == 2:
        marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        marked = image.copy()

    for x, y in zip(x_positions, y_positions):
        cv2.rectangle(
            marked,
            (int(x), int(y)),
            (int(x + width - 1), int(y + height - 1)),
            (0, 0, 255),
            2,
        )

    return marked


def resize(image, scale_factor: int, up_or_down: str):
    result = image.copy()
    for _ in range(scale_factor.bit_length() - 1):
        if up_or_down == "up":
            result = cv2.pyrUp(result)
        else:
            result = cv2.pyrDown(result)
    return result


def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(project_dir / "lambo.png"))
    shapes = cv2.imread(str(project_dir / "shapes.png"))
    template = cv2.imread(str(project_dir / "shapes_template.jpg"))

    results = {
        "01_sobel.png": sobel_edge_detection(image),
        "02_canny.png": canny_edge_detection(image, 50, 50),
        "03_template_match.png": template_match(shapes, template),
        "04_pyramid_up.png": resize(image, 2, "up"),
        "05_pyramid_down.png": resize(image, 2, "down"),
    }

    for filename, result in results.items():
        cv2.imwrite(str(results_dir / filename), result)


if __name__ == "__main__":
    main()
