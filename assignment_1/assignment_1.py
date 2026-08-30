import cv2
from pathlib import Path


def print_image_information(image):
    height, width, channels = image.shape

    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {image.size}")
    print(f"data type: {image.dtype}")


def save_camera_information():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return

    success, frame = camera.read()

    if not success:
        print("Could not read frame from camera.")
        camera.release()
        return

    fps = camera.get(cv2.CAP_PROP_FPS)

    height, width = frame.shape[:2]

    solutions_directory = Path(__file__).parent / "solutions"
    solutions_directory.mkdir(exist_ok=True)

    output_file = solutions_directory / "camera_outputs.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")

    camera.release()

    print(f"Camera information saved to: {output_file}")


def main():
    image_path = Path(__file__).parent / "iris-1.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load iris-1.jpg")
        return

    print("Image information:")
    print_image_information(image)

    print()
    save_camera_information()


if __name__ == "__main__":
    main()