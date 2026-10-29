from pathlib import Path

import cv2
import numpy as np


def harris_corner_detection(reference_image):
    gray = np.float32(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    marked = reference_image.copy()
    marked[corners > 0.01 * corners.max()] = (0, 0, 255)
    return marked


def align_images(image_to_align, reference_image, max_features, good_match_percent):
    cv2.setRNGSeed(0)
    gray_image = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)
    keypoints_reference, descriptors_reference = sift.detectAndCompute(gray_reference, None)

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = matcher.knnMatch(descriptors_image, descriptors_reference, k=2)
    good_matches = [
        first for first, second in pairs
        if first.distance < good_match_percent * second.distance
    ]

    if len(good_matches) > max_features:
        source = np.float32([
            keypoints_image[match.queryIdx].pt for match in good_matches
        ]).reshape(-1, 1, 2)
        destination = np.float32([
            keypoints_reference[match.trainIdx].pt for match in good_matches
        ]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
        height, width = reference_image.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, homography, (width, height))
        matches = cv2.drawMatches(
            image_to_align, keypoints_image,
            reference_image, keypoints_reference,
            good_matches, None,
            matchColor=(0, 255, 0),
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return aligned, matches


def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    image_to_align = cv2.imread(str(project_dir / "align_this.jpg"))
    reference_image = cv2.imread(str(project_dir / "reference_img.png"))

    harris = harris_corner_detection(reference_image)
    aligned, matches = align_images(image_to_align, reference_image, 10, 0.7)

    cv2.imwrite(str(results_dir / "harris.png"), harris)
    cv2.imwrite(str(results_dir / "aligned.png"), aligned)
    cv2.imwrite(str(results_dir / "matches.png"), matches)


if __name__ == "__main__":
    main()
