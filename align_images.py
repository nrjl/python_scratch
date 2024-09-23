import argparse
import logging

import cv2
import imutils
import numpy as np
from tqdm import tqdm


def align_images(im1, im2, n_features=500, match_proportion=0.2, show_matches=False):
    # Assume inputs are greyscale

    # get orb features
    orb = cv2.ORB_create(n_features)
    kp1, descriptors1 = orb.detectAndCompute(im1, None)
    kp2, descriptors2 = orb.detectAndCompute(im2, None)

    # match the features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score and get top
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[: int(len(matches) * match_proportion)]

    # visualise if needed
    if show_matches:
        match_im = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)
        match_im = imutils.resize(match_im, width=1000)
        cv2.imshow("ORB keypoint matches", match_im)
        cv2.waitKey(0)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = points1.copy()

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    aligned = cv2.warpPerspective(im1, h, im2.shape)
    return aligned


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="Optional index for reference image"
    )
    parser.add_argument(
        "-n",
        "--max-frames",
        type=int,
        default=-1,
        help="Optional index for reference image",
    )
    parser.add_argument("video", help="Video file")
    args = vars(parser.parse_args())

    cap = cv2.VideoCapture(args["video"])

    # Get the index frame and total number of frames (CAP_PROP_FRAME_COUNT doesn't work)
    n_frames = 0
    while True:
        res, frame = cap.read()
        if not res:
            break
        if n_frames == args["index"]:
            ref_frame = frame.copy()
        n_frames += 1
    # reset reader
    cap.set(cv2.CAP_PROP_POS_FRAMES, -1)

    logger.info(f"Found {n_frames} frames.")
    assert n_frames >= 2, "Must be at least two images"

    # cv2.imshow("Reference frame (any key to continue)", ref_frame)
    # cv2.waitKey(0)

    # Make output
    out = cv2.VideoWriter(
        "stable.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (int(cap.get(3)), int(cap.get(4))),
    )
    if (args["max_frames"] > 0) and (args["max_frames"] < n_frames):
        n_frames = args["max_frames"]

    # convert to grayscale
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(n_frames), desc="Iterating over images"):
        res, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aligned = align_images(frame_gray, ref_gray, show_matches=False)
        # cv2.imwrite(f"aligned/{i:03d}.png", aligned)
        out.write(aligned)

    cap.release()
    out.release()