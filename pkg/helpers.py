import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def get_roi_from_frame(frame, fromCenter=False):
    r = cv2.selectROI(frame, fromCenter=fromCenter)
    cv2.destroyAllWindows()

    # If ROI invalid, return whole image
    if len(np.unique(r)) < 4:
        r = np.zeros(4).astype("int")
        r[3] = len(frame)
        r[2] = len(frame[0])

    return np.array(r).astype("int")


def get_roi_from_video(vidPath, fromCenter=False):
    vs = cv2.VideoCapture(vidPath)
    ret, frame = vs.read()

    return get_roi_from_frame(frame, fromCenter)


def crop_with_roi(frame, roi):
    return frame[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]


def show_img(frame: np.ndarray):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    plt.axis("off")


def get_output_vid_frame_size(vidPath, pipeline, outputHeight):
    cap = cv2.VideoCapture(vidPath)
    _, frame = cap.read()
    frame = pipeline(frame)
    frame = imutils.resize(frame, height=outputHeight)
    return frame.shape[-2::-1]
