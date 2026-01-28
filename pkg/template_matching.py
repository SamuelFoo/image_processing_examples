import cv2
import numpy as np

from pkg.helpers import crop_with_roi


def get_templates_from_video(vidPath, pipeline, templateWidth, templateHeight):
    box = None
    templates = []

    def on_mouse(event, x, y, flags, userdata):
        nonlocal box
        # Draw box
        if event == cv2.EVENT_LBUTTONDOWN:
            p = (x, y)
            p1 = (int(p[0] - templateWidth / 2), int(p[1] - templateHeight / 2))
            p2 = (int(p[0] + templateWidth / 2), int(p[1] + templateHeight / 2))
            box = [p1[0], p1[1], templateWidth, templateHeight]

            frameDraw = frame.copy()
            cv2.rectangle(frameDraw, p1, p2, (255, 0, 0), 1)
            cv2.imshow("Frame", frameDraw)

    cap = cv2.VideoCapture(vidPath)
    cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Frame", on_mouse)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = pipeline(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("e") or key == ord("E"):
            if box is not None:
                templates.append(crop_with_roi(frame, box))

    cap.release()
    cv2.destroyAllWindows()
    return templates


def get_template_matches(frame, template, confidenceThresh):
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= confidenceThresh)

    confidences = res[loc]

    boxes = []
    w, h = template.shape[-2::-1]
    for pt in zip(*loc[::-1]):
        boxes.append([*pt, pt[0] + w, pt[1] + h])

    return boxes, confidences
