import cv2


class SelectionWindow:

    def __init__(self, title, frame):
        self.title = title
        self.frame = frame.copy()

        self.minPointsLeft = 0
        self.func = self.callback_func

        self.selectionPts = []

    def displayWindow(self):
        cv2.namedWindow(self.title)
        if self.func != None:
            cv2.setMouseCallback(self.title, self.func)
        cv2.imshow(self.title, self.frame)

        while True:
            key = cv2.waitKey(0)

            if self.minPointsLeft <= 0 and key == ord("q") or key == ord("Q"):
                cv2.destroyWindow(self.title)
                break

    def callback_func(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.selectionPts.append((x, y))
            cv2.circle(self.frame, (x, y), 2, (255, 255, 255), thickness=1)
            cv2.imshow(self.title, self.frame)
