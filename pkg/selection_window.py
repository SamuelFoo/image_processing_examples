import cv2

from pkg.circles import get_circle_from_3_pts


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


class CircleSelectionWindow(SelectionWindow):

    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.min_points_left = 3
        self.centerx, self.centery, self.radius = 0, 0, 0
        self.frameCopy = self.frame.copy()
        self.selected_points = []

    def callback_func(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.min_points_left -= 1
            self.selected_points.append((x, y))
            cv2.circle(self.frame, (x, y), 2, (255, 0, 0), thickness=1)

            if len(self.selected_points) >= 3:
                x1, y1, x2, y2, x3, y3 = (
                    *self.selected_points[-3],
                    *self.selected_points[-2],
                    *self.selected_points[-1],
                )
                self.centerx, self.centery, self.radius = get_circle_from_3_pts(
                    (x1, y1), (x2, y2), (x3, y3)
                )

                # New frame to "remove" drawn circle in last frame
                self.frame = self.frameCopy.copy()

                cv2.circle(
                    self.frame,
                    (round(self.centerx), round(self.centery)),
                    round(self.radius),
                    (255, 0, 0),
                    thickness=1,
                )

            cv2.imshow(self.title, self.frame)

    def get_circle(self):
        return (self.centerx, self.centery, self.radius)
