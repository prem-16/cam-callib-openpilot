import cv2 as cv
import numpy as np
import math
from pathlib import Path

# Parameters for optical flow (tuned slightly)
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Camera intrinsic matrix
FOCAL_LENGTH = 910.0
FRAME_SIZE = (1164, 874)
K = np.array([[FOCAL_LENGTH, 0.0, FRAME_SIZE[0] / 2],
              [0.0, FOCAL_LENGTH, FRAME_SIZE[1] / 2],
              [0.0, 0.0, 1.0]])
K_inv = np.linalg.inv(K)

class Frame:
    def __init__(self, image, pitch: float, yaw: float):
        self.image = image
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.pitch = pitch
        self.yaw = yaw
        self.prev_gray = None

    def calc_pitch_yaw(self, prev_frame=None):
        lines = self.detect_lines()
        if lines is not None and len(lines) > 1:
            return self.calculate_vanishing_point(lines)
        elif prev_frame is not None:
            return self.calc_optical_flow(prev_frame)
        return 0, 0

    def detect_lines(self):
        blur_gray = cv.GaussianBlur(self.gray, (5, 5), 1)
        edges = cv.Canny(blur_gray, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=5)
        return lines

    def calculate_vanishing_point(self, lines):
        vanishing_point = None
        min_error = float('inf')

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                m1, c1 = self.get_line_equation(lines[i])
                m2, c2 = self.get_line_equation(lines[j])

                if m1 != m2:
                    x0, y0 = self.calculate_intersection(m1, c1, m2, c2)
                    error = self.calculate_vanishing_point_error(x0, y0, lines)
                    if error < min_error:
                        min_error = error
                        vanishing_point = (x0, y0)

        if vanishing_point is not None:
            return self.project_vanishing_point(vanishing_point)
        return 0, 0

    def calc_optical_flow(self, prev_frame):
        if prev_frame.prev_gray is None:
            prev_frame.prev_gray = prev_frame.gray

        p0 = cv.goodFeaturesToTrack(prev_frame.gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame.gray, self.gray, p0, None, **lk_params)

        if p1 is None or st is None:
            return 0, 0

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        flow = good_new - good_old
        avg_flow = np.mean(flow, axis=0)

        yaw = np.arctan2(avg_flow[0], FOCAL_LENGTH)
        pitch = np.arctan2(-avg_flow[1], FOCAL_LENGTH)

        return pitch, yaw

    def get_line_equation(self, line):
        x1, y1, x2, y2 = line[0]
        m = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')
        c = y1 - m * x1
        return m, c

    def calculate_intersection(self, m1, c1, m2, c2):
        x0 = (c2 - c1) / (m1 - m2)
        y0 = m1 * x0 + c1
        return x0, y0

    def calculate_vanishing_point_error(self, x0, y0, lines):
        error = 0
        for line in lines:
            m, c = self.get_line_equation(line)
            x_intersect, y_intersect = self.calculate_intersection(m, c, -1/m, y0 - (-1/m) * x0)
            error += (x_intersect - x0)**2 + (y_intersect - y0)**2
        return math.sqrt(error)

    def project_vanishing_point(self, vanishing_point):
        img_pts = np.array([*vanishing_point, 1])
        cam_pts = K_inv.dot(img_pts)
        vp_unproj = cam_pts[:2] / cam_pts[2]

        yaw = np.arctan(vp_unproj[0])
        pitch = np.arctan(-vp_unproj[1]) * np.cos(yaw)

        return pitch, yaw


class Frames:
    def __init__(self, video_path, video_num):
        self.video_path = Path(video_path)
        self.video_num = video_num
        self.frames = self.load_frames()

    def load_frames(self):
        cap = cv.VideoCapture(str(self.video_path.joinpath(f'{self.video_num}.hevc')))
        file = open(self.video_path.joinpath(f'{self.video_num}.txt'), 'r')
        lines = file.readlines()
        frames = []

        for i, line in enumerate(lines):
            ret, img = cap.read()
            if not ret:
                break

            data = line.strip().split()
            pitch = float(data[0])
            yaw = float(data[1])
            frames.append(Frame(img, pitch, yaw))

        assert not cap.read()[0], "Mismatch between frames and labels"
        return frames

    def process_frames(self):
        save_path = None
        prev_frame = None

        for frame in self.frames:
            pitch_pred, yaw_pred = frame.calc_pitch_yaw(prev_frame)
            prev_frame = frame
            print(f"Predicted Pitch: {pitch_pred}, Yaw: {yaw_pred}")
            save_path = self.save_pred(pitch_pred, yaw_pred, save_path)

    def save_pred(self, pitch_pred, yaw_pred, f=None):
        if f is None:
            output_file = self.video_path.joinpath('prediction', f'{self.video_num}.txt')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            f = open(output_file, "w+")

        f.write(f"{pitch_pred} {yaw_pred}\n")
        return f




if __name__ == "__main__":
    path_raw = "./labeled/"
    video_num = 0
    for i in range(5):
        video_num = i;
        frames = Frames(path_raw, video_num)
        frames.process_frames()
        print(f"Processed video {i}")
