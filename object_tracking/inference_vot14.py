import cv2

from dataset import VOT14Reader
from trackers import ColorMSTracker


dataset = VOT14Reader(dataset_path="./vot14")

sample = dataset[1] # 13 for ball
polygon = sample[1][0]
frame = sample[0][0]

roi = frame[polygon[1]:polygon[1] + polygon[3],
            polygon[0]:polygon[0] + polygon[2]]

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
tracker = ColorMSTracker(tracked_window=roi, term_crit=term_crit)

for frame in sample[0][1:]:
    polygon = tracker.track(frame, polygon)
    x, y, w, h = polygon
    final_image = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 3)
    cv2.imshow('final_image', final_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
