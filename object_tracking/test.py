import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

from dataset import VOT14Reader
from trackers import ColorTracker, NumpyColorTracker, SIFTTracker, ScaleSpaceTracker

from scratch.meanshift import meanshift
from scratch.camshift import camshift


from functools import partial

dataset = VOT14Reader(dataset_path="./vot14")

sample = dataset[13] # 13 for ball
polygon = sample[1][0]
frame = sample[0][0]

# tracker = partial(cv2.CamShift, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
tracker = meanshift
# tracker = partial(cv2.meanShift, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

# color_tracker = NumpyColorTracker(image=frame, roi_window=polygon, tracker=tracker)
color_tracker = SIFTTracker(image=frame, roi_window=polygon, track_core=tracker)
# color_tracker = ScaleSpaceTracker(image=frame, roi_window=polygon, tracker=tracker, scale_range=[-2, 8, 1])


fig, ax = plt.subplots()
images = []
for frame in sample[0][1:]:
    color_tracker.track(frame)
    polygon = color_tracker.get_current_window()
    x, y, w, h = polygon
    # break
    # final_image = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 3)
    # images.append([ax.imshow(final_image, animated=True)])