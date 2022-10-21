from curses import window
import logging

import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.utils import compute_iou

logger = logging.getLogger()


class VOT14Evaluator:
    def __init__(self, dataset_reader):
        self.dataset_reader = dataset_reader 
        self.all_trackers = {}
        self.all_records = {}
        
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def set_tracker(self, tracker_name, tracker):
        if tracker_name in self.all_trackers.keys():
            raise f"tracker_name: {tracker_name} is exist!"
        self.all_trackers[tracker_name] = tracker
        logger.info(f"Set {tracker} to {tracker_name}!")

    def get_tracker(self, tracker_name):
        return self.all_trackers[tracker_name]

    def evaluate_by_index(self,
                          tracker_name,
                          sample_index,
                          skip_after_reinit=5,
                          ignore_after_reinit=5,
                          iou_threshold=0.7):
        sample = self.dataset_reader[sample_index]
        all_polygons_gt = sample[1]
        all_frames = sample[0]

        tracker = self.get_tracker(tracker_name)
        tracker = tracker(image=all_frames[0], roi_window=all_polygons_gt[0])

        if not sample_index in self.all_records.keys():
            self.all_records[sample_index] = []
        record = {"iou_threshold": iou_threshold,
                  "ignore_after_reinit": ignore_after_reinit,
                  "skip_after_reinit": skip_after_reinit,
                  "iou_records": [],
                  "status": []}

        reinitialize = False
        skip_frames = []
        ignore_frames = []
        for frame_index, (frame, polygon_gt) in tqdm(enumerate(zip(all_frames, all_polygons_gt)), desc="Evaluating"):
            # # skip first 5 frames
            # if frame_index < skip_frames:
            #     continue
            # print(frame_index)
            if frame_index in skip_frames:
                record["status"].append("skip")
                record["iou_records"].append(0)
                continue
            if reinitialize:
                tracker = self.get_tracker(tracker_name)
                tracker = tracker(image=frame, roi_window=polygon_gt)
                record["status"].append("reinitialize")
                record["iou_records"].append(0)
                reinitialize=False
                continue
            if frame_index in ignore_frames:
                record["status"].append("ignore")
                record["iou_records"].append(0)
                continue

            tracker.track(frame)
            polygon = tracker.get_current_window()
            iou_value = compute_iou(polygon, polygon_gt)
            record["status"].append("track")
            record["iou_records"].append(iou_value)
            if iou_value < iou_threshold:
                reinitialize = True
                skip_frames = range(frame_index, frame_index+skip_after_reinit+1)
                ignore_frames = range(frame_index+skip_after_reinit,
                                      frame_index+skip_after_reinit+ignore_after_reinit+1)

        self.all_records[sample_index].append(record)
        logger.info(f"Evaluated record is saved in self.all_records[{sample_index}]!")

    def get_robustness_by_index():
        ...

    def get_accuracy_by_index():
        ...

    def get_accuracy():
        ...

    def get_robustness():
        ...

    def draw_iou_record(self, sample_index, run_index):
        record = self.all_records[sample_index][run_index]
        record_size = len(record["status"])

        track_chunks = []
        chunk = []
        for idx, status in enumerate(record["status"]):
            if status != "track":
                if len(chunk) != 0: track_chunks.append(chunk)
                chunk = []
                continue
            chunk.append(idx)

        fig = plt.figure(figsize=(24, 6))
        ax = plt.axes()

        ax.set(xlim=(0, record_size), ylim=(record["iou_threshold"] * 0.9, 1),
               xlabel='frame', ylabel='IOU', title=f"IOU record of sample {sample_index} - run {run_index}")
        ax.plot([record["iou_threshold"]]*record_size, "b--")
        ax.fill_between(range(0, record_size),[0] * record_size, [record["iou_threshold"]]*record_size, color="red", alpha=0.2)
        for chunk in track_chunks:
            if len(chunk) != 1:
                ax.plot(range(chunk[0], chunk[-1]+1), record["iou_records"][chunk[0]:chunk[-1]+1], "g")
            else:
                ax.scatter(chunk[0], record["iou_records"][chunk[0]], c="g")
            ax.axvline(x=chunk[-1], color='y', linewidth=3, alpha=0.2)
