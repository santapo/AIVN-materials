from utils.utils import compute_iou
import cv2
import logging
from tqdm import tqdm

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

    def _init_tracker(self, tracker, sample, frame_index):
        polygon = sample[1][frame_index]
        frame = sample[0][frame_index]

        roi = frame[polygon[1]:polygon[1] + polygon[3],
                    polygon[0]:polygon[0] + polygon[2]]
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        tracker = tracker(tracked_window=roi, term_crit=term_crit)
        return tracker, polygon

    def evaluate_by_index(self,
                          tracker_name,
                          sample_index,
                          skip_after_reinit=5,
                          ignore_after_reinit=5,
                          iou_threshold=0.7):
        tracker = self.get_tracker(tracker_name)

        sample = self.dataset_reader[sample_index]
        tracker, polygon = self._init_tracker(tracker, sample, frame_index=0)

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
        for frame_index, (frame, polygon_gt) in tqdm(enumerate(zip(sample[0], sample[1])), desc="Evaluating"):
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
                tracker, polygon = self._init_tracker(tracker, sample, frame_index)
                record["status"].append("reinitialize")
                record["iou_records"].append(0)
                reinitialize=False
                continue
            if frame_index in ignore_frames:
                record["status"].append("ignore")
                record["iou_records"].append(0)
                continue

            polygon = tracker.track(frame, polygon)
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

    # def plot_accuracy_by_frame():
    #     ...

    # def plot_robustness():
    #     ...
