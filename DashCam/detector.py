from collections import deque
from datetime import datetime
import os
import pickle
import time
import cv2
import numpy as np
from scipy.ndimage.measurements import label
from desciptor import Descriptor
from slidingwindow import myslidingWindow, slidingWindow


class Detector:
    def __init__(self, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                 x_range=(0, 1), y_range=(0, 1), scale=1.5, x_ranges={}, y_ranges={}, init_sizes={}, x_overlaps={}, y_steps={}, scales={}):

        self.init_size = init_size
        self.x_overlap = x_overlap
        self.y_step = y_step
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None
        self.classifiers = {}
        self.descriptors = {}
        self.scalers = {}
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        self.init_sizes = init_sizes
        self.x_overlaps = x_overlaps
        self.y_steps = y_steps
        self.scales = scales

    def loadClassifier(self, filepath=None, classifier_data=None):

        if filepath is not None:
            filepath = os.path.abspath(filepath)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    "File " + filepath + " does not exist.")
            classifier_data = pickle.load(open(filepath, "rb"))
        else:
            classifier_data = classifier_data

        if classifier_data is None:
            raise ValueError("Invalid classifier data supplied.")

        self.classifier = classifier_data["classifier"]
        self.scaler = classifier_data["scaler"]
        self.cv_color_const = classifier_data["cv_color_const"]
        self.channels = classifier_data["channels"]

        self.descriptor = Descriptor(
            hog_features=classifier_data["hog_features"],
            hist_features=classifier_data["hist_features"],
            spatial_features=classifier_data["spatial_features"],
            hog_lib=classifier_data["hog_lib"],
            size=classifier_data["size"],
            hog_bins=classifier_data["hog_bins"],
            pix_per_cell=classifier_data["pix_per_cell"],
            cells_per_block=classifier_data["cells_per_block"],
            block_stride=classifier_data["block_stride"],
            block_norm=classifier_data["block_norm"],
            transform_sqrt=classifier_data["transform_sqrt"],
            signed_gradient=classifier_data["signed_gradient"],
            hist_bins=classifier_data["hist_bins"],
            spatial_size=classifier_data["spatial_size"])

        return self

    def loadClassifiers(self, classifier_datas=None):

        if len(classifier_datas) == 0:
            raise ValueError("Invalid classifier data supplied.")
        for key in classifier_datas:
            self.classifiers[key] = classifier_datas[key]["classifier"]
            self.scalers[key] = classifier_datas[key]["scaler"]
            self.cv_color_const = classifier_datas[key]["cv_color_const"]
            self.channels = classifier_datas[key]["channels"]

            self.descriptors[key] = Descriptor(
                hog_features=classifier_datas[key]["hog_features"],
                hist_features=classifier_datas[key]["hist_features"],
                spatial_features=classifier_datas[key]["spatial_features"],
                hog_lib=classifier_datas[key]["hog_lib"],
                size=classifier_datas[key]["size"],
                hog_bins=classifier_datas[key]["hog_bins"],
                pix_per_cell=classifier_datas[key]["pix_per_cell"],
                cells_per_block=classifier_datas[key]["cells_per_block"],
                block_stride=classifier_datas[key]["block_stride"],
                block_norm=classifier_datas[key]["block_norm"],
                transform_sqrt=classifier_datas[key]["transform_sqrt"],
                signed_gradient=classifier_datas[key]["signed_gradient"],
                hist_bins=classifier_datas[key]["hist_bins"],
                spatial_size=classifier_datas[key]["spatial_size"])

        return self

    def classify(self, image, feature_vector=None):
        if self.cv_color_const > -1:
            image = cv2.cvtColor(image, self.cv_color_const)
            if len(image.shape) > 2:
                image = image[:, :, self.channels]
            else:
                image = image[:, :, np.newaxis]
        if (len(self.classifiers) == 0):

            feature_vectors = [self.descriptor.getFeatureVector(
                image[y_upper:y_lower, x_upper:x_lower, :])
                for (x_upper, y_upper, x_lower, y_lower) in self.windows]

            feature_vectors = self.scaler.transform(feature_vectors)
            predictions = self.classifier.predict(feature_vectors)
            return [self.windows[ind] for ind in np.argwhere(predictions == 1)[:, 0]]
        else:
            output = []
            feature_vectors = {}
            feature_vectors = {
                "far": [],
                "middle": [],
                "right": [],
                "left": []
            }
            # if (feature_vector is None):
            #     feature_vectors = {
            #         "far": [],
            #         "middle": [],
            #         "right": [],
            #         "left": []
            #     }
            # else:
            #     for key in feature_vector:
            #         feature_vectors[key] = []
            for key in self.descriptors:
                if self.descriptors[key] is None:
                    continue
                #     for (x_upper, y_upper, x_lower, y_lower) in self.windows[i]:
                #         valami_image = image[y_upper:y_lower,
                #                              x_upper:x_lower, :]
                #         valami = [
                #             self.descriptors[key].getFeatureVector(valami_image)]
                valami = [self.descriptors[key].getFeatureVector(
                    image[y_upper:y_lower, x_upper:x_lower, :])
                    for (x_upper, y_upper, x_lower, y_lower) in self.windows]
                feature_vectors[key].extend(valami)

                feature_vectors[key] = self.scalers[key].transform(
                    feature_vectors[key])
                predictions = self.classifiers[key].predict(
                    feature_vectors[key])
                output.extend([self.windows[ind]
                               for ind in np.argwhere(predictions == 1)[:, 0]])
            return output

    def detectVideo(self, video_capture=None, num_frames=9, threshold=120,
                    min_bbox=None, show_video=True, draw_heatmap=True,
                    draw_heatmap_size=0.2, write=False, write_fps=24, feature_vector=None):

        cap = video_capture
        if not cap.isOpened():
            raise RuntimeError("Error opening VideoCapture.")
        (grabbed, frame) = cap.read()
        (h, w) = frame.shape[:2]
        if (len(self.classifiers) == 0):
            self.windows = slidingWindow((w, h), init_size=self.init_size, x_overlap=self.x_overlap, y_step=self.y_step,
                                         x_range=self.x_range, y_range=self.y_range, scale=self.scale)
            for (x, y, x_width, y_width) in self.windows:
                cv2.rectangle(frame, (x, y), (x+x_width,
                              y+y_width), (0, 0, 0), 1)
        else:
            self.windows = []
            for key in self.classifiers:
                self.windows.extend(myslidingWindow((w, h), init_size=self.init_sizes[key],
                                                    x_overlap=self.x_overlaps[key], y_step=self.y_steps[key],
                                    x_range=self.x_ranges[key], y_range=self.y_ranges[key], scale=self.scales[key]))
            i = 0
            r = 0
            g = 0
            b = 0
            delay = 15
            for (x, y, x_width, y_width) in self.windows:
                cv2.rectangle(frame, (x, y), (x_width,
                              y_width), (r, g, b), 2)
                r += delay
                if r > 255:
                    g += delay
                    if g > 255:
                        b += delay
                        if b > 255:
                            b = 0
                            r = 0
                            g = 0
        cv2.imshow("My Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        if min_bbox is None:
            min_bbox = (int(0.02 * w), int(0.02 * h))

        inset_size = (int(draw_heatmap_size * w), int(draw_heatmap_size * h))

        if write:
            vidFilename = "Datasets/TrainedData/Videos/" + \
                datetime.now().strftime("%Y%m%d%H%M") + ".avi"
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(vidFilename, fourcc, write_fps, (w, h))

        current_heatmap = np.zeros((frame.shape[:2]), dtype=np.uint8)
        summed_heatmap = np.zeros_like(current_heatmap, dtype=np.uint8)
        last_N_frames = deque(maxlen=num_frames)
        heatmap_labels = np.zeros_like(current_heatmap, dtype=np.uint8)

        weights = np.linspace(1 / (num_frames + 1), 1, num_frames)
        while True:
            (grabbed, frame) = cap.read()
            if not grabbed:
                break

            current_heatmap[:] = 0
            summed_heatmap[:] = 0
            for (x_upper, y_upper, x_lower, y_lower) in self.classify(frame, feature_vector=feature_vector):
                current_heatmap[y_upper:y_lower, x_upper:x_lower] += 10

            last_N_frames.append(current_heatmap)
            for i, heatmap in enumerate(last_N_frames):
                cv2.add(summed_heatmap, (weights[i] * heatmap).astype(np.uint8),
                        dst=summed_heatmap)

            cv2.dilate(summed_heatmap, np.ones((7, 7), dtype=np.uint8),
                       dst=summed_heatmap)

            if draw_heatmap:
                inset = cv2.resize(summed_heatmap, inset_size,
                                   interpolation=cv2.INTER_AREA)
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                frame[:inset_size[1], :inset_size[0], :] = inset

            summed_heatmap[summed_heatmap <= threshold] = 0

            num_objects = label(summed_heatmap, output=heatmap_labels)

            for obj in range(1, num_objects + 1):
                (Y_coords, X_coords) = np.nonzero(heatmap_labels == obj)
                x_upper, y_upper = min(X_coords), min(Y_coords)
                x_lower, y_lower = max(X_coords), max(Y_coords)

                if (x_lower - x_upper > min_bbox[0]
                        and y_lower - y_upper > min_bbox[1]):
                    cv2.rectangle(frame, (x_upper, y_upper), (x_lower, y_lower),
                                  (0, 255, 0), 6)

            if write:
                writer.write(frame)

            if show_video:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

        cap.release()

        if write:
            writer.release()
