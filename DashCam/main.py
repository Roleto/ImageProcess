import cv2
from train import processFiles, trainSVM
from detector import Detector
pos_dires = {
    "far": "../../Kéfeldolgozás/Datasets/Images/far-images",
    "left": "../../Kéfeldolgozás/Datasets/Images/left-images",
    "middle": "../../Kéfeldolgozás/Datasets/Images/middleclose-images",
    "right": "../../Kéfeldolgozás/Datasets/Images/right-images"
}
# "D:\Egyetem\Kéfeldolgozásr\Datasets"
pos_dir = "../../Kéfeldolgozás/Datasets/Images/vehicles"
neg_dir = "../../Kéfeldolgozás/Datasets/Images/non-vehicles"

# video_file = "Datasets/Videos/Cars.mp4"
# video_file = "Datasets/Videos/daschcam.mp4"
# video_file = "Datasets/Videos/nagyobb_daschcam.mp4"
video_file = "../..//Képfeldolgozás/Datasets/Videos/hd_daschcam_cropped.mp4"
# video_file = "Datasets/Videos/hd_daschcam.mp4"
# "D:\Egyetem\Képfeldolgozás\Datasets"


def myExemple():

    # feature_data = processFiles(pos_dir=pos_dir, neg_dir=neg_dir, recurse=True,
    #                             color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
    #                             hist_features=True, spatial_features=True, hog_lib="cv",
    #                             size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
    #                             hog_bins=20, hist_bins=16, spatial_size=(20, 20))

    # classifier_data = trainSVM(feature_data=feature_data, C=1000)

    far_pos_dir = pos_dires["far"]+"/vehicles"
    middle_pos_dir = pos_dires["middle"]+"/vehicles"
    far_neg_dir = pos_dires["far"]+"/non-vehicles"
    middle_neg_dir = pos_dires["middle"]+"/non-vehicles"
    feature_far_data_filename = "Datasets/TrainedData/myfarfile.pkl"
    feature_middle_data_filename = "Datasets/TrainedData/mymiddlefile.pkl"

    far_feature_data = processFiles(far_pos_dir, far_neg_dir, recurse=True,
                                    color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                    hist_features=True, spatial_features=True, hog_lib="cv",
                                    size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                    hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                    output_file=True, output_filename=feature_far_data_filename)
    middle_feature_data = processFiles(middle_pos_dir, middle_neg_dir, recurse=True,
                                       color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                       hist_features=True, spatial_features=True, hog_lib="cv",
                                       size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                       hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                       output_file=True, output_filename=feature_middle_data_filename)

    classifier_far_data_filename = "Datasets/TrainedData/classifier_far_data.pkl"
    classifier_middle_data_filename = "Datasets/TrainedData/classifier_middle_data.pkl"

    far_classifier_data = trainSVM(feature_data=far_feature_data, output_file=True,
                                   output_filename=classifier_far_data_filename)
    middle_classifier_data = trainSVM(feature_data=middle_feature_data, output_file=True,
                                      output_filename=classifier_middle_data_filename, C=1000)
    x_ranges = {
        "far": (0.15, 0.5),
        "middle": (0.3, 5,),
        "right": (0.4, 0.85)
        # még egy jobb oldalt azt ha az se jó akkor megértetem a heatmappet
    }
    y_ranges = {
        "far": (0.2, 0.8),
        "middle": (0.46, 0.8),
        "right": (0.81, 0.9)
    }
    detector = Detector(init_size=(16, 16), x_overlap=0.75, y_step=0.1,
                        x_range=x_ranges, y_range=y_ranges, scale=1.75)
    classifier_datas = {
        "far": far_classifier_data,
        "middle": middle_classifier_data,
        "right": middle_classifier_data  # NEED TO CHANGE!!!!!!!!!!!!!!!!!!!!!!!
    }
    detector.loadClassifiers(classifier_datas=classifier_datas)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
                         draw_heatmap_size=0.3)


def drawRect(x, y, x_width, y_width, color):
    video_capture = cv2.VideoCapture(video_file)
    cap = video_capture
    if not cap.isOpened():
        raise RuntimeError("Error opening VideoCapture.")
    (grabbed, frame) = cap.read()
    (h, w) = frame.shape[:2]
    print((h, w))
    if (type(x) is int):
        cv2.rectangle(frame, (x, y),
                      (x+x_width, y+y_width), color, 2)
    else:
        for i in range(len(x)):
            cv2.rectangle(frame, (x[i], y[i]),
                          (x[i]+x_width[i], y[i]+y_width[i]), color[i], 2)
    cv2.imshow("My Video", frame)
    cv2.waitKey(0)


def FarMidleMethod():

    far_pos_dir = pos_dires["far"]+"/vehicles"
    middle_pos_dir = pos_dires["middle"]+"/vehicles"
    far_neg_dir = pos_dires["far"]+"/non-vehicles"
    middle_neg_dir = pos_dires["middle"]+"/non-vehicles"
    feature_far_data_filename = "Datasets/TrainedData/myfarfile.pkl"
    feature_middle_data_filename = "Datasets/TrainedData/mymiddlefile.pkl"

    far_feature_data = processFiles(far_pos_dir, far_neg_dir, recurse=True,
                                    color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                    hist_features=True, spatial_features=True, hog_lib="cv",
                                    size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                    hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                    output_file=True, output_filename=feature_far_data_filename)
    middle_feature_data = processFiles(middle_pos_dir, middle_neg_dir, recurse=True,
                                       color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                       hist_features=True, spatial_features=True, hog_lib="cv",
                                       size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                       hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                       output_file=True, output_filename=feature_middle_data_filename)

    classifier_far_data_filename = "Datasets/TrainedData/classifier_far_data.pkl"
    classifier_middle_data_filename = "Datasets/TrainedData/classifier_middle_data.pkl"

    far_classifier_data = trainSVM(feature_data=far_feature_data, output_file=True,
                                   output_filename=classifier_far_data_filename)
    middle_classifier_data = trainSVM(feature_data=middle_feature_data, output_file=True,
                                      output_filename=classifier_middle_data_filename, C=1000)
    x_ranges = {
        "far": (0.15, 0.25),
        "middle": (0.36, 1)
    }
    y_ranges = {
        "far": (0.1, 0.4),
        "middle": (0.41, 0.9)
    }
    detector = Detector(init_size=(16, 16), x_overlap=0.75, y_step=0.1,
                        x_range=x_ranges, y_range=y_ranges, scale=1.75)
    classifier_datas = {
        "far": far_classifier_data,
        "middle": middle_classifier_data
    }
    detector.loadClassifiers(classifier_datas=classifier_datas)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
                         draw_heatmap_size=0.3)


def OnlyLeft():

    left_pos_dir = pos_dires["left"]+"/vehicles"
    left_neg_dir = pos_dires["left"]+"/non-vehicles"
    feature_left_data_filename = "Datasets/TrainedData/myleftfile.pkl"
    classifier_left_data_filename = "Datasets/TrainedData/classifier_left_data.pkl"

    classifier_datas = {}

    left_feature_data = processFiles(left_pos_dir, left_neg_dir, recurse=False,
                                     color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                     hist_features=True, spatial_features=True, hog_lib="cv",
                                     size=(32, 32), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                     hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                     output_file=True, output_filename=feature_left_data_filename)

    left_classifier_data = trainSVM(
        feature_data=left_feature_data, output_file=True, output_filename=classifier_left_data_filename)
    classifier_datas["left"] = left_classifier_data

    x_ranges = {  # width
        "far": (.3, .54),
        "middle": (.0, .32),
        "right": (.6, .9),
        "left": (0, .32)
    }
    y_ranges = {  # height
        "far": (0.45, 0.55),
        "middle": (.7, .8),
        "right": (.45, .8),
        "left": (.42, .7)
    }
    init_sizes = {
        "far": (32, 32),
        "middle": (64, 64),
        "right": (64, 64),
        "left": (64, 64)
    }
    x_overlaps = {  # 360 width
        "far": .7,
        "middle": .3,
        "right": .5,
        "left": .2
    }
    y_steps = {  # 640 height
        "far": .3,
        "middle": .5,
        "right": .25,
        "left": .5
    }
    scales = {
        "far": 1.5,
        "middle": 1.7,
        "right": 1,
        "left": 1.7
    }
    detector = Detector(init_sizes=init_sizes, x_overlaps=x_overlaps, y_steps=y_steps,
                        x_ranges=x_ranges, y_ranges=y_ranges, scales=scales)
    detector.loadClassifiers(classifier_datas=classifier_datas)
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
                         draw_heatmap=True, draw_heatmap_size=0.3, write=True)


def DashCam(keys=["All"], train=True, heatmap=True):

    far_pos_dir = pos_dires["far"]+"/vehicles"
    middle_pos_dir = pos_dires["middle"]+"/vehicles"
    right_pos_dir = pos_dires["right"]+"/vehicles"
    left_pos_dir = pos_dires["left"]+"/vehicles"
    far_neg_dir = pos_dires["far"]+"/non-vehicles"
    middle_neg_dir = pos_dires["middle"]+"/non-vehicles"
    right_neg_dir = pos_dires["right"]+"/non-vehicles"
    left_neg_dir = pos_dires["left"]+"/non-vehicles"
    feature_far_data_filename = "Datasets/TrainedData/myfarfile.pkl"
    classifier_far_data_filename = "Datasets/TrainedData/classifier_far_data.pkl"
    feature_middle_data_filename = "Datasets/TrainedData/mymiddlefile.pkl"
    classifier_middle_data_filename = "Datasets/TrainedData/classifier_middle_data.pkl"
    feature_right_data_filename = "Datasets/TrainedData/myrightfile.pkl"
    classifier_right_data_filename = "Datasets/TrainedData/classifier_right_data.pkl"
    feature_left_data_filename = "Datasets/TrainedData/myleftfile.pkl"
    classifier_left_data_filename = "Datasets/TrainedData/classifier_left_data.pkl"

    classifier_datas = {}

    if keys[0] == "All" or "far" in keys:
        if (train):
            far_feature_data = processFiles(far_pos_dir, far_neg_dir, recurse=True,
                                            color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                            hist_features=True, spatial_features=True, hog_lib="cv",
                                            size=(16, 16), pix_per_cell=(2, 2), cells_per_block=(2, 2),
                                            hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                            output_file=True, output_filename=feature_far_data_filename)

            far_classifier_data = trainSVM(
                feature_data=far_feature_data, output_file=True, output_filename=classifier_far_data_filename)
        else:
            far_classifier_data = trainSVM(
                filepath=classifier_far_data_filename, output_file=True, output_filename=classifier_far_data_filename)
        classifier_datas["far"] = far_classifier_data

    if keys[0] == "All" or "right" in keys:
        if (train):
            right_feature_data = processFiles(right_pos_dir, right_neg_dir, recurse=True,
                                              color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                              hist_features=True, spatial_features=True, hog_lib="cv",
                                              size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                              hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                              output_file=True, output_filename=feature_right_data_filename)

            right_classifier_data = trainSVM(
                feature_data=right_feature_data, output_file=True, output_filename=classifier_right_data_filename)
        else:
            right_classifier_data = trainSVM(
                filepath=classifier_right_data_filename, output_file=True, output_filename=classifier_right_data_filename)
        classifier_datas["right"] = right_classifier_data

    if keys[0] == "All" or "left" in keys:
        if (train):
            left_feature_data = processFiles(left_pos_dir, left_neg_dir, recurse=False,
                                             color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                             hist_features=True, spatial_features=True, hog_lib="cv",
                                             size=(64, 64), pix_per_cell=(16, 16), cells_per_block=(4, 4),
                                             hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                             output_file=True, output_filename=feature_left_data_filename)

            left_classifier_data = trainSVM(
                feature_data=left_feature_data, output_file=True, output_filename=classifier_left_data_filename)
        else:
            left_classifier_data = trainSVM(
                filepath=classifier_left_data_filename, output_file=True, output_filename=classifier_left_data_filename)
        classifier_datas["left"] = left_classifier_data

    if keys[0] == "All" or "middle" in keys:
        if (train):
            middle_feature_data = processFiles(middle_neg_dir, middle_pos_dir, recurse=True,
                                               color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                               hist_features=True, spatial_features=True, hog_lib="cv",
                                               size=(64, 64), pix_per_cell=(16, 16), cells_per_block=(4, 4),
                                               hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                               output_file=True, output_filename=feature_left_data_filename)
            #    output_file=True, output_filename=feature_middle_data_filename)

            middle_classifier_data = trainSVM(feature_data=middle_feature_data, output_file=True,
                                              output_filename=classifier_middle_data_filename)
        else:
            middle_classifier_data = trainSVM(
                filepath=classifier_middle_data_filename, output_file=True, output_filename=classifier_middle_data_filename)
        classifier_datas["middle"] = middle_classifier_data

    x_ranges = {  # width
        "far": (.3, .54),
        "middle": (.0, .32),
        "right": (.6, .9),
        "left": (0, .32)
    }
    y_ranges = {  # height
        "far": (0.45, 0.55),
        "middle": (.7, .8),
        "right": (.45, .8),
        "left": (.42, .65)
    }
    init_sizes = {
        "far": (32, 32),
        "middle": (64, 64),
        "right": (64, 64),
        "left": (64, 64)
    }
    x_overlaps = {  # 360 width
        "far": .7,
        "middle": .3,
        "right": .5,
        "left": .4
    }
    y_steps = {  # 640 height
        "far": .3,
        "middle": .5,
        "right": .25,
        "left": .5
    }
    scales = {
        "far": 1.5,
        "middle": 1.7,
        "right": 1,
        "left": 1.6
    }
    detector = Detector(init_sizes=init_sizes, x_overlaps=x_overlaps, y_steps=y_steps,
                        x_ranges=x_ranges, y_ranges=y_ranges, scales=scales)
    detector.loadClassifiers(classifier_datas=classifier_datas)
    cap = cv2.VideoCapture(video_file)
    if (keys[0] == "All"):
        detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
                             draw_heatmap=True, draw_heatmap_size=0.3, write=True)
    else:
        feature_vector = {}
        for key in keys:
            feature_vector["{}".format(key)] = []
        detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
                             draw_heatmap=heatmap, draw_heatmap_size=0.3, write=True, feature_vector=feature_vector)


def trainedSVM():
    pos_dir = pos_dires["far"]+"/vehicles"
    neg_dir = pos_dires["far"]+"/non-vehicles"
    feature_data_filename = "Datasets/TrainedData/myfarfile.pkl"

    far_feature_data = processFiles(pos_dir, neg_dir, recurse=True,
                                    color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                    hist_features=True, spatial_features=True, hog_lib="cv",
                                    size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                    hog_bins=20, hist_bins=16, spatial_size=(20, 20),
                                    output_file=True, output_filename=feature_data_filename)

    far_classifier_data = trainSVM(feature_data=far_feature_data, C=1000)
    classifier_datas = {
        "far": far_classifier_data
    }

    # far_detector = Detector(init_size=(64, 64), x_overlap=0.7, y_step=0.01,
    #                         x_range=(0.15, 0.5), y_range=(0.1, 5), scale=1.5)
    far_detector = Detector(init_size=(64, 64), x_overlap=0.7, y_step=0.01,
                            x_range=(0.15, 0.4), y_range=(0.1, 0.5), scale=1.5)
    far_detector.loadClassifier(classifier_data=far_classifier_data)

    cap = cv2.VideoCapture(video_file)
    far_detector.detectVideo(video_capture=cap, num_frames=9, threshold=120, draw_heatmap=False,
                             draw_heatmap_size=0.3, write=True)


if __name__ == "__main__":
    # example1()
    # example2()
    # example3()
    # example4()
    # example5()
    # myExemple()
    # OnlyLeft()
    DashCam(["left", "far"])
    # drawRect(x,y,x_w,y_w)
    # drawRect(0, 150, 200, 510, (0, 0, 0))  # left y = 0,0.3125 x = 0.235,0.5625
    # drawRect(200, 200, 160, 460, (0, 0, 0)) # middle y = 0.3125,.5625 x = 0.56,1
    # drawRect(400, 150, 260, 260, (0, 0, 0))  # right y =0.625,1 x = .5626,1
    # drawRect(150, 150, 200, 50, (0, 0, 0))  # far y = 0.3125,.5625 x = 0.4,0.55
    # drawRect([0, 400, 200, 200], [150, 150, 200, 150], [200, 240, 200, 200], [400, 400, 400, 50], [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 0, 0)])
    # FarMidleMethod()
    # trainedExemple()
