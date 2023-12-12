import cv2
import numpy as np
from skimage import feature


class Descriptor:
    class _skHOGDescriptor:
        def __init__(self, hog_bins, pix_per_cell, cells_per_block,
                     block_norm, transform_sqrt):

            self.hog_bins = hog_bins
            self.pix_per_cell = pix_per_cell
            self.cells_per_block = cells_per_block
            self.block_norm = block_norm
            self.transform_sqrt = transform_sqrt

        def compute(self, image):
            multichannel = len(image.shape) > 2
            return feature.hog(image, orientations=self.hog_bins,
                               pixels_per_cell=self.pix_per_cell,
                               cells_per_block=self.cells_per_block,
                               block_norm=self.block_norm, transform_sqrt=self.transform_sqrt,
                               multichannel=multichannel, feature_vector=True)

    def __init__(self, hog_features=False, hist_features=False,
                 spatial_features=False, hog_lib="cv", size=(64, 64),
                 hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_stride=None, block_norm="L1", transform_sqrt=True,
                 signed_gradient=False, hist_bins=16, spatial_size=(16, 16)):

        self.hog_features = hog_features
        self.hist_features = hist_features
        self.spatial_features = spatial_features
        self.size = size
        self.hog_lib = hog_lib
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block

        if hog_lib == "cv":
            winSize = size
            cellSize = pix_per_cell
            blockSize = (cells_per_block[0] * cellSize[0],
                         cells_per_block[1] * cellSize[1])

            if block_stride is not None:
                blockStride = self.block_stride
            else:
                blockStride = (int(blockSize[0] / 2), int(blockSize[1] / 2))

            nbins = hog_bins
            derivAperture = 1
            winSigma = -1.
            # L2Hys (currently the only available option)
            histogramNormType = 0
            L2HysThreshold = 0.2
            gammaCorrection = 1
            nlevels = 64
            signedGradients = signed_gradient

            self.HOGDescriptor = cv2.HOGDescriptor(winSize, blockSize,
                                                   blockStride, cellSize, nbins, derivAperture, winSigma,
                                                   histogramNormType, L2HysThreshold, gammaCorrection,
                                                   nlevels, signedGradients)
        else:
            self.HOGDescriptor = self._skHOGDescriptor(hog_bins, pix_per_cell,
                                                       cells_per_block, block_norm, transform_sqrt)

        self.hist_bins = hist_bins
        self.spatial_size = spatial_size

    def getFeatureVector(self, image):
        if image.shape[:2] != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        feature_vector = np.array([])

        if self.hog_features:
            hogdescriptor = np.expand_dims(
                self.HOGDescriptor.compute(image), 1)

            feature_vector = np.hstack(
                (feature_vector, hogdescriptor[:, 0]))

        if self.hist_features:
            if len(image.shape) < 3:
                image = image[:, :, np.newaxis]

            hist_vector = np.array([])
            for channel in range(image.shape[2]):
                channel_hist = np.histogram(image[:, :, channel],
                                            bins=self.hist_bins, range=(0, 255))[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature_vector = np.hstack((feature_vector, hist_vector))

        if self.spatial_features:
            spatial_image = cv2.resize(image, self.spatial_size,
                                       interpolation=cv2.INTER_AREA)
            spatial_vector = spatial_image.ravel()
            feature_vector = np.hstack((feature_vector, spatial_vector))

        return feature_vector
