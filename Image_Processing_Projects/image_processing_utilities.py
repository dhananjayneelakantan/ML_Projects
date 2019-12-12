import cv2
import numpy as np

def size_checker_and_resize_jpg(image, h, w):
    he, wi = image.shape
    if (he != h or wi != w):
        image = cv2.resize(image, (w, h))
    return image


"""
right cropper function crops the right half of an image and returns it
"""


def right_cropper(image):
    he, wi = image.shape
    w_half = int(wi / 2)
    image_cropped = image[:, w_half:]
    return image_cropped


"""
left cropper function crops the left half of an image and returns it 
"""


def left_cropper(image):
    he, wi = image.shape
    w_half = int(wi / 2)
    image_cropped = image[:, :w_half]
    return image_cropped


"""
detect_right_edge function detects the right edge of an image by using the THRESHOLD_PIXELS_COUNT value that was set.
We need a black to white transition to detect the image
"""


def detect_right_edge(image, THRESHOLD_PIXELS_COUNT):
    image = image
    h, w = image.shape
    maxx = 0
    edge = 0
    margin = int((h / 50))
    for x1 in range(0, int(w / 2)):
        x = w - x1
        vertical_slice = image[0:h, -x:-(x - margin)]
        vertical_slice_pixels_count = vertical_slice.sum()

        if vertical_slice_pixels_count > THRESHOLD_PIXELS_COUNT:
            image = image[0:h, 0:x]
            return image

        if vertical_slice_pixels_count > maxx:
            maxx = vertical_slice_pixels_count
            edge = x
    return image


"""
detect_left_edge function detects the right edge of an image by using the THRESHOLD_PIXELS_COUNT value that was set.
We need a black to white transition to detect the image
"""


def detect_left_edge(image, THRESHOLD_PIXELS_COUNT):
    h, w = image.shape
    maxx = 0
    margin = int((h / 50))
    for x in range(0, int(w / 2)):
        vertical_slice = image[0:h, x:x + margin]
        vertical_slice_pixels_count = vertical_slice.sum()

        if vertical_slice_pixels_count > THRESHOLD_PIXELS_COUNT:
            image = image[0:h, x:w]
            return image

        if vertical_slice_pixels_count > maxx:
            maxx = vertical_slice_pixels_count
    return image

"""
Preprocessing steps to be followed prior to applying Machine Learning or Image Classification.
"""


def thresholding(image):
    thresh = cv2.threshold(image, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def binarise(image):
    max_output_value = 255
    neighborhood_size = 99
    subtract_from_mean = 10
    image_binarized = cv2.adaptiveThreshold(image,
                                            max_output_value,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            neighborhood_size,
                                            subtract_from_mean)
    return image_binarized


def blur(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def opening(image, opening_iteration):
    kernel = np.ones((5, 5), np.uint8)
    opening_or_erosion = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=opening_iteration)
    return opening_or_erosion


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    try:
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    except:
        return None, None
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def feature_extraction(img):
    cell_size = (10, 10)  # h x w in pixels
    block_size = (5, 5)  # h x w in cells
    nbins = 8  # number of orientation bins

    # winSize is the size of the image cropped to a multiple of the cell size
    # cell_size is the size of the cells of the img patch over which to calculate the histograms
    # block_size is the number of cells which fit in the patch
    hogs = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                       img.shape[0] // cell_size[0] * cell_size[0]),
                             _blockSize=(block_size[1] * cell_size[1],
                                         block_size[0] * cell_size[0]),
                             _blockStride=(cell_size[1], cell_size[0]),
                             _cellSize=(cell_size[1], cell_size[0]),
                             _nbins=nbins)
    hist = hogs.compute(img)

    return hist



