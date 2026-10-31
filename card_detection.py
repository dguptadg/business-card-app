import cv2

#this will be used for the corner ordering : 
import numpy as np

#to resize the image down if any of it's dimension is > than the selected max_dimension, if the any of the dimension is greater than the max_dimension then it is scaled down w.r.t max_dimension
def resize_image(image, max_dimension=2048):
    height, width = image.shape[:2]

    largest_dimension = max(height, width)

    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


#convert the image from color to gray scale : 
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply the gaussian blur with the help of gaussian kernel in order to get rid of any kind of noise present in the image input by user so that the image can become ready for the next step which will be canny edge detection, it is preffered to use the gaussian blur before the canny edge so that the image is free from noise and fake edge detection can be avoided (gaussian gives more weightage to the pixels near the center than to those which are far due to the formual of gaussian function)
def apply_gaussian_blur(gray_image):
    return cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )


#implement the canny edge function in order to detect the edges (creates sobel x & sobel y and then merges into 1 matrix using the under root of square sum formula and then use the direction by tan inverse to make the candidate edges as strong edges and keep the weak edges as weak on the basis of slected thresholds)
def detect_edges(blurred_image):
    return cv2.Canny(
        blurred_image,
        40,
        120
    )


#dilate the edges using : Morphological dialation which will make the broken edges strong 
def dilate_edges(edge_image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.dilate(
        edge_image,
        kernel,
        iterations=1
    )


#function for contour detection : 
def find_contours(edge_image):
    contours, hierarchy = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


#keep the contours with area from 15%-90% of the total area in order to keep only the relevant contours : 
def filter_contours_by_area(contours, image):
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    filtered_contours = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if (
            contour_area >= 0.15 * image_area
            and contour_area <= 0.90 * image_area
        ):
            filtered_contours.append(contour)

    return filtered_contours


#Quardilateral detection for the filtered countours : 
def find_quadrilateral_contours(contours):
    quadrilaterals = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            quadrilaterals.append(approx)

    return quadrilaterals


#find the corner points of the quardilateral detected via quardilateral detection amongst the valid countours  :
def order_points(points):
    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype="float32")

    # Top-left and bottom-right
    sums = points.sum(axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    # Top-right and bottom-left
    diffs = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


#now to handle the tilted card images -> prespective wrap technique 
def warp_perspective(image, ordered_points):
    width = 1050
    height = 600

    destination_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype="float32"
    )

    transformation_matrix = cv2.getPerspectiveTransform(
        ordered_points,
        destination_points
    )

    warped = cv2.warpPerspective(
        image,
        transformation_matrix,
        (width, height)
    )

    return warped


#now define the CLAHE function : 
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced