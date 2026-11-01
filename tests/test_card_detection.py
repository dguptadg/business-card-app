import numpy as np
from card_detection import resize_image

from card_detection import convert_to_grayscale, apply_gaussian_blur

from card_detection import detect_edges, dilate_edges

from card_detection import find_contours, filter_contours_by_area

from card_detection import find_quadrilateral_contours, order_points

from card_detection import warp_perspective, apply_clahe


#Test Image Resizing Using Pytest
def test_large_image_resize():
    image = np.zeros((3000, 1500, 3), dtype=np.uint8)

    resized = resize_image(image)

    assert resized.shape == (2048, 1024, 3)


def test_small_image_unchanged():
    image = np.zeros((600, 400, 3), dtype=np.uint8)

    resized = resize_image(image)

    assert resized.shape == image.shape


#est Grayscale Conversion and Gaussian Blur
def test_grayscale_conversion():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255

    gray = convert_to_grayscale(image)

    assert gray.shape == (100, 100)
    assert gray[0, 0] == 0
    assert gray[0, 99] == 255


def test_gaussian_blur():
    image = np.zeros((11, 11), dtype=np.uint8)
    image[5, 5] = 255

    blurred = apply_gaussian_blur(image)

    assert blurred.shape == image.shape
    assert 0 < blurred[5, 5] < 255
    assert blurred[5, 4] > 0


#Test Canny Edge Detection and Dilation :
def test_canny_edge_detection():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255

    edges = detect_edges(image)

    assert edges.shape == image.shape
    assert np.count_nonzero(edges) > 0


def test_edge_dilation():
    edges = np.zeros((100, 100), dtype=np.uint8)
    edges[50, 40] = 255
    edges[50, 42] = 255

    dilated = dilate_edges(edges)

    assert dilated.shape == edges.shape
    assert dilated[50, 41] == 255
    assert np.count_nonzero(dilated) > np.count_nonzero(edges)


#Test contour detection and area filtering :
def test_find_contours():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255

    contours = find_contours(image)

    assert len(contours) == 1


def test_filter_contours_by_area():
    image = np.zeros((100, 100), dtype=np.uint8)

    # Large contour (acceptable area)
    image[20:80, 20:80] = 255

    # Small contour (should be rejected)
    image[5:10, 5:10] = 255

    contours = find_contours(image)
    filtered = filter_contours_by_area(contours, image)

    assert len(contours) == 2
    assert len(filtered) == 1


#Test quadrilateral detection and corner ordering :
def test_quadrilateral_detection():
    rectangle = np.array([
        [[10, 10]],
        [[90, 10]],
        [[90, 70]],
        [[10, 70]]
    ], dtype=np.int32)

    triangle = np.array([
        [[10, 10]],
        [[90, 10]],
        [[50, 80]]
    ], dtype=np.int32)

    quadrilaterals = find_quadrilateral_contours(
        [rectangle, triangle]
    )

    assert len(quadrilaterals) == 1
    assert len(quadrilaterals[0]) == 4


def test_corner_ordering():
    points = np.array([
        [[85, 70]],  # Bottom-right
        [[15, 80]],  # Bottom-left
        [[10, 10]],  # Top-left
        [[90, 15]]   # Top-right
    ], dtype=np.int32)

    ordered = order_points(points)

    expected = np.array([
        [10, 10],
        [90, 15],
        [85, 70],
        [15, 80]
    ], dtype=np.float32)

    np.testing.assert_array_equal(ordered, expected)


#Test perspective transformation and CLAHE enhancement :
def test_perspective_transformation():
    image = np.full((100, 200, 3), 150, dtype=np.uint8)

    ordered_points = np.array([
        [0, 0],
        [199, 0],
        [199, 99],
        [0, 99]
    ], dtype=np.float32)

    warped = warp_perspective(image, ordered_points)

    assert warped.shape == (600, 1050, 3)
    assert np.all(warped[300, 525] == 150)


def test_clahe_enhancement():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    image[:, :50] = 80
    image[:, 50:] = 180

    enhanced = apply_clahe(image)

    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8
    assert enhanced[50, 25] < enhanced[50, 75]