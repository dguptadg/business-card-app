import numpy as np
from quality_check import check_resolution

import pytest
from quality_check import is_blurry, check_brightness

from quality_check import has_glare

import quality_check as qc

#to test the resolution : 
def test_valid_resolution():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = check_resolution(image)

    assert result == True


#to cehck for the invalid width : 
def test_invalid_width():
    image = np.zeros((480, 639, 3), dtype=np.uint8)

    assert check_resolution(image) is False

#to check for invalid heights : 
def test_invalid_height():
    image = np.zeros((479, 640, 3), dtype=np.uint8)

    assert check_resolution(image) is False


#Test blur and brightness validation : 
def test_blurry_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert is_blurry(image)


def test_sharp_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255

    assert not is_blurry(image)


@pytest.mark.parametrize("value, expected", [
    (0, "dark"),
    (55, "good"),
    (215, "good"),
    (255, "bright")
])
def test_brightness(value, expected):
    image = np.full((100, 100, 3), value, dtype=np.uint8)

    assert check_brightness(image) == expected


#Test Glare Detection Using Pytest
def test_no_glare():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert not has_glare(image)


def test_glare_detected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    image[30:70, 30:70] = 255

    assert has_glare(image)


#Test the Complete validate_quality() Function
@pytest.mark.parametrize("resolution, blurry, brightness, glare, reason", [
    (False, False, "good", False, "low_resolution"),
    (True, True, "good", False, "blur"),
    (True, False, "dark", False, "dark"),
    (True, False, "bright", False, "bright"),
    (True, False, "good", True, "glare"),
    (True, False, "good", False, None)
])
def test_validate_quality(
    monkeypatch, resolution, blurry, brightness, glare, reason
):
    monkeypatch.setattr(qc, "check_resolution", lambda image: resolution)
    monkeypatch.setattr(qc, "is_blurry", lambda image: blurry)
    monkeypatch.setattr(qc, "check_brightness", lambda image: brightness)
    monkeypatch.setattr(qc, "has_glare", lambda image: glare)

    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = qc.validate_quality(image)

    assert result == {
        "valid": reason is None,
        "reason": reason
    }
