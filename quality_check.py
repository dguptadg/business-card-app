import cv2

#function to find the blur score (convert to gray-scale -> apply laplacian filter -> find the variance)
def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return blur_score

#to check if the image is blurr or not on the basis of blur score found using : detect_blur function 
def is_blurry(image):
    blur_score = detect_blur(image)
    return blur_score < 90

#to find the brightness of the image (used the mean of all pixel values)
def get_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    return brightness

#check the brightness of the image using the value obtained from : get_brightness() function 
def check_brightness(image):
    brightness = get_brightness(image)

    if brightness < 55:
        return "dark"
    elif brightness > 215:
        return "bright"
    else:
        return "good"

#find the glare score of the image using this function (see the ration of saturated pixels to total pixels of the image) only the 80% of the image is considered as the total number of pixels, if ratio is more then the glare is also more 
def get_glare_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    x1 = int(w * 0.2)
    x2 = int(w * 0.8)

    y1 = int(h * 0.2)
    y2 = int(h * 0.8)

    center_region = gray[y1:y2, x1:x2]

    saturated_pixels = (center_region > 245).sum()

    total_pixels = (
        center_region.shape[0] * center_region.shape[1]
    )

    glare_percentage = (saturated_pixels / total_pixels) * 100

    return glare_percentage

#use this function to classify if the image has glare or not on the basis of selected threshold and the glare % value obtained from the function : get_glare_percentage(image)
GLARE_PERCENT_THRESHOLD = 4

def has_glare(image):
    glare_percentage = get_glare_percentage(image)

    return glare_percentage > GLARE_PERCENT_THRESHOLD


#check the resolution of the image using the shape : 
def check_resolution(image):
    height, width = image.shape[:2]

    return width >= 640 and height >= 480


#add the validate function in order to find the reason for rejection of image on the basis of above implemented functions 
def validate_quality(image):
    if not check_resolution(image):
        return {"valid": False, "reason": "low_resolution"}

    if is_blurry(image):
        return {"valid": False, "reason": "blur"}

    brightness_status = check_brightness(image)

    if brightness_status == "dark":
        return {"valid": False, "reason": "dark"}

    if brightness_status == "bright":
        return {"valid": False, "reason": "bright"}

    if has_glare(image):
        return {"valid": False, "reason": "glare"}

    return {"valid": True, "reason": None}


