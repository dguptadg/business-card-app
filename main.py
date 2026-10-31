import cv2

#import the functions of quality_check.py file :
from quality_check import detect_blur
from quality_check import is_blurry
from quality_check import get_brightness
from quality_check import check_brightness
from quality_check import get_glare_percentage
from quality_check import has_glare
from quality_check import check_resolution
from quality_check import validate_quality

#import the functions of card_detection.py file :
from card_detection import resize_image
from card_detection import convert_to_grayscale, apply_gaussian_blur
from card_detection import detect_edges
from card_detection import dilate_edges
from card_detection import find_contours
from card_detection import filter_contours_by_area
from card_detection import find_quadrilateral_contours
from card_detection import order_points
from card_detection import warp_perspective
from card_detection import apply_clahe

def main():
    image = cv2.imread("images/input/card1.jpeg")

    if image is None:
        print("Image not found")
        return

    height, width = image.shape[:2]

    print("Image loaded successfully")
    print(f"Width: {width}")
    print(f"Height: {height}")

    #blur score is found using the function : detec_blur imported from : quality_check.py 
    blur_score = detect_blur(image)
    print(f"Blur Score: {blur_score}")

    #to check if the image is blur on the basis of blur score found above on the basis of selected threshold from the function : is_blurry 
    if is_blurry(image):
        print("Image is blurry")
    else:
        print("Image is sharp")

    #get/print the brightness of the image found using function : get_brightness() imported from the quality_check.py 
    brightness = get_brightness(image)
    print(f"Brightness Score: {brightness}")

    #classify the brightness of the image on the basis of brightness value obtained from the function : get_brightness()
    brightness_status = check_brightness(image)
    print(f"Brightness Status: {brightness_status}")

    #find the glare value of the image using the function : get_glare_percentage which is imported from the quality_check.py
    glare = get_glare_percentage(image)
    print(f"Glare Percentage: {glare:.2f}%")

    #classify if the image has too much of glare with the help of this function based on the glare percentage obtained from get_glare_percentage() above and the threshold decided in the quality_check.py
    if has_glare(image):
        print("Glare Detected")
    else:
        print("No Glare")

    #find the resolution of the image using this function defined in quality_check.py : check if the resolution is valid or not with the help of size found using : check_resolution(image) 
    print("Resolution Valid:", check_resolution(image))

    #to print the validated image results by the help of function in qualtiy_check.py
    result = validate_quality(image)
    print("Quality Validation:", result)

    #resize the image using the function from card_detection.py file : 
    resized = resize_image(image)
    #print the original & resized shape : 
    print("Original Shape:", image.shape)
    print("Resized Shape:", resized.shape)

    #convert the image to gray scale and apply the gaussian blur in order to prepare the image for canny edge detection
    gray = convert_to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)

    print("Grayscale Shape:", gray.shape)
    print("Blurred Shape:", blurred.shape)

    #implement the canny edge from the card_detection.py file 
    edges = detect_edges(blurred)
    print("Edges Shape:", edges.shape)
    cv2.imshow("Detected Edges", edges)

    #dialate the edges using morphological dialation : 
    dilated_edges = dilate_edges(edges)
    cv2.imshow("Dilated Edges", dilated_edges)


    #contour detection to find the outermost contours for the dilated image 
    contours = find_contours(dilated_edges)
    print("Total Contours Detected:", len(contours))

    #filter the contours by area to keep the relevant contours : 
    filtered_contours = filter_contours_by_area(contours, resized)
    print("Filtered Contours:", len(filtered_contours))

    #find the countours which form a quardilateral : 
    quadrilaterals = find_quadrilateral_contours(filtered_contours)
    print("Quadrilaterals Detected:", len(quadrilaterals))

    #if the quardilateral is detected -> print it's corner points : 
    if quadrilaterals:
        ordered_points = order_points(quadrilaterals[0])
        print("Ordered Corners:", ordered_points)

        #now implement the prespective transform from the card_detection.py to handle the tilted card images : 
        warped = warp_perspective(resized, ordered_points)
        print("Warped Image Shape:", warped.shape)
        cv2.imshow("Warped Business Card", warped)

        #now implement the CLAHE algorithm to the prespective-transformed card : 
        enhanced = apply_clahe(warped)
        print("Enhanced Image Shape:", enhanced.shape)
        cv2.imshow("CLAHE Enhanced Card", enhanced)
        
    else:
        print("No quadrilateral detected")

    #keep the image windown open indefinitely until user exits that window : 
    cv2.waitKey(0)
    #destroy all the windows after the specific milli-second defined in the waitkey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()