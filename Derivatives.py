from Helper import *


def derive_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0).astype("uint8")
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1).astype("uint8")
    both_1 = cv2.bitwise_and(sobel_x, sobel_y)
    ret, threshold = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    show_multi_images(i1=both_1, t=threshold)

def canny_edge_detection(im):
    blurr = cv2.GaussianBlur(im, (3,3), cv2.BORDER_DEFAULT)
    return cv2.Canny(blurr, 125, 175)
