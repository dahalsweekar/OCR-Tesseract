import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from deskew import determine_skew
from skimage.transform import rotate

print(pytesseract.get_tesseract_version())

def normalize_image(img):
    norm_img = cv2.normalize(img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def resize_img(img):
    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
    return img


def deskew(img):
    angle = determine_skew(img)
    newImg = rotate(img,angle,resize=True) * 255
    #h, w = img.shape[:2]
    #center = (w // 2, h // 2)
    #M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #newImg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImg.astype(np.uint8)


def noise_removal(img):
    return cv2.medianBlur(img, 1)


def threshold_img(img):
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


def ocr_tessaract(img):
    return pytesseract.image_to_string(img)


def draw_bbox(bbox, img):
    pass


def detected_text(img, original):
    dicti = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(dicti['text'])
    for i in range(n_boxes):
        if int(dicti['conf'][i]) > 60:
            (x, y, w, h) = (dicti['left'][i], dicti['top'][i], dicti['width'][i], dicti['height'][i])
            original = cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original


def image_preprocess(img):
    #resized_image = resize_img(img)
    dskewed_image = deskew(img)
    noiseless_image = noise_removal(dskewed_image)
    thresh_image = threshold_img(noiseless_image)
    return thresh_image


def preprocess_resize_dskew(img):
    #img = resize_img(img)
    img = deskew(img)
    return img


if __name__ == '__main__':
    img = cv2.imread('Resources/High_res/ext_test.png')
    processed_image = image_preprocess(img)
    norm_img = normalize_image(img)
    text = ocr_tessaract(processed_image)
    resize_dskew_image = preprocess_resize_dskew(img)
    boxed_image = detected_text(processed_image, resize_dskew_image)
    print(f'Extracted Text:\n {text}')
    cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Normalized Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Image", boxed_image)
    cv2.imshow("Threshold Image", threshold_img(img))
    cv2.imshow("Threshold Image", threshold_img(img))
    cv2.imshow("Normalized Image", norm_img)
    cv2.waitKey(0)
