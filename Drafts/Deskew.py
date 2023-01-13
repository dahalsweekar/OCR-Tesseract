import cv2
import numpy as np
import pytesseract
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

doc = cv2.imread('Resources/skewed_img.png')

#Step1: Resize
def resize_img(img):
    img = cv2.resize(img,(600,600),interpolation = cv2.INTER_AREA)
    return img

#Step2: Normalization
def normalize_img(img):
    norm = np.zeros_like(img)
    norm_img = cv2.normalize(img,norm,0,255,cv2.NORM_MINMAX)
    return norm_img

#Step3 : Deskew
def Deskew(img):
    #img = rgb2gray(img)
    angle = determine_skew(img)
    h , w = img.shape[:2]
    center = (w//2,h//2)
    #rotated = rotate(img, angle, resize=True)
    print(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImg

#Step4 : Noise Removal
def noise_removal(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

def eroding_img(img):
    kernel = np.ones((1,1),np.uint8)
    img = cv2.erode(img,kernel,iterations=1)
    return img

def thresholding_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.threshold(img,150,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)[1]
    return img

def OCR_detect(img):
    img = np.array(img)
    text = pytesseract.image_to_string(img)
    return text

imgF = resize_img(doc)
#imgF = normalize_img(imgF)
imgF = Deskew(imgF)
#imgF = noise_removal(imgF)
#imgF = eroding_img(imgF)
#imgF = thresholding_img(imgF)
text = OCR_detect(imgF)
text_before = OCR_detect(doc)
print(f'Extracted Text before preprocessing:\n{text_before}')
print(f'Extracted Text after preprocessing:\n{text}')
#cv2.imshow("Original Image",doc)
cv2.imshow("Original Image", doc)
cv2.imshow("Processed Image", imgF)
cv2.waitKey(0)

