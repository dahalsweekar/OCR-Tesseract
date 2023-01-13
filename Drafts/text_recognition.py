import cv2
import numpy as np
import pytesseract

image = cv2.imread('Resources/text_cam.jpg')
img = cv2.resize(image,(600,600),interpolation = cv2.INTER_AREA)

#Step1: Normalization
norm = np.zeros_like(img)
norm_img = cv2.normalize(img,norm,0,255,cv2.NORM_MINMAX)

#Step2: Deskew
def Deskew(image):
    pass

#Step3: Preprocess
def processImg(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(1,1),0)
    img = cv2.bitwise_not(img)
    img = cv2.threshold(img,150,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)[1]
    kernel = np.ones((1,1),np.uint8)
    img = cv2.erode(img,kernel,iterations=5)
    return img

#Step4: OCR using Tessaract
def OCR_detect(img):
    #img = np.array(img)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

imgP = processImg(img)
text = OCR_detect(imgP)
blank_space = np.zeros_like(img)
blank_space = cv2.putText(blank_space,text,(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
cv2.imshow("Original Image",img)
cv2.imshow("Threshold",imgP)
cv2.imshow("Extracted Text",blank_space)
Deskew(img)
print(text)
cv2.waitKey(0)