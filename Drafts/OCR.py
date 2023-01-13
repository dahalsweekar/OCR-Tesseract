import cv2
import numpy as np
import pytesseract

document = cv2.imread('Resources/text_cam.jpg')
document = cv2.resize(document,(600,600),interpolation = cv2.INTER_AREA)

norm = np.zeros_like(document)
norm_doc = cv2.normalize(document,norm,0,255,cv2.NORM_MINMAX)

def getskewangle(img):
    imgC = img.copy()
    imgC1 = img.copy()
    imgG = cv2.cvtColor(imgC,cv2.COLOR_BGR2GRAY)
    imgB = cv2.GaussianBlur(imgG,(9,9),0)
    imgT = cv2.threshold(imgB,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold image",imgT)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,5))
    imgD = cv2.dilate(imgT,kernel,iterations=5)
    cv2.imshow("Dilated image",imgD)
    contours, heirarchy = cv2.findContours(imgD,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours,key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    imgC1 = cv2.drawContours(imgC1,largestContour,-1,(0,255,0),2)
    cv2.imshow("Largest Contour",imgC1)
    rect = cv2.minAreaRect(largestContour)
    angle = rect[-1]
    if angle < -45:
        angle = -(90+angle)
    else:
        angle = -angle
    return -1.0 * angle

def Rotation(img,angle:float):
    newImg = img.copy()
    (h, w) = newImg.shape[:2]
    center = (w//2, h//2)
    print(center)
    M = cv2.getRotationMatrix2D(center, angle-90, 1.0)
    print(angle)
    newImg = cv2.warpAffine(newImg, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow("Deskewed image",newImg)
    return newImg

def deskew(img):
    angle = getskewangle(img)
    return Rotation(img,-1.0*angle)

def image_to_text(img):
    document = np.array(img)
    text = pytesseract.image_to_string(document)
    return text

final_image = deskew(document)
cv2.imshow("Raw Document",document)
text = image_to_text(document)
print(text)
#cv2.imshow("Normalized document",final_image)
cv2.waitKey(0)