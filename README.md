# OCR-Tesseract

OPTICAL CHARACTER RECOGNITION
PYTHON-TESSARACT

INTRODUCTION

Optical Character Recognition (OCR) is the method of identifying the characters in the image. OCR is a mechanism through which a machine can interpret a character in an image. The purpose of the OCR is to significantly increase the performance of work done. Over the years, OCR is developed and managed by groups which has led to increased performance and accuracy of text detection.

Types of OCR:

1. Oculus OCR
2. Swift OCR
3. Tessaract

Tessaract OCR

Tessaract is a character recognition engine that is available under Apache 2.0 license. It is an open-source OCR that is developed by Hp between 1984 and 1994. In 2005, it was made open source. It is managed by google and has resulted in a significant boost in performance. The speech recognition tool of google is improved by 49%.

Architecture of Tessaract

The connected components called blobs are identified. These blobs are organized in text lines. Text lines are analyzed. Then these lines are broken into words.
Tesseract performs multiple passes to recognize and organize the words. In the first phase, it tries to recognize the words and in the second phase, it passes the received data to the Adaptive classifier as training data. This is the first pass.
In the second pass, it tries to recognize the previously unrecognized characters by running over the page again.

PREREQUISITES

Installations:
sudo apt-get install pytessaract

Libraries:
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from deskew import determine_skew


IMPLEMENTATION

To achieve the greatest accuracy of the tessaract OCR the image must be preprocessed as necessary. In this project, the image goes through different phases namely resizing, deskewing, noise removal, and thresholding. Finally, the processed image is passed through the OCR python library called pytessaract.


1. Deskewing
Deskew is necessary because the tesseract library is contingent upon the orientation of the text. The text should be perpendicular to the y-axis, for the tessaract to recognize the text.
angle = determine_skew(img)
h , w = img.shape[:2]
center = (w//2,h//2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
newImg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

2. Noise Removal
To remove unnecessary small dots of pixels from the image, noise removal should be done. It is done by using slightly blurring the image and removing excessive sharpness.

cv2.medianBlur(img,1)

3. Thresholding
Thresholding is the method of segmenting the image by classifying the pixels in the image as black or white. First, the image is converted to gray and finally thresholding is done. To identify the optimal threshold value i.e the middle value between generated two peaks.

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

4. Tessaract OCR
Finally, the preprocessed image is passed through the command to accurately extract text from the image.

Previous: without normalization + threshold

![without_norm](https://user-images.githubusercontent.com/99968233/212304093-75e6a092-a630-4987-be8b-01e3b3e8f4e4.png)

The shadows affects significant portion of the image thus reducing accuracy of detection.

After: with normalization + threshold

![with_norm](https://user-images.githubusercontent.com/99968233/212304088-60912dc5-d4ea-4686-a592-166c333413c9.png)

Accuracy is significantly improved as a result.

Code for normalization:

rgb_plane = cv2.split(img) #expensive operation // splits into r,g,b
for plane in rgb_plane:
   dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
   bg_img = cv2.medianBlur(dilated_img, 21)
   diff_img = 255 - cv2.absdiff(plane, bg_img)
   norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

Enite Code:

import cv2
from deskew import determine_skew
from skimage.transform import rotate
import numpy as np
import pytesseract
from pytesseract import Output
import os

def deskew(img):
   angle = determine_skew(img)
   newImg = rotate(img,angle,resize=True) * 255
   return newImg.astype(np.uint8)

def normalize(img):
   rgb_plane = cv2.split(img) #expensive operation // splits into r,g,b
   for plane in rgb_plane:
       dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
       bg_img = cv2.medianBlur(dilated_img, 21)
       diff_img = 255 - cv2.absdiff(plane, bg_img)
       norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
       return norm_img

def noise_removal(img):
   return cv2.medianBlur(img,1)

def thresh_img(img):
   img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   return img

def ocr_tessaract(img):
   return pytesseract.image_to_string(img)

def boxes_image(img, original):
   dicti = pytesseract.image_to_data(img, output_type=Output.DICT)
   n_boxes = len(dicti['text'])
   for i in range(n_boxes):
       if int(dicti['conf'][i]) > 60:
           (x, y, w, h) = (dicti['left'][i], dicti['top'][i], dicti['width'][i], dicti['height'][i])
           original = cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
   return original

def final_processing(img):
   dskewed_image = deskew(img)
   new_image = dskewed_image.copy()
   norm_img = normalize(dskewed_image)
   noiselesss_image = noise_removal(norm_img)
   thresh_image = thresh_img(noiselesss_image)
   return thresh_image,new_image

def image_import(path):
   images = []
   for items in os.listdir(path):
       Image = path + '/' + items
       curImage = cv2.imread(Image)
       images.append(curImage)
   return images

if __name__ == '__main__':
   path = 'Resources/High_res'
   images = []
   processed_image_list = []
   image_copy_list = []
   count = 0
   next_count = 0
   images = image_import(path)
   for image in images:
       processed_image, image_copy = final_processing(image)
       processed_image_list.append(processed_image)
       image_copy_list.append(image_copy)

   for items in processed_image_list:
       count = count + 1
       text = ocr_tessaract(items)
       print(f'Extracted Text for Image\n{text}')

   for images in image_copy_list:
       next_count = next_count + 1
       if next_count != 8:
           boxed_image = boxes_image(processed_image_list[next_count],image_copy_list[next_count])
           cv2.namedWindow(f"Boxed Image {next_count}",cv2.WINDOW_NORMAL)
           cv2.imshow(f"Boxed Image {next_count}",boxed_image)

   cv2.waitKey(0)

Result:
The testing was done in total of 8 images. Each image has different resolutions.
Image 1:
![ext_test](https://user-images.githubusercontent.com/99968233/212303977-c94d14e7-0278-4a7d-a572-03fe2548b6b9.png)
Extracted Text:

The no-load operation at transformer is the condition, when primary
side is supplied with source and the secondary side is left open circuit
When primary side is supplied by a source, it draws a small amount al

current called no-load current Io or exciting current.

In ideal transformer where there is no losses in the core at the
transformer, this current Io totally lags the primary voltage v; by 90°
But in actual case, there will be some power loss as iron loss (core loss}

Image 2:
![test1](https://user-images.githubusercontent.com/99968233/212303986-0ae405b1-f862-4612-a9e2-da1dba842487.png)
Extracted Text: 
Aamir Paudel
Damodar Bhandari

PUBLISHER

G.L. Book House Pvt. Ltd.

infront of Thapathali Engineering Campus, Maitighar, Kathmandu

Image 3:
![test2](https://user-images.githubusercontent.com/99968233/212304007-16e7c32a-aa9e-4f90-b0d8-b2ff022731a6.png)
Extracted Text:
212
2.3

DC Generator

31
32
33
34
4S

(B heures)

Operation of Transformer with load

Equivalent Circuits and Phasor Dingraca

Tests: Polarity oF. Open Circuit tet, Short Ciro test and Boqervatena
Circuit Parameters

Voltage Regulstion

Loseca in a transformer

Efficiency, condition for moumum efficency and all dey efficrcacy

rearurnent Trnaformers: Potenual Trenkiorimet (FT) amd Caress
Transformer (CT)

‘Auto transformer: construction, workng proctple and Cs saving

Three phase Transformers

(6 beara)
Construchonal Details end Armature Winding

Working principle and Commutstor Action

EMF equation

Method of excitanon: *eparately and seif excited, Types of DC Generator
Characteristics of series chins apt! =

Image 4:
![test3](https://user-images.githubusercontent.com/99968233/212304015-23617b7a-8962-482f-9500-6cf318e907a0.png)
Extracted Text:
CONTENTS |

Chapters Pages,
41 Magnetic Circuit 1
2 Transformer 23
3 DC Generator 93
4 DC Motor 435
5 Three Phase Induction Machines 167
6 Synchronous Machine 183

Fractional Kilowatt Motors * 203
Image 5:
![test4](https://user-images.githubusercontent.com/99968233/212304030-63130916-f49b-4101-bd94-1fdb24243d9b.png)
Extracted Text:
Three Phase Induction Machines
5 Three Phase Induction Motor : (oy
5.1.1 Constructional Details and Types i
2 ral Principle, Rotatin;
" Steed. Slip. “indaced EMF, ig Magnetic Fit Sync,
Torque Equation aren its Freug
5.1.3 Torque-Slip characteristics %

4.2 Three Phase Induction Generator
5.2.1 Working Principle, voltage build up in an Induction Gen,
erator

5.22 Power Stages
Three Phase Synchronous Machines
6.1 Three Phase Synchronous Generator 6 hoary
6.1.1 Constructional Details, Armature Windings, Types of Rov,

Exciter
6.1.2 Working Principle


Image 6:
![test5](https://user-images.githubusercontent.com/99968233/212304047-ce2bf894-d7f9-4bbf-b767-8e027af7f157.png)
Extracted Text:
Q.4 What is eddy current loss? Discus how can we minimize it.

| => The time varying flux in the core induces emf in the coils according to
faraday's law of electromagnetic induction. But since the core itself is a
conductor (all magnetic material are) emf will also be induced in the
core resulting circulating currents in the core. These currents are known
as eddy current in the core. These currents are known as eddy current &

have a power loss (IR) associated with it. This loss being known as
eddy current loss.

— This loss depends upon the - resistivity of the material

- mean length of the path of the circulating current for a given cross

Image 7:
![test7](https://user-images.githubusercontent.com/99968233/212304078-833af73d-e06a-4125-9bca-8fd8aafb18a9.png)
Extracted Text:
Q.1LA magnetic circuit consists of a circular iron core having e
length of 10 cm & cross sectional area of 100 mm’. The air eu
mm & the core has 600 turns of winding. Calculate the magni .
of currents to be passed through the winding to produce Air §
flux of 1 Tesla.

Given p, = 4000

Se

Image 8:
![test6](https://user-images.githubusercontent.com/99968233/212304063-e8151ad6-6862-48bf-9f75-87dfd7ae3356.png)
Extracted Text:
Q.2 An iron ring of mean length of 1. 2 m and crOSs-sectional
0.05 m? is would with a coil of 900 turns. If a current of 2 at
coil produces a flux density of 2 T in the iron ring. calculate, i

i the mmf
ii, total flux in the core
iii, Magnetic field strength
iv. Relative permeability of the core

Solution;

Detected Texts:
![boxed_images](https://user-images.githubusercontent.com/99968233/212303961-73955fff-8ffa-435a-b6d6-3d9b7e7a03b3.png)
 
