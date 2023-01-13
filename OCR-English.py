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


