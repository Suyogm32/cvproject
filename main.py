import cv2
import imutils
import pytesseract
import streamlit as st
from PIL import Image
import io
import os
# adding a file uploader

st.title("Number Plate Recognition")

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'


def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        # st.write(uploadedfile.name)
    return uploadedfile.name

def detect_number_plate(image):
    image = cv2.imread(image)
    image = imutils.resize(image, width=300)
    # cv2.imshow("original image", image)
    # cv2.waitKey(0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("greyed image", gray_image)
    # cv2.waitKey(0)

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
    # cv2.imshow("smoothened image", gray_image)
    # cv2.waitKey(0)

    edged = cv2.Canny(gray_image, 30, 200)
    # cv2.imshow("edged image", edged)
    # cv2.waitKey(0)

    cnts, new = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("contours", image1)
    # cv2.waitKey(0)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    screenCnt = None
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("Top 30 contours", image2)
    # cv2.waitKey(0)

    i = 7
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]
            cv2.imwrite('./' + str(i) + '.png', new_img)
            i += 1
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    # cv2.imshow("image with detected license plate", image)
    # st.image(new_img)
    # cv2.waitKey(0)


    # cv2.imshow("cropped", cv2.imread(Cropped_loc))


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    # image = Image.open(bytes_data)
    st.image(image)
    img_name=save_uploadedfile(uploaded_file)
    detect_number_plate('tempDir/'+img_name)
    Cropped_loc = './7.png'
    st.image(Cropped_loc)
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
    print("Number plate is:", plate)
    st.write("Number plate is:",plate)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

