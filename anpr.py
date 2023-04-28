# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 01:25:55 2023

@author: nidhi
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

import imutils 
import easyocr
import sys

import os
import pickle

import PIL
from PIL import ImageTk, Image
from tkinter import filedialog

plate_cascade = cv2.CascadeClassifier(r"C:\Users\nidhi\UNIVERSITY\HonorsThesis\archive\indian_license_plate.xml")

def detect_plate(img, text=''): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()
    plate = None
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) # detects number plate and returns the coordinates and dimensions of contours.
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (0,255,255), 3) # drawing rectangles around the edges.
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (0,255,255), 1, cv2.LINE_AA)
        
    return plate_img, plate # processed image. 

# Testing the above function
def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB) 
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111) 
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

# Match contours to license plate
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1] 
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 15 contours for license plate
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread(r'contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            char_copy = np.zeros((44,24))
            # extracting each character 
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

            #invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indices
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


# Find characters in the resulting images
def segment_characters(image) : 
  img_lp = cv2.resize(image, (333, 75))
  img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
  _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  img_binary_lp = cv2.erode(img_binary_lp, (3,3))
  img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

  LP_WIDTH = img_binary_lp.shape[0]
  LP_HEIGHT = img_binary_lp.shape[1]

  # Make borders white
  img_binary_lp[0:3,:] = 255
  img_binary_lp[:,0:3] = 255
  img_binary_lp[72:75,:] = 255
  img_binary_lp[:,330:333] = 255

 # Estimations of character contours sizes of cropped license plates
  dimensions = [LP_WIDTH/6,
                LP_WIDTH/2,
                LP_HEIGHT/10,
                2*LP_HEIGHT/3]
  plt.show()
  cv2.imwrite('contour.jpg',img_binary_lp)

  # Get contours within cropped license plate
  char_list = find_contours(dimensions, img_binary_lp)

  return char_list
    
import re

def filter_special_chars(text):
    # regular expression to match any character that is not a letter or a digit
    regex = r"[^a-zA-Z0-9]"
    # replace all matches of the regular expression with an empty string
    filtered_text = re.sub(regex, "", text)
    # capitalize all the letters
    filtered_text = filtered_text.upper()
    return filtered_text


#%%
def mainfunc(img):
    output_img, platenum = detect_plate(img)
     
    if platenum is None:
        print("No license plate detected") 
        exit()  # exit with an error code of 1
    
    char = segment_characters(platenum)
    segmented_image = 'contour.jpg'
    
    # Use Easy OCR To Read Text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(segmented_image)  
    result
    
    if not result:
        with open('C:/Users/nidhi/OneDrive/Documents/UNIVERSITY/Thirdyear/Honors Thesis/Car_Dataset/Test_Results.txt', 'a') as f:
          f.write(f"\n Image: {img}\n")
          f.write("Error (Plate detected Wrongly) \n")
        print("OCR could not read any text. Program will exit.")
        exit() 
        
    text = result[0][-2] 
    confidence = result[0][-1] 
    
    percentage_str = "{:.2f}%".format(confidence) 
    print(f"The percentage is: {percentage_str}")
    plate_number = filter_special_chars(text)
    print(plate_number)
    
    
    output_img, plate = detect_plate(img, plate_number)
    
   
    output_pil = Image.fromarray(output_img)
   
    
    return output_pil, plate_number, confidence
    
#%%

# Tkinter GUI

from tkinter import *
import PIL
from PIL import ImageTk, Image
from tkinter import filedialog

    
root = Tk()
root.title('ANPR App')
root.geometry('600x400')
root.configure(bg='#020d0c')

def selectimg():
    global my_image, root_filename
   
    
    root_filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    my_image = Image.open(root_filename)
  
    # Resize the image to the desired size
    width, height = 250, 250
    my_image = my_image.resize((width, height))
    
    space_5label = Label(root, height=1, bg='black')
    space_5label.grid(row = 6, column = 1)

    # Convert the PIL image to a PhotoImage and display it on the canvas
    my_image = ImageTk.PhotoImage(my_image)
    my_image_label = Label(image = my_image).grid(row = 6, column = 2) 
    
    

def analyseimg():
  
    global my_image 
    
    pil_img = ImageTk.getimage(my_image) 
  
    np_img = np.asarray(pil_img)
   
    
    width, height = pil_img.size
    
    imagewewant = cv2.imread(root_filename)
    analysedimg, numberplate, confidence_score = mainfunc(imagewewant)

    resize_image = analysedimg.resize((width, height))
 
    # Convert the PIL image to a PhotoImage
    analysed_photo = ImageTk.PhotoImage(resize_image) 
     
    analysed_label = Label(image = analysed_photo) 
    analysed_label.image = analysed_photo
    analysed_label.grid(row = 6, column = 8)
    
    space_2label = Label(root, height=1, bg='black')
    space_2label.grid(row = 7, column = 8)
    
    plate_1label = Label(root, text = f"The number plate is: {numberplate}", font=("Arial", 14), bg="#04dbd4", fg="white")
    plate_1label.grid(row = 8, column = 8) 
    
    space_6label = Label(root, height=1, bg='black')
    space_6label.grid(row = 9, column = 8)
    
    percentage_str = "{:.2f}%".format(confidence_score) 
    
    plate_2label = Label(root, text = f"OCR Confidence Score: {percentage_str}", font=("Arial", 14), bg="#04dbd4", fg="white")
    plate_2label.grid(row = 10, column = 8) 
    
            
space_ulabel = Label(root, height=1, bg='black')
space_ulabel.grid(row = 0, column = 6)    
    
select_btn = Button(root, text = 'Select Image', command = selectimg, font=("Arial", 14), bg="#04dbd4", fg="white", borderwidth=0, padx=20, pady=10, activebackground="#0066B3", activeforeground="white").grid(row = 1, column = 6)

space_label = Label(root, height=1, bg='black')
space_label.grid(row = 2, column = 6)

analyse_btn = Button(root, text = 'Analyse Image', command = analyseimg, font=("Arial", 14), bg="#04dbd4", fg="white", borderwidth=0, padx=20, pady=10, activebackground="#0066B3", activeforeground="white").grid(row = 3, column = 6)

space_tlabel = Label(root, height=1, bg='black')
space_tlabel.grid(row = 4, column = 2)

space_1label = Label(root, height=1, bg='black')
space_1label.grid(row = 4, column = 8)

root.mainloop()
