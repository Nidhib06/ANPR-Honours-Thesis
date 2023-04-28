# ANPR-Honours-Thesis

This thesis project consists of 2 parts in the code. The number plate recognition, and the tkinter GUI. 
To run this code, edit line 23 and replace the file name with the indian_license_plate.xml file given in the repository.

This is the line in the code:
plate_cascade = cv2.CascadeClassifier(r"C:\Users\nidhi\UNIVERSITY\Honors Thesis\archive\indian_license_plate.xml")

Also replace the filename to a desired filename in line 168

line in the code:
with open('C:/Users/nidhi/Test_Results.txt', 'a') as f:

Finally, run the tkinter GUI, select the car you would like to detect the number plate on (a sample image is given in the repository) 
and click the analyse image button to receive the output. 
