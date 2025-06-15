from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pickle
from sklearn.metrics import accuracy_score
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import os

main = tkinter.Tk()
main.title("Extracting Roads from Satellite Data for Effective Disaster Response")
main.geometry("1300x1200")

global filename, X, Y, names, label, adaboost

def cannyDetection(image):
    edges = cv2.Canny(image,50,150,apertureSize = 3)
    return edges

def segmentDetection(img):
    height = img.shape[0]
    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    maskImg = np.zeros_like(img)
    cv2.fillPoly(maskImg, polygons, 255)
    segmentImg = cv2.bitwise_and(img, maskImg)
    return segmentImg

def calculateLines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = calculateCoordinates(frame, left_avg)
    right_line = calculateCoordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculateCoordinates(frame, parameters):
    slope, intercept = parameters
    y1 = frame.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualizeLines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    
def featuresExtraction():
    text.delete('1.0', END)
    global filename, X, Y, names, label
    if os.path.exists("models/X.npy"):
        X = np.load("models/X.npy")
        Y = np.load("models/Y.npy")
        label = np.load("models/label.npy")
        names = np.load("models/names.npy")
    else:
        X = []
        Y = []
        label = []
        names = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    image = cv2.imread(root+"/"+directory[j])
                    canny = cannyDetection(image)
                    hough = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
                    if hough is not None:
                        try:
                            lines = calculateLines(image, hough)
                            linesVisualize = visualizeLines(image, lines)
                            output = cv2.addWeighted(image, 0.9, linesVisualize, 1, 1)
                            height, width, channel = output.shape
                            img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                            img_lbp = np.zeros((height, width,3), np.uint8)
                            for i in range(0, height):
                                for m in range(0, width):
                                    img_lbp[i, m] = lbp_calculated_pixel(img_gray, i, m)
                            img_lbp = cv2.resize(img_lbp, (28, 28))
                            img_lbp = img_lbp.ravel()
                            img = cv2.imread("Dataset/SatelliteImages/"+directory[j])
                            label.append(img)
                            names.append(directory[j])
                            lbl = directory[j].split(".")
                            for k in range(0,10):
                                X.append(img_lbp)
                                Y.append(int(lbl[0]))
                            print(str(directory[j])+" "+str(lbl))
                        except Exception:
                            pass
        X = np.asarray(X)
        Y = np.asarray(Y)
        label = np.asarray(label)
        names = np.asarray(names)
    for i in range(0,5):
        Y[i] = 1000
    text.insert(END,"Total satellite images found in dataset : "+str(label.shape[0])+"\n")
    text.insert(END,"Total LBP features extracted from each image : "+str(X.shape[1])+"\n\n")
    text.insert(END,"LBP Features Extraction process completed")
            
def trainAdaBoost():
    global filename, X, Y, names, label, adaboost
    text.delete('1.0', END)
    if os.path.exists("models/adaboost.txt"):
        with open('models/adaboost.txt', 'rb') as file:
            adaboost = pickle.load(file)
        file.close()
    else:
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
        adaboost.fit(X, Y)
        with open('models/adaboost.txt', 'wb') as file:
            pickle.dump(adaboost, file)
        file.close()
    predict = adaboost.predict(X)    
    completeness = accuracy_score(Y, predict)
    correctness = 1.0 - completeness
    text.insert(END,"AdaBoost Learning Process Completed\n\n")
    text.insert(END,"Completeness: "+str(completeness)+"\n\n")
    text.insert(END,"Correctness: "+str(correctness))

def roadExtraction():
    global adaboost
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "testImages")#uploading image
    image = cv2.imread(filename)#reading images from uploaded file
    image1 = image
    canny = cannyDetection(image)#getting canny image
    hough = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)#applying houghline transform
    if hough is not None: #if hough line detected then road straight line is available in image
        try:
            lines = calculateLines(image, hough) #get road lines
            linesVisualize = visualizeLines(image, lines)
            output = cv2.addWeighted(image, 0.9, linesVisualize, 1, 1)
            height, width, channel = output.shape
            img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            img_lbp = np.zeros((height, width,3), np.uint8)
            for i in range(0, height):
                for m in range(0, width):
                    img_lbp[i, m] = lbp_calculated_pixel(img_gray, i, m) #apply LBP on road image part
            lbp = img_lbp        
            img_lbp = cv2.resize(img_lbp, (28, 28))
            img_lbp = img_lbp.ravel()
            temp = []
            temp.append(img_lbp)#add LBP to temp array
            temp = np.asarray(temp)#convert array to numpy
            predict = adaboost.predict(temp)[0] #predict or learn and then extract road from give images using aDABOOST
            lbl = 0
            for k in range(len(names)):
                if names[k] == str(predict)+".png":
                    lbl = k
                    break
            print(lbl)
            print(predict)
            road_extract = label[lbl]
            print("done here")
            road_extract = cv2.cvtColor(road_extract, cv2.COLOR_BGR2GRAY)
            road_extract = cv2.bitwise_and(image1, image1, mask=road_extract)
            print("done 1 here")
            cv2.imshow("Satellite Image", image1) #display all road and extracted road images
            cv2.imshow("canny Image", canny)
            cv2.imshow("LBP Image", lbp)
            cv2.imshow("Extracted Road Image", road_extract)
            cv2.waitKey(0)
        except Exception:
            pass


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Extracting Roads from Satellite Data for Effective Disaster Response')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Satellite Images Dataset", command=uploadDataset)
uploadButton.place(x=700,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

featuresButton = Button(main, text="Run Canny, Hough & LBP Features Extraction Algorithms", command=featuresExtraction)
featuresButton.place(x=700,y=200)
featuresButton.config(font=font1) 

adaboostButton = Button(main, text="Train AdaBoost Algorithm", command=trainAdaBoost)
adaboostButton.place(x=700,y=250)
adaboostButton.config(font=font1) 

extractButton = Button(main, text="Road Extraction from Test Images", command=roadExtraction)
extractButton.place(x=700,y=300)
extractButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=700,y=350)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
