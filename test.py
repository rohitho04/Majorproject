import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

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


X = np.load("models/X.npy")
Y = np.load("models/Y.npy")
label = np.load("models/label.npy")
names = np.load("models/names.npy")

print(X.shape)
print(Y.shape)
print(label.shape)
print(names.shape)


if os.path.exists("models/adaboost.txt"):
    with open('models/adaboost.txt', 'rb') as file:
        clf = pickle.load(file)
    file.close()
else:
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X, Y)
    with open('models/adaboost.txt', 'wb') as file:
        pickle.dump(clf, file)
    file.close()


image = cv2.imread("Dataset/SatelliteImages/154.png")
image1 = image
canny = cannyDetection(image)
hough = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
if hough is not None:
    try:
        lines = calculateLines(image, hough)
        linesVisualize = visualizeLines(image, lines)
        output = cv2.addWeighted(image, 0.9, linesVisualize, 1, 1)
        image1 = output
        height, width, channel = output.shape
        img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):
            for m in range(0, width):
                img_lbp[i, m] = lbp_calculated_pixel(img_gray, i, m)
        lbp = img_lbp        
        img_lbp = cv2.resize(img_lbp, (28, 28))
        img_lbp = img_lbp.ravel()
        temp = []
        temp.append(img_lbp)
        temp = np.asarray(temp)
        predict = clf.predict(temp)[0]
        lbl = 0
        for k in range(len(names)):
            if names[k] == str(predict)+".png":
                lbl = k
                break
        print(lbl)    
        road_extract = label[lbl]
        cv2.imshow("Satellite Image", image1)
        cv2.imshow("Extracted Road", road_extract)
        cv2.imshow("canny Image", canny)
        cv2.imshow("LBP Image", lbp)
        cv2.waitKey(0)
        print(predict)
    except Exception:
        pass


