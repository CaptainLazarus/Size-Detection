from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midPoint(a,b):
    c = ((a[0]+b[0])*0.5 , (a[1]+b[1])*0.5)
    return c

def initArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i" , "--image" , help="Input Image" , required=True)
    parser.add_argument("-w" , "--width" , type=int , help="Width of Reference Object" , default=25)
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = initArgs()
    image = cv2.imread(args["image"])
    image = imutils.resize(image , width=600)
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray , (11,11) , 0)

    edged = cv2.Canny(gray , 50 , 100)
    edged = cv2.dilate(edged , None , iterations=1)
    edged = cv2.erode(edged , None , iterations=1)

    cnts = cv2.findContours(edged.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts , _) = contours.sort_contours(cnts)
    ratio = None

    for c in cnts:
        if cv2.contourArea(c) < 25:
            continue
    
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box , dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")] , -1 , (0,0,255) , 2)

        for (x,y) in box:
            cv2.circle(orig , (int(x) , int(y)) , 5 , (0,255,0) , -1)

        (tl,tr,br,bl) = box
        (midTX , midTY) = midPoint(tl , tr)
        (midBX , midBY) = midPoint(bl , br)
        (midLX , midLY) = midPoint(tl , bl)
        (midRX , midRY) = midPoint(tr , br)

        cv2.circle(orig , (int(midTX) , int(midTY)) , 5 , (0,255,255) , -1)
        cv2.circle(orig , (int(midBX) , int(midBY)) , 5 , (0,255,255) , -1)
        cv2.circle(orig , (int(midLX) , int(midLY)) , 5 , (0,255,255) , -1)
        cv2.circle(orig , (int(midRX) , int(midRY)) , 5 , (0,255,255) , -1)

        cv2.line(orig , (int(midTX) , int(midTY)) , (int(midBX) , int(midBY)) , (255,0,255) , 2)
        cv2.line(orig , (int(midLX) , int(midLY)) , (int(midRX) , int(midRY)) , (255,0,255) , 2)

        dA = dist.euclidean((midTX , midTY) , (midBX , midBY))
        dB = dist.euclidean((midLX , midLY) , (midRX , midRY))
        if ratio is None:
            ratio = dA/args["width"]
        
        dimA = dA/ratio
        dimB = dB/ratio

        cv2.putText(orig , "{:.2f}mm".format(dimA) , (int(midTX-10) , int(midTY-10)) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (255,255,255) , 2)
        cv2.putText(orig , "{:.2f}mm".format(dimB) , (int(midRX+10) , int(midRY+10)) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (255,255,255) , 2)


        cv2.imshow("Image" , orig)
        cv2.waitKey(0)