import numpy as np
import cv2
import sys

im = cv2.imread('images/train2me.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
#cv2.imshow('oldthresh',thresh)

#################      Now finding Contours         ###################

#thresh changes after findContours
thresh_copy = thresh.copy()
_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
thresh = thresh_copy

#cv2.imshow('newthresh',thresh)

samples =  np.empty((0,875))
responses = []
keys = [i for i in range(48,58)] #1,2,3...,8,9,0

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>25:
            print("w:",w)
            print("h:",h)
            if w<20:
                diff = 20-w
                
                print(diff)
                x -= diff/2
                x=int(x)
                w += diff
                print("w:",w)
                #print(type(diff),type(x),type(w),type(y),type(h))
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(25,35))
            #cv2.imshow('roi',roi)
            cv2.imshow('roismall',roismall)
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,875))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)