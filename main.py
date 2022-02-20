import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle



def detect_mask(image,model):
    prediction = model.predict(image.reshape(1,224,224,3))
    return prediction[0][0]<0.3


def detect_face(image):
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    coods = haar.detectMultiScale(image)
    return coods


def draw_label(image,text,pos,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    x1 = pos[0]
    x2 = pos[0] + text_size[0][0] + 2
    y1 = pos[1]
    y2 = pos[1] + text_size[0][1] - 2
    
    cv2.rectangle(image,pos,(x2,y2),bg_color,cv2.FILLED)
    cv2.putText(image,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
    
    
def run():
    model = tf.keras.models.load_model('model.h5')
    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()

        image = cv2.resize(frame,(224,224))
        prediction = detect_mask(image,model)

        if prediction:
            draw_label(frame,"Has mask",(30,30),(0,255,0))
            coordinates = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            for x,y,w,h in coordinates:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        else:
            draw_label(frame,"No mask",(30,30),(0,0,255))
            coordinates = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            for x,y,w,h in coordinates:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

        cv2.imshow("Mask Detector",frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    capture.release()




def main():
    run()


if __name__ == "__main__":
    main()