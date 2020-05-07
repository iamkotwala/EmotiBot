#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import Frame, Tk, PhotoImage, Button, Label
from tkinter.constants import RIGHT, LEFT, TOP, BOTTOM, X, Y, BOTH
from PIL import Image, ImageTk
import time
import mysql.connector
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#---------------------------------------------------------------------------------------

#Create the connection object   
myconn = mysql.connector.connect(host = "localhost", user = "root", passwd = "", database = "EmotiBot")
#creating the cursor object  
cur = myconn.cursor()
def insertdb(emotio, accu, timstmp):
    sql = "insert into logs(Emotion, Accuracy, TimeStamp) values (%s, %s, %s)"
    val = (emotio, accu, timstmp)
    try:
        #inserting the values into the table
        cur.execute(sql,val)
        #commit the transaction 
        myconn.commit()
    except:
        myconn.rollback()
        print("failed")
    
#---------------------------------------------------------------------------------------

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', help='train/display')
a = ap.parse_args()
mode = a.mode

# Define data
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# 28709(train data)/64(batch size) = 448 iterations to complete 1 epoch
train_datagen = ImageDataGenerator(rescale=1. / 255)
vali_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
        target_size=(48, 48), batch_size=batch_size,
        color_mode='grayscale', class_mode='categorical')  # 2304

validation_generator = vali_datagen.flow_from_directory(val_dir,
        target_size=(48, 48), batch_size=batch_size,
        color_mode='grayscale', class_mode='categorical')  # 2304

# the model
model = Sequential()  # stack layers model. tf.keras model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
          input_shape=(48, 48, 1)))
# layer 1. input layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# kernel size is to divide 48x48 imge into 3x3 actiavtion fn
model.add(MaxPooling2D(pool_size=(2, 2)))
# take the max of that region and create a new, output matrix
model.add(Dropout(0.25))
# preventing the model from overfitting, where randomly selected neurons are dropped out
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # convert matrix into vector

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))  # dropout was udes to minimize overfitting.
model.add(Dense(7, activation='softmax'))  # output layer


#---------------------------------------------------------------------------------------
# main tk
win = Tk()
win.iconbitmap('C:\\Users\\iamvr\\Desktop\\EmotiBot\\Logos & Images\\logoico.ico')
win.title('EmotiBot (Made with love in Python)')
win.config(background='#D9D9D9')
win.resizable(width=False, height=False)
    
#def on():
    
# frame1 for webcamera
frame1 = Frame(
    win,
    width=600,
    bg='black',
    height=300,
    padx=10,
    pady=10,
    highlightcolor='black')
frame1.pack(side=LEFT, fill=Y, padx=10, pady=10, expand=True)

#def off():
#    cap.release()
#    l1 = Label(frame1, text="turn ON the webcam",font=("Helvetica", 12),bg="white", fg="#0E7A3F")
#    l1.pack(fill=Y,padx=150,pady=300)
    
# frame2 for logo
frame2 = Frame(
    win,
    bg='#D9D9D0',
    width=310,
    height=5,
    padx=0,
    pady=0)
frame2.pack(side=TOP, fill=X, padx=(0, 10), pady=(10, 0))
photo = PhotoImage(file='C:\\Users\\iamvr\\Desktop\\EmotiBot\\Logos & Images\\logom.png')
l = Label(frame2, image=photo, padx=0, pady=0)
l.pack()

# log title
l1 = Label(frame2, text='Logs', font=('Helvetica', 14, 'italic'), bg='#D9D9D9', fg='#000000')
l1.pack(side=BOTTOM, fill=Y, padx=2)

# frame3 for logs
frame3 = Frame(
    win,
    bg='black',
    width=310,
    height=140,
    padx=0,
    pady=0)
frame3.pack(fill=X, expand=True, padx=(0, 10), pady=0)
frame3.pack_propagate(0)  # stops frame from shrinking



# frame4 for buttons
frame4 = Frame(
    win,
    bg='#D9D9D9',
    width=300,
    height=60,
    padx=0,
    pady=0)
frame4.pack(side=BOTTOM, fill=BOTH, expand=1, padx=(0, 10), pady=(0, 10))

# button1
img = PhotoImage(file='C:\\Users\\iamvr\\Desktop\\EmotiBot\\Logos & Images\\play.png')  # make sure to add "/" not "\"
photoimage = img.subsample(2, 2)
#bon = Button(frame4, text='Camera ON ', image=photoimage, compound=LEFT, command=lambda : on())
#bon.pack(side=LEFT, pady=(12, 0), padx=(40, 0))

# button2

img2 = PhotoImage(file='C:\\Users\\iamvr\\Desktop\\EmotiBot\\Logos & Images\\stop.png')  # make sure to add "/" not "\"
photoimage2 = img2.subsample(2, 2)
#boff = Button(frame4, text=' Camera OFF ', image=photoimage2, compound=LEFT, command=lambda : off())
#boff.pack(side=RIGHT, pady=(12, 0), padx=(0, 40))

#---------------------------------------------------------------------------------------
emotion_list = []
def show_frame():
        (_, frame) = cap.read()
        frame = cv2.flip(frame, 1)
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(cv2image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10),(255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame,emotion_dict[maxindex],(x + 20, y - 60),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),2,cv2.LINE_AA)
            #print(emotion_dict[maxindex])
            logs = Label(frame3, text = (emotion_dict[maxindex], time.ctime()), font=("Consolas", 9), bg="#000000", fg="#ffffff")
            logs.pack(pady=(0, 0))
            
            emotion_list.append(maxindex) 
            if (((emotion_list[len(emotion_list)-2]) != maxindex) or len(emotion_list)==1):
                insertdb(emotion_dict[maxindex], 89.9, time.ctime())
                
        # cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
        
 
# emotions will be displayed on your face from the webcam feed
if mode == 'display':
    model.load_weights('model.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {
        0: 'Angry',
        1: 'Disgusted',
        2: 'Fearful',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprised'}

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    # Capture video frames
    lmain = Label(frame1)
    lmain.grid(row=0, column=0, pady=(3, 0))
    show_frame()  # Display 2
    win.mainloop()


    # if cv2.waitKey(1) & 0xFF == ord('q'):
#cap.release()
#cv2.destroyAllWindows()