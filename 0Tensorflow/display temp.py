import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import *
from PIL import Image, ImageTk
import cv2

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
a = ap.parse_args()
mode = a.mode

win = Tk()
win.title('EMOTIBOT')
win.config(background = "#D9D9D9")
win.resizable(width=FALSE, height=FALSE)

#frame1 for webcamera
frame1 = Frame(win, width=600, bg="black",height=300, padx=10, pady=10,highlightbackground="grey", highlightcolor="black", highlightthickness=5)
frame1.pack(side=LEFT ,fill=Y,padx=10, pady=10)



#frame2 for logo
frame2 = Frame(win, bg='#D9D9D9', width=300, height=5,padx=10, pady=10,highlightbackground="grey", highlightcolor="black")
frame2.pack(side = TOP,fill=X,padx=10) 
photo = PhotoImage(file='C:\\Users\\iamvr\\Desktop\\EmotiBot\\LogosDesign\\logom.png')
l = Label(frame2, image=photo,padx=10,pady=10)
l.pack()
l1 = Label(frame2, text="logs",font=("Helvetica", 16),bg="#D9D9D9", fg="#0E7A3F")
l1.pack(side=BOTTOM,fill=Y,padx=2)

#frame3 for logs
frame3 = Frame(win, bg='black', width=300, height=150,padx=5, pady=5,highlightbackground="#707173", highlightcolor="black")
frame3.pack(fill=X,expand=True,padx=5, pady=5)
frame3.pack_propagate(0) #stops frame from shrinking

 
#frame4 for buttons
frame4 = Frame(win, bg='grey', width=300, height=60,padx=5, pady=5,highlightbackground="grey", highlightcolor="black")
frame4.pack(side=BOTTOM, fill=BOTH,expand=1,padx=5,pady=10)

b = Button(frame4, text='ON', padx = 40)
b.pack(side=LEFT,pady = 20, padx = 20)
c = Button(frame4, text='OFF', padx=40)
c.pack(side=RIGHT,pady = 20, padx = 20)

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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48), #2304
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = vali_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48), #2304
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

#the model
model = Sequential() #stack layers model. tf.keras model

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
#layer one. inpput layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#kernel size is to divide 48x48 imge into 3x3 actiavtion fn
model.add(MaxPooling2D(pool_size=(2, 2)))
#take the max of that region and create a new, output matrix
model.add(Dropout(0.25))
#preventing the model from overfitting, where randomly selected neurons are dropped out

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #convert matrix into vector
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5)) #dropout was udes to minimize overfitting.
model.add(Dense(7, activation='softmax')) #output layer

# emotions will be displayed on your face from the webcam feed
if mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    
    def show_frame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        #while True:
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(cv2image,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame) 
        
    #Capture video frames
    lmain = Label(frame1)
    lmain.grid(row=0, column=0)
    
    show_frame()  #Display 2
    win.mainloop()
     

    #cap.release()
    #cv2.destroyAllWindows()