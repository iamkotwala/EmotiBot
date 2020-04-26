import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import os
#----------------------------------------------------
import tkinter as tk
from PIL import Image, ImageTk

#----------------------------------------------------Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
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


def show_frame():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

    # Find haar cascade to draw bounding box around face
    # ret, frame = cap.read()
    # if not ret:
    #   break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))


# emotions will be displayed on your face from the webcam feed
if mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}



    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Slider window (slider controls stage position)
        sliderFrame = tk.Frame(window, width=600, height=100)
        sliderFrame.grid(row=600, column=0, padx=10, pady=2)

        show_frame()  # Display 2
        window.mainloop()  # Starts GUI

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()