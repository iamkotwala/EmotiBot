from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob, cv2, pickle, os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

nImageRows = 48
nImageCols = 48
nChannels = 1

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights("model.h5")


actual = []
predicted = []

states = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]
binary_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

error_count = 0
for state in states:
	images = glob.glob("data/test/"+str(state)+"/*")
	
	for index, image_path in enumerate(images):
		image = cv2.imread(image_path)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
		prediction = model.predict(cropped_img)
		maxindex = int(np.argmax(prediction))
		actual.append(state)
		predicted.append(binary_dict[maxindex])
		
		
	print("")
print("Errors :",error_count)
f = open("./tensorflow_prediction.pickle", "wb")
f.write(pickle.dumps([actual,predicted]))

f.close()
