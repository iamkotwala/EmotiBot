#import numpy as np
#import matplotlib as plt
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
a = ap.parse_args()
mode = a.mode

#def plot_model_history(model_history):
#    """
#    Plot Accuracy and Loss curves given the model_history
#    """
#    fig, axs = plt.subplots(1,2,figsize=(15,5))
#    # summarize history for accuracy
#    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
#    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#    axs[0].set_title('Model Accuracy')
#    axs[0].set_ylabel('Accuracy')
#    axs[0].set_xlabel('Epoch')
#    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#    axs[0].legend(['train', 'val'], loc='best')
#    # summarize history for loss
#    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#    axs[1].set_title('Model Loss')
#    axs[1].set_ylabel('Loss')
#    axs[1].set_xlabel('Epoch')
#    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#    axs[1].legend(['train', 'val'], loc='best')
#    fig.savefig('plot.png')
#    plt.show()
    
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

#model.summary()
# Train the model
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy']) #configures the model for training.
    # weight decay:  While weight decay is an additional term in the weight update rule that causes the weights to exponentially decay to zero.

    model_info = model.fit_generator(  #keras method for fitting
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)

#    plot_model_history(model_info)
    model.save_weights('model1.h5')
