import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input

def create_model(num_classes=4, input_shape=(224,224,3)):
    model = Sequential()

    model.add(Input(shape=(224,224)))

    # Block 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 2
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    #Block 3
    model.add(Conv2D(128), (3,3), activation='relu', padding='same')
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))


    return model