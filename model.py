from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization,Flatten


def MakeModel():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="Same", activation='relu', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3,3), padding="Same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2))) 
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3,3), padding="Same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2))) 
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model
