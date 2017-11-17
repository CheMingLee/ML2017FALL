import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, BatchNormalization, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
import h5py
import sys

train = pd.read_csv(sys.argv[1])
train = np.array(train)
label = train[:,0]

feat = []
for i in range(len(train)):
    feat.append(np.reshape(np.array(train[i,1].split(' ')), (48,48)))
    
feat = np.array(feat, float)

# reverse
feat = np.concatenate((feat, feat[:,::-1]), axis=0)
label = np.concatenate((label, label), axis=0)

feat = feat.reshape(len(feat),48,48,1)
feat = feat/255
label = np_utils.to_categorical(label, 7)

# DataGenerator
Xtr, Xval, Ytr, Yval = train_test_split(feat, label, test_size=0.05)
datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=15.)
datagen.fit(Xtr)

# CNN model
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(32,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

model.add(Dense(7))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = Adam(lr=0.0005, decay=5e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# fit
EarlyStopping(monitor='val_loss', patience=5, mode='auto')
acc_his = model.fit_generator(datagen.flow(Xtr, Ytr, batch_size=128), steps_per_epoch=Xtr.shape[0]/128, epochs=150, validation_data=(Xval, Yval))
mdhis = pd.DataFrame(acc_his.history).to_csv('CNN_DataGen_history2.csv')
model.save('model.h5')
