#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:30:47 2019

!!Work on train with 1000 this code don't work alone!!

@author: shoichi
"""
%load_ext tensorboard
import numpy as np
import data
import train1000
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D,Activation,MaxPooling2D,BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import preprocessing

def model():
	model = Sequential()
	model.add(Conv2D(64, (5,5), padding="same", input_shape=X_train.shape[1:],activation="relu"))                                            #畳み込み１層目
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (5,5), padding="same", activation="relu"))         #畳み込み2層目
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(BatchNormalization())
	model.add(Dense(10))                                              #全結合2層目
	model.add(BatchNormalization())
	model.add(Activation("softmax"))
	model.summary()
	return model

if __name__ == '__main__':
	(X_train,Y_train),(X_test,Y_test) = train1000.cifar10()
	
	epochs = 50
	batch_size = 128
	nb_classes = 10
	
	model = model()
	model.compile(loss='kullback_leibler_divergence', optimizer=Adam(), metrics=['kullback_leibler_divergence', 'acc'])
	tb_cb = tf.keras.callbacks.TensorBoard(log_dir="keras1000log/")
	result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1,shuffle=True, callbacks=[tb_cb])
	
	print('train data:')
	eva=model.evaluate(X_train,Y_train,verbose=1)
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )
		print()
	print('test data:')
	eva = model.evaluate(X_test,Y_test,verbose=1)
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )
	%tensorboard --logdir keras1000log/
