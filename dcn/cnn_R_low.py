# -*- coding: utf-8 -*-

from keras.models import Model #泛型模型  
import matplotlib.pyplot as plt 
from keras.models import Sequential
import numpy as np
import scipy.io
import keras.layers
from keras.layers import Dense, Input ,Dropout
from keras.layers import Convolution1D
from datetime import datetime
from keras.optimizers import rmsprop
read_temp=scipy.io.loadmat('data2_trainlow.mat')
S_est=read_temp['S_est']
S_abs=read_temp['S_abs']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
S_label1 = np.expand_dims(S_label, 2)
[Sample,L,dim]=np.shape(S_est)
nb_epoch=3
batch_size=64

#

cnn_low = Sequential() 
cnn_low.add(Convolution1D(12,25,  input_shape=(L,dim), activation='relu',name="cnn_1", padding='same'))
cnn_low.add(Convolution1D(6,15, activation='relu',name="cnn_2", padding='same'))
cnn_low.add(Convolution1D(3,5, activation='relu',name="cnn_4", padding='same'))
cnn_low.add(Convolution1D(1,3,activation='relu',name="cnn_5", padding='same'))
cnn_low.compile(loss='mse', optimizer='adam')
cnn_low.summary()
start_time = datetime.now()
history_cnn_low=cnn_low.fit(S_est, S_label1,epochs=nb_epoch, batch_size=batch_size,shuffle=True
                ,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_cnn_low=stop_time-start_time
cnn_low.save('cnn_low.h5')

##
[r2,c]=np.shape(R_est)
P=6
I=120
t=int(np.floor(I/P))
autoencoder= keras.models.load_model( 'autoencoder.h5')
optimizer=rmsprop(lr=0.001)
from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est)
Y_autocode_filter=autoencoder.predict(R_est)
Y_autocode_filter[:,0*c:1*c]=normalizer.transform(Y_autocode_filter[:,0*c:1*c])
Y_autocode_filter[:,1*c:2*c]=normalizer.transform(Y_autocode_filter[:,1*c:2*c])
Y_autocode_filter[:,2*c:3*c]=normalizer.transform(Y_autocode_filter[:,2*c:3*c])
Y_autocode_filter[:,3*c:4*c]=normalizer.transform(Y_autocode_filter[:,3*c:4*c])
Y_autocode_filter[:,4*c:5*c]=normalizer.transform(Y_autocode_filter[:,4*c:5*c])
Y_autocode_filter[:,5*c:6*c]=normalizer.transform(Y_autocode_filter[:,5*c:6*c])

model_low_liu1 = Sequential() 
model_low_liu1.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu1.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu1.add(Dense(t, activation='tanh'))
model_low_liu1.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu1=model_low_liu1.fit(Y_autocode_filter[:,:c], S_label[:,:t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu1=stop_time-start_time

model_low_liu2 = Sequential()
model_low_liu2.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu2.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu2.add(Dense(t, activation='tanh'))
model_low_liu2.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu2=model_low_liu2.fit(Y_autocode_filter[:,c:2*c], S_label[:,1*t:2*t],
                            epochs=nb_epoch, batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu2=stop_time-start_time

model_low_liu3 = Sequential()
model_low_liu3.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu3.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu3.add(Dense(t, activation='tanh'))
model_low_liu3.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu3=model_low_liu3.fit(Y_autocode_filter[:,2*c:3*c], S_label[:,2*t:3*t],
                            epochs=nb_epoch, batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu3=stop_time-start_time

model_low_liu4 = Sequential()
model_low_liu4.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu4.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu4.add(Dense(t, activation='tanh'))
model_low_liu4.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu4=model_low_liu4.fit(Y_autocode_filter[:,3*c:4*c], S_label[:,3*t:4*t],
                            epochs=nb_epoch, batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu4=stop_time-start_time

model_low_liu5 = Sequential()
model_low_liu5.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu5.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu5.add(Dense(t, activation='tanh'))
model_low_liu5.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu5=model_low_liu5.fit(Y_autocode_filter[:,4*c:5*c], S_label[:,4*t:5*t],
                            epochs=nb_epoch, batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu5=stop_time-start_time

model_low_liu6 = Sequential()
model_low_liu6.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_low_liu6.add(Dense(int(4*c/9), activation='tanh'))
model_low_liu6.add(Dense(t, activation='tanh'))
model_low_liu6.compile(loss='mse', optimizer=optimizer)
start_time = datetime.now()
history_liu6=model_low_liu6.fit(Y_autocode_filter[:,5*c:6*c], S_label[:,5*t:6*t],
                            epochs=nb_epoch, batch_size=batch_size 
                            ,shuffle=True,verbose=2,validation_split=0.99)
stop_time = datetime.now()
time_liu6=stop_time-start_time
model_low_liu1.save('model_low_liu1.h5')
model_low_liu2.save('model_low_liu2.h5')
model_low_liu3.save('model_low_liu3.h5')
model_low_liu4.save('model_low_liu4.h5')
model_low_liu5.save('model_low_liu5.h5')
model_low_liu6.save('model_low_liu6.h5')

