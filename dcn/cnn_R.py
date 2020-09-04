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
from keras.optimizers import rmsprop,Adam
read_temp=scipy.io.loadmat('data2_trainlow.mat')
S_est=read_temp['S_est']
S_abs=read_temp['S_abs']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
S_label1 = np.expand_dims(S_label, 2)
[Sample,L,dim]=np.shape(S_est)
nb_epoch=600
batch_size=64

optimizer=Adam(lr=0.001)

cnn_tanh = Sequential() 
cnn_tanh.add(Convolution1D(12,25,  input_shape=(L,dim), activation='tanh',name="cnn_1", padding='same'))
cnn_tanh.add(Convolution1D(6,15, activation='tanh',name="cnn_2", padding='same'))
cnn_tanh.add(Convolution1D(3,5, activation='tanh',name="cnn_4", padding='same'))
cnn_tanh.add(Convolution1D(1,3,activation='tanh',name="cnn_5", padding='same'))
cnn_tanh.compile(loss='mse', optimizer=optimizer)
cnn_tanh.summary()
history_cnn_tanh=cnn_tanh.fit(S_est, S_label1,epochs=nb_epoch, batch_size=batch_size,shuffle=True
                ,verbose=2,validation_split=0.2)




cnn_sigmoid = Sequential() 
cnn_sigmoid.add(Convolution1D(12,25,  input_shape=(L,dim), activation='sigmoid',name="cnn_1", padding='same'))
cnn_sigmoid.add(Convolution1D(6,15, activation='sigmoid',name="cnn_2", padding='same'))
cnn_sigmoid.add(Convolution1D(3,5, activation='sigmoid',name="cnn_4", padding='same'))
cnn_sigmoid.add(Convolution1D(1,3,activation='sigmoid',name="cnn_5", padding='same'))
cnn_sigmoid.compile(loss='mse', optimizer=optimizer)
cnn_sigmoid.summary()

history_cnn_sigmoid=cnn_sigmoid.fit(S_est, S_label1,epochs=nb_epoch, batch_size=batch_size,shuffle=True
                ,verbose=2,validation_split=0.2)

#
cnn = Sequential() 
cnn.add(Convolution1D(12,25,  input_shape=(L,dim), activation='relu',name="cnn_1", padding='same'))
cnn.add(Convolution1D(6,15, activation='relu',name="cnn_2", padding='same'))
cnn.add(Convolution1D(3,5, activation='relu',name="cnn_4", padding='same'))
cnn.add(Convolution1D(1,3,activation='relu',name="cnn_5", padding='same'))
cnn.compile(loss='mse', optimizer=optimizer)
cnn.summary()
history_cnn=cnn.fit(S_est, S_label1,epochs=nb_epoch, batch_size=batch_size,shuffle=True
                ,verbose=2,validation_split=0.2)

#
#



dnn = Sequential() 
dnn.add(Dense(int(2*L/3), activation='relu', input_dim=2*L))
dnn.add(Dense(int(4*L/9), activation='relu'))
dnn.add(Dense(int(2*L/3), activation='relu'))
dnn.add(Dense(L, activation='relu'))
dnn.compile(loss='mse', optimizer=optimizer)
dnn.summary()
history_dnn=dnn.fit(S_abs, S_label,epochs=nb_epoch, batch_size=batch_size,
                        shuffle=True,verbose=2,validation_split=0.2)

#

dnn_sigmoid = Sequential() 
dnn_sigmoid.add(Dense(int(2*L/3), activation='sigmoid', input_dim=2*L))
dnn_sigmoid.add(Dense(int(4*L/9), activation='sigmoid'))
dnn_sigmoid.add(Dense(int(2*L/3), activation='sigmoid'))
dnn_sigmoid.add(Dense(L, activation='sigmoid'))
dnn_sigmoid.compile(loss='mse', optimizer=optimizer)
dnn_sigmoid.summary()
history_dnn_sigmoid=dnn_sigmoid.fit(S_abs, S_label,epochs=nb_epoch, batch_size=batch_size,
                        shuffle=True,verbose=2,validation_split=0.2)

dnn_tanh = Sequential() 
dnn_tanh.add(Dense(int(2*L/3), activation='tanh', input_dim=2*L))
dnn_tanh.add(Dense(int(4*L/9), activation='tanh'))
dnn_tanh.add(Dense(int(2*L/3), activation='tanh'))
dnn_tanh.add(Dense(L, activation='tanh'))
dnn_tanh.compile(loss='mse', optimizer=optimizer)
dnn_tanh.summary()
history_dnn_tanh=dnn_tanh.fit(S_abs, S_label,epochs=nb_epoch, batch_size=batch_size,
                        shuffle=True,verbose=2,validation_split=0.2)

[r2,c]=np.shape(R_est)
P=6
I=120
t=int(np.floor(I/P))
autoencoder= keras.models.load_model( 'autoencoder.h5')

from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est)
Y_autocode_filter=autoencoder.predict(R_est)
Y_autocode_filter[:,0*c:1*c]=normalizer.transform(Y_autocode_filter[:,0*c:1*c])
Y_autocode_filter[:,1*c:2*c]=normalizer.transform(Y_autocode_filter[:,1*c:2*c])
Y_autocode_filter[:,2*c:3*c]=normalizer.transform(Y_autocode_filter[:,2*c:3*c])
Y_autocode_filter[:,3*c:4*c]=normalizer.transform(Y_autocode_filter[:,3*c:4*c])
Y_autocode_filter[:,4*c:5*c]=normalizer.transform(Y_autocode_filter[:,4*c:5*c])
Y_autocode_filter[:,5*c:6*c]=normalizer.transform(Y_autocode_filter[:,5*c:6*c])

model_liu1 = Sequential() 
model_liu1.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu1.add(Dense(int(4*c/9), activation='tanh'))
model_liu1.add(Dense(t, activation='tanh'))
model_liu1.compile(loss='mse', optimizer=optimizer)
history_liu1=model_liu1.fit(Y_autocode_filter[:,:c], S_label[:,:t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)



model_liu2 = Sequential() 
model_liu2.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu2.add(Dense(int(4*c/9), activation='tanh'))
model_liu2.add(Dense(t, activation='tanh'))
model_liu2.compile(loss='mse', optimizer=optimizer)
history_liu2=model_liu2.fit(Y_autocode_filter[:,c:2*c], S_label[:,t:2*t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)

   
model_liu3 = Sequential() 
model_liu3.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu3.add(Dense(int(4*c/9), activation='tanh'))
model_liu3.add(Dense(t, activation='tanh'))
model_liu3.compile(loss='mse', optimizer=optimizer)
history_liu3=model_liu3.fit(Y_autocode_filter[:,2*c:3*c], S_label[:,2*t:3*t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)

model_liu4 = Sequential() 
model_liu4.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu4.add(Dense(int(4*c/9), activation='tanh'))
model_liu4.add(Dense(t, activation='tanh'))
model_liu4.compile(loss='mse', optimizer=optimizer)
history_liu4=model_liu4.fit(Y_autocode_filter[:,3*c:4*c], S_label[:,3*t:4*t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)

model_liu5 = Sequential() 
model_liu5.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu5.add(Dense(int(4*c/9), activation='tanh'))
model_liu5.add(Dense(t, activation='tanh'))
model_liu5.compile(loss='mse', optimizer=optimizer)
history_liu5=model_liu5.fit(Y_autocode_filter[:,4*c:5*c], S_label[:,4*t:5*t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)

model_liu6 = Sequential() 
model_liu6.add(Dense(int(2*c/3), activation='tanh', input_dim=c))
model_liu6.add(Dense(int(4*c/9), activation='tanh'))
model_liu6.add(Dense(t, activation='tanh'))
model_liu6.compile(loss='mse', optimizer=optimizer)
history_liu6=model_liu6.fit(Y_autocode_filter[:,5*c:6*c], S_label[:,5*t:6*t],epochs=nb_epoch
                            , batch_size=batch_size,shuffle=True,verbose=2,validation_split=0.2)


figsize = 8,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(np.array(history_cnn.history['val_loss'])*1000)
plt.plot(np.array(history_cnn_tanh.history['val_loss'])*1000)
plt.plot(np.array(history_cnn_sigmoid.history['val_loss'])*1000)
plt.plot(np.array(history_liu1.history['val_loss'])*1000)
plt.plot(np.array(history_liu2.history['val_loss'])*1000)
plt.plot(np.array(history_liu3.history['val_loss'])*1000)
plt.plot(np.array(history_liu4.history['val_loss'])*1000)
plt.plot(np.array(history_liu5.history['val_loss'])*1000)
plt.plot(np.array(history_liu6.history['val_loss'])*1000)
plt.plot(np.array(history_dnn.history['val_loss'])*1000)
plt.plot(np.array(history_dnn_tanh.history['val_loss'])*1000)
plt.plot(np.array(history_dnn_sigmoid.history['val_loss'])*1000)
plt.legend(['DCN+relu','DCN+tanh','DCN+sig','Method in [5]1','Method in [5]2','Method in [5]3','Method in [5]4','Method in [5]5','Method in [5]6','DNN+relu','DNN+tanh','DNN+sig'])
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 13,}
plt.xlabel('Epoch',font2)
plt.ylabel('Test MSE(*1e$^-$$^3$)',font2)

plt.ylim([8,20])

plt.show()
#
#
#
figsize = 8,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(np.array(history_cnn.history['loss'])*1000)
plt.plot(np.array(history_cnn_tanh.history['loss'])*1000)
plt.plot(np.array(history_cnn_sigmoid.history['loss'])*1000)
plt.plot(np.array(history_liu1.history['loss'])*1000)
plt.plot(np.array(history_liu2.history['loss'])*1000)
plt.plot(np.array(history_liu3.history['loss'])*1000)
plt.plot(np.array(history_liu4.history['loss'])*1000)
plt.plot(np.array(history_liu5.history['loss'])*1000)
plt.plot(np.array(history_liu6.history['loss'])*1000)
plt.plot(np.array(history_dnn.history['loss'])*1000)
plt.plot(np.array(history_dnn_tanh.history['loss'])*1000)
plt.plot(np.array(history_dnn_sigmoid.history['loss'])*1000)
plt.legend(['DCN+relu','DCN+tanh','DCN+sig','Method in [5]1','Method in [5]2','Method in [5]3','Method in [5]4','Method in [5]5','Method in [5]6','DNN+relu','DNN+tanh','DNN+sig'])
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 13,}
plt.xlabel('Epoch',font2)

plt.ylim([8,20])
plt.ylabel('Train MSE(*1e$^-$$^3$)',font2)

plt.show()
#
#
