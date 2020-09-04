# -*- coding: utf-8 -*-

import keras.models
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.signal
import heapq
from sklearn.svm import SVR
from sklearn.externals import joblib


autoencoder= keras.models.load_model('autoencoder.h5')
#
model_low_liu1= keras.models.load_model( 'model_low_liu1.h5')
model_low_liu2= keras.models.load_model( 'model_low_liu2.h5')
model_low_liu3= keras.models.load_model(  'model_low_liu3.h5')
model_low_liu4= keras.models.load_model( 'model_low_liu4.h5')
model_low_liu5= keras.models.load_model( 'model_low_liu5.h5')
model_low_liu6= keras.models.load_model( 'model_low_liu6.h5')
cnn_low= keras.models.load_model( 'cnn_low.h5')
#


read_temp=scipy.io.loadmat('data2_test.mat')
K=2
k=1

S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
theta=read_temp['theta']
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']
S_label1 = np.expand_dims(S_label, 2)
from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est)
[r2,c]=np.shape(R_est)
[r2,I]=np.shape(S_label)
DOA=np.arange(I)-60

Lu=np.zeros((I,r2))
DCN=np.zeros((I,r2))


test_auto=np.zeros((K,r2))
test_RBM=np.zeros((K,r2))
test_liu=np.zeros((K,r2))
test_SBL=np.zeros((K,r2))
test_SBL_s=np.zeros((K,r2))
test_cnn=np.zeros((K,r2))
test_cnn_abs=np.zeros((K,r2))
for i in range(r2):
    T=R_est[i,:].reshape(1,c)
    Y_autocode_T =autoencoder.predict(T)
    Y_autocode_T[:,0*c:1*c]=normalizer.transform(Y_autocode_T[:,0*c:1*c])
    Y_autocode_T[:,1*c:2*c]=normalizer.transform(Y_autocode_T[:,1*c:2*c])
    Y_autocode_T[:,2*c:3*c]=normalizer.transform(Y_autocode_T[:,2*c:3*c])
    Y_autocode_T[:,3*c:4*c]=normalizer.transform(Y_autocode_T[:,3*c:4*c])
    Y_autocode_T[:,4*c:5*c]=normalizer.transform(Y_autocode_T[:,4*c:5*c])
    Y_autocode_T[:,5*c:6*c]=normalizer.transform(Y_autocode_T[:,5*c:6*c])  
    
    DF_T_liu1 = model_low_liu1.predict(Y_autocode_T[:,0*c:1*c]) 
    DF_T_liu2 = model_low_liu2.predict(Y_autocode_T[:,1*c:2*c]) 
    DF_T_liu3 = model_low_liu3.predict(Y_autocode_T[:,2*c:3*c]) 
    DF_T_liu4 = model_low_liu4.predict(Y_autocode_T[:,3*c:4*c]) 
    DF_T_liu5 = model_low_liu5.predict(Y_autocode_T[:,4*c:5*c]) 
    DF_T_liu6 = model_low_liu6.predict(Y_autocode_T[:,5*c:6*c]) 
    DF_T_liu = [DF_T_liu1,DF_T_liu2,DF_T_liu3
                      ,DF_T_liu4,DF_T_liu5,DF_T_liu6]
    DF_T_liu=np.array(DF_T_liu)
    DF_T_liu=np.reshape(DF_T_liu,I)
    Lu[:,i]= DF_T_liu

    
    
    DF_T_cnn=cnn_low.predict(S_est[i,:,:].reshape(1,I,2))
    DF_T_cnn=np.array(DF_T_cnn)    
    DF_T_cnn=np.reshape(DF_T_cnn,I)
    DCN[:,i]=DF_T_cnn
    
    

 





figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(gamma[k,:]),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
plt.title('SBL')
plt.show()

figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(gamma_R[k,:]/max(gamma_R[k,:])),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
plt.title('SBL_R')
plt.show()

figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(DCN[:,k]),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')

plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
plt.title('DCN')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,np.mean(Lu,1),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
plt.title('liu')
plt.show()



