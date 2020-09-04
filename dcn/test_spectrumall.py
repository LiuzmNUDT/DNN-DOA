# -*- coding: utf-8 -*-

import keras.models
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.signal
import heapq
def findpeaks(x, K,DOA):
    x=np.array(x)
    indexes,_= scipy.signal.find_peaks(x)    
    maxi=heapq.nlargest(K, x[indexes])
    ind=np.zeros((K,))
    ind=np.int_(ind)
    p=np.zeros((K,))

    if len(indexes)==0:
        ind[0]=60
        ind[1]=59
        ind=np.int_(ind)
        p=DOA[ind]

    else:   
        if len(indexes)<K:
           ind[0]=indexes
           ind[1]=indexes+1
           ind=np.int_(ind)
           p=DOA[ind]
        else:
             for i in range(K):
                ind[i]=np.where(x==maxi[i])[0][0]
                ind[i]=np.int_(ind[i])
                if ind[i]==0:
                   p[i]=DOA[ind[i]]    
                else:
                    l=int(ind[i]-1)
                    
                    r=int(ind[i]+1)
                    ind[i]=np.int_(ind[i])                   
                    if x[l]<x[r]:
                         p[i]=x[r]/(x[r]+x[ind[i]])*DOA[r]+x[ind[i]]/(x[r]+x[ind[i]])*DOA[ind[i]]
                    else:
                         p[i]=x[l]/(x[l]+x[ind[i]])*DOA[l]+x[ind[i]]/(x[l]+x[ind[i]])*DOA[ind[i]]
                         
            
     
    ind=np.int_(ind)                          
  
#    return DOA[ind],p
    return p,DOA[ind]


autoencoder= keras.models.load_model( 'autoencoder.h5')

model_low_liu1= keras.models.load_model( 'model_low_liu1.h5')
model_low_liu2= keras.models.load_model( 'model_low_liu2.h5')
model_low_liu3= keras.models.load_model(  'model_low_liu3.h5')
model_low_liu4= keras.models.load_model( 'model_low_liu4.h5')
model_low_liu5= keras.models.load_model( 'model_low_liu5.h5')
model_low_liu6= keras.models.load_model( 'model_low_liu6.h5')

cnn_low= keras.models.load_model( 'cnn_low.h5')

read_temp=scipy.io.loadmat('data2_testall.mat')
S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
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
DCN_abs=np.zeros((I,r2))
auto=np.zeros((I,r2))
ARBM=np.zeros((I,r2))
K=2
test_auto=np.zeros((K,r2))
test_RBM=np.zeros((K,r2))
test_liu=np.zeros((K,r2))
test_SBL=np.zeros((K,r2))
test_SBL_R=np.zeros((K,r2))
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
    

    
    ind_liu,_=findpeaks(DF_T_liu,K,DOA)
    ind_cnn,_=findpeaks(DF_T_cnn,K,DOA)    
    ind_SBL_R,_=findpeaks(gamma_R[i,:],K,DOA)
    ind_SBL,_=findpeaks(gamma[i,:],K,DOA)


    DOA_SBL_R=ind_SBL_R
    DOA_SBL=ind_SBL
    DOA_cnn=ind_cnn
    DOA_liu=ind_liu
    

    test_liu[:,i]=np.sort(DOA_liu)
    test_SBL[:,i]=np.sort(DOA_SBL)
    test_SBL_R[:,i]=np.sort(DOA_SBL_R)
    test_cnn[:,i]=np.sort(DOA_cnn)


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_liu.T,'.')
plt.plot(DOA_train.T)
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2) #将文件保存至文件中并且画出图

plt.title('liu')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_cnn.T,'.')
plt.plot(DOA_train.T)
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2)
plt.title('DCN')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_SBL.T,'.')
plt.plot(DOA_train.T)
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('SBL')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_SBL_R.T,'.')
plt.plot(DOA_train.T)
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('SBL-R')
plt.show()



figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_liu.T-DOA_train.T,'.')
plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
plt.ylim([-20,20])

[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('err-liu')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_cnn.T-DOA_train.T,'.')
plt.tick_params(labelsize=13)
plt.ylim([-20,20])

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2)
plt.title('err-DCN')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_SBL.T-DOA_train.T,'.')
plt.tick_params(labelsize=13)
plt.ylim([-20,20])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('err-SBL')
plt.show()


figsize = 5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_SBL_R.T-DOA_train.T,'.')
plt.tick_params(labelsize=13)
plt.ylim([-20,20])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('err-SBLR')
plt.show()

