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
           ind[1]=60
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



read_temp=scipy.io.loadmat('data2_snr.mat')
T_SBC_R=read_temp['T_SBC_R']
T_SBC=read_temp['T_SBC']
S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
theta=read_temp['theta']
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']
SNR=read_temp['SNR']

S_label1 = np.expand_dims(S_label, 2)
from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est[:,:,0])
[r2,c,S]=np.shape(R_est)
[r2,I,S]=np.shape(S_label)
K=2

test_liu_high=np.zeros((r2,1))
test_cnn_high=np.zeros((r2,1))

test_liu_low=np.zeros((r2,1))
test_cnn_low=np.zeros((r2,1))

test_liu1=np.zeros((r2,1))
test_cnn1=np.zeros((r2,1))
test_liu2=np.zeros((r2,1))
test_cnn2=np.zeros((r2,1))
test_liu3=np.zeros((r2,1))
test_cnn3=np.zeros((r2,1))

test_SBL=np.zeros((r2,1))
test_SBL_R=np.zeros((r2,1))

RMSE_liu_low=np.zeros((S,1))
RMSE_cnn_low=np.zeros((S,1))
RMSE_liu_high=np.zeros((S,1))
RMSE_cnn_high=np.zeros((S,1))

RMSE_liu1=np.zeros((S,1))
RMSE_cnn1=np.zeros((S,1))

RMSE_liu2=np.zeros((S,1))
RMSE_cnn2=np.zeros((S,1))
RMSE_liu3=np.zeros((S,1))
RMSE_cnn3=np.zeros((S,1))


RMSE_SBL=np.zeros((S,1))
RMSE_SBL_R=np.zeros((S,1))


T_low_cnn=np.zeros((r2,S))
T_low_liu=np.zeros((r2,S))
DOA=np.arange(I)-60


import time
for j in range(S):
    for i in range(r2):
        T=R_est[i,:,j].reshape(1,c)
        
        start_time = time.clock()
        Y_autocode_T =autoencoder.predict(T)
        Y_autocode_T[:,0*c:1*c]=normalizer.transform(Y_autocode_T[:,0*c:1*c])
        Y_autocode_T[:,1*c:2*c]=normalizer.transform(Y_autocode_T[:,1*c:2*c])
        Y_autocode_T[:,2*c:3*c]=normalizer.transform(Y_autocode_T[:,2*c:3*c])
        Y_autocode_T[:,3*c:4*c]=normalizer.transform(Y_autocode_T[:,3*c:4*c])
        Y_autocode_T[:,4*c:5*c]=normalizer.transform(Y_autocode_T[:,4*c:5*c])
        Y_autocode_T[:,5*c:6*c]=normalizer.transform(Y_autocode_T[:,5*c:6*c])  
        
        DF_T_low_liu1 = model_low_liu1.predict(Y_autocode_T[:,0*c:1*c]) 
        DF_T_low_liu2 = model_low_liu2.predict(Y_autocode_T[:,1*c:2*c]) 
        DF_T_low_liu3 = model_low_liu3.predict(Y_autocode_T[:,2*c:3*c]) 
        DF_T_low_liu4 = model_low_liu4.predict(Y_autocode_T[:,3*c:4*c]) 
        DF_T_low_liu5 = model_low_liu5.predict(Y_autocode_T[:,4*c:5*c]) 
        DF_T_low_liu6 = model_low_liu6.predict(Y_autocode_T[:,5*c:6*c])  
        DF_T_low_liu = [DF_T_low_liu1,DF_T_low_liu2,DF_T_low_liu3
                          ,DF_T_low_liu4,DF_T_low_liu5,DF_T_low_liu6]
        DF_T_low_liu=np.array(DF_T_low_liu)
        DF_T_low_liu=np.reshape(DF_T_low_liu,I)
        stop_time = time.clock()
        T_low_liu[i,j]=stop_time-start_time
        
        


        
        
        start_time = time.clock()        
        DF_T_cnn_low=cnn_low.predict(S_est[i,:,:,j].reshape(1,I,2))
        DF_T_cnn_low=np.array(DF_T_cnn_low)    
        DF_T_cnn_low=np.reshape(DF_T_cnn_low,I)
        stop_time = time.clock()
        T_low_cnn[i,j]=stop_time-start_time



        
        DOA_liu_low,_=findpeaks(DF_T_low_liu,K,DOA)
        DOA_cnn_low,_=findpeaks(DF_T_cnn_low,K,DOA)  


        DOA_SBL,_=findpeaks(gamma[i,:,j],K,DOA)
        DOA_SBL_R,_=findpeaks(gamma_R[i,:,j],K,DOA)
        
    

        

        test_SBL_R[i]=np.mean(np.square(np.sort(DOA_SBL_R)-DOA_train[:,i,j]))
        test_SBL[i]=np.mean(np.square(np.sort(DOA_SBL)-DOA_train[:,i,j]))
        
        test_liu_low[i]=np.mean(np.square(np.sort(DOA_liu_low)-DOA_train[:,i,j]))
        test_cnn_low[i]=np.mean(np.square(np.sort(DOA_cnn_low)-DOA_train[:,i,j]))
        

    
    
    RMSE_SBL_R[j]=np.sqrt(np.mean(test_SBL_R))
    RMSE_SBL[j]=np.sqrt(np.mean(test_SBL))
    RMSE_liu_low[j]=np.sqrt(np.mean(test_liu_low))
    RMSE_cnn_low[j]=np.sqrt(np.mean(test_cnn_low))

    print(j)    



figsize = 10,8
figure, ax = plt.subplots(figsize=figsize)

plt.tick_params(labelsize=13)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
plt.semilogy(SNR.T,RMSE_SBL_R-0.0*np.ones((S,1)))
plt.semilogy(SNR.T,RMSE_SBL-0.0*np.ones((S,1)))
plt.semilogy(SNR.T,RMSE_liu_low+0.0*np.ones((S,1)))
plt.semilogy(SNR.T,RMSE_cnn_low)

plt.legend(['SBL_R','SBL','Method in [5]','proposed'])
plt.xlabel('SNR(dB)',font2)
plt.ylabel('RMSE($^o$)',font2) #将文件保存至文件中并且画出图
plt.ylim([0.01,100])
plt.xlim([-15,15])
plt.show()




