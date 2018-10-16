
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.linalg as la

from ensemble_model import *
from utils import *

# # array signal parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 10        # array sensor number
N = 400       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # spatial filter training parameters
doa_min = -60      # minimal DOA (degree)
doa_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6       # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1    # number of repeated sampling with random noise

noise_flag_sf = 1    # 0: noise-free; 1: noise-present
amp_or_phase = 0   # show filter amplitude or phase: 0-amplitude; 1-phase

# # autoencoder parameters
input_size_sf = M * (M-1)
hidden_size_sf = int(1/2 * input_size_sf)
output_size_sf = input_size_sf
batch_size_sf = 32
num_epoch_sf = 1000
learning_rate_sf = 0.001

# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1         # DOA step (degree) for generating different scenarios
K_ss = 2            # signal number
doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE   # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10    # number of repeated sampling with random noise

noise_flag_ss = 1    # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1    # inter-grid angle in spatial spectrum
NUM_GRID_SS = int((doa_max - doa_min + 0.5 * grid_ss) / grid_ss)   # spectrum grids
L = 2    # number of hidden layer
input_size_ss = M * (M-1)
hidden_size_ss = [int(2/3* input_size_ss), int(4/9* input_size_ss), int(1/3* input_size_ss)]
output_size_ss = int(NUM_GRID_SS / SF_NUM)
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 300

# # test data parameters
test_DOA = np.array([31.5, 41.5])
test_K = len(test_DOA)
test_SNR = np.array([10, 10])

# # retrain the networks or not
reconstruct_nn_flag = True
retrain_sf_flag = True
retrain_ss_flag = True

# # file path of neural network parameters
model_path_nn = 'initial_model_AI.npy'
model_path_sf = 'spatialfilter_model_AI.npy'
model_path_ss = 'spatialspectrum_model_AI.npy'

# # array imperfection parameters
mc_flag = True
ap_flag = False
pos_flag = False

rmse_path = 'arrayimperf'
if mc_flag == True:
    rmse_path += '_mc'
if ap_flag == True:
    rmse_path += '_ap'
if pos_flag == True:
    rmse_path += '_pos'
rmse_path += '.npy'

Rho = np.arange(11) * 0.1
num_epoch_test = 1000
RMSE = []
for rho in Rho:
    # mutual coupling matrix
    if mc_flag == True:
        mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
        MC_coef = mc_para ** np.array(np.arange(M))
        MC_mtx = la.toeplitz(MC_coef)
    else:
        MC_mtx = np.identity(M)
    # amplitude & phase error
    if ap_flag == True:
        amp_coef = rho * np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2])
        phase_coef = rho * np.array([0.0, -30, -30, -30, -30, -30, 30, 30, 30, 30])
        AP_coef = [(1+amp_coef[idx])*np.exp(1j*phase_coef[idx]/180*np.pi) for idx in range(M)]
        AP_mtx = np.diag(AP_coef)
    else:
        AP_mtx = np.identity(M)
    # sensor position error
    if pos_flag == True:
        pos_para_ = rho * np.array([0.0, -1, -1, -1, -1, -1, 1, 1, 1, 1]) * 0.2 * d
        pos_para = np.expand_dims(pos_para_, axis=-1)
    else:
        pos_para = np.zeros([M, 1])

    # # train multi-task autoencoder for spatial filtering
    if reconstruct_nn_flag == True:
        tf.reset_default_graph()
        enmod_0 = Ensemble_Model(input_size_sf=input_size_sf,
                               hidden_size_sf=hidden_size_sf,
                               output_size_sf=output_size_sf,
                               SF_NUM=SF_NUM,
                               learning_rate_sf=learning_rate_sf,
                               input_size_ss=input_size_ss,
                               hidden_size_ss=hidden_size_ss,
                               output_size_ss=output_size_ss,
                               learning_rate_ss=learning_rate_ss,
                               reconstruct_nn_flag=True,
                               train_sf_flag=True,
                               train_ss_flag=True,
                               model_path_nn=model_path_nn,
                               model_path_sf=model_path_sf,
                               model_path_ss=model_path_ss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            var_dict_nn = {}
            for var in tf.trainable_variables():
                value = sess.run(var)
                var_dict_nn[var.name] = value
            np.save(model_path_nn, var_dict_nn)

    # # train multi-task autoencoder for spatial filtering
    if retrain_sf_flag == True:
        # # generate spatial filter training dataset
        data_train_sf = generate_training_data_sf_AI(M, N, d, wavelength, SNR_sf, doa_min, NUM_REPEAT_SF, grid_sf, GRID_NUM_SF,
                                                  output_size_sf, SF_NUM, SF_SCOPE, MC_mtx, AP_mtx, pos_para)

        tf.reset_default_graph()
        enmod_1 = Ensemble_Model(input_size_sf=input_size_sf,
                               hidden_size_sf=hidden_size_sf,
                               output_size_sf=output_size_sf,
                               SF_NUM=SF_NUM,
                               learning_rate_sf=learning_rate_sf,
                               input_size_ss=input_size_ss,
                               hidden_size_ss=hidden_size_ss,
                               output_size_ss=output_size_ss,
                               learning_rate_ss=learning_rate_ss,
                               reconstruct_nn_flag=False,
                               train_sf_flag=True,
                               train_ss_flag=False,
                               model_path_nn=model_path_nn,
                               model_path_sf=model_path_sf,
                               model_path_ss=model_path_ss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epoch_sf):
                [data_batches, label_batches] = generate_spec_batches(data_train_sf, batch_size_sf, noise_flag_sf)
                for batch_idx in range(len(data_batches)):
                    data_batch = data_batches[batch_idx]
                    label_batch = label_batches[batch_idx]
                    feed_dict = {enmod_1.data_train_: data_batch, enmod_1.label_sf_: label_batch}
                    _, loss = sess.run([enmod_1.train_op_sf, enmod_1.loss_sf], feed_dict=feed_dict)

                    print('Epoch: {}, Batch: {}, loss: {:g}'.format(epoch, batch_idx, loss))

            var_dict_sf = {}
            for var in tf.trainable_variables():
                value = sess.run(var)
                var_dict_sf[var.name] = value
            np.save(model_path_sf, var_dict_sf)

    # # train DNN for spectrum estimation, with autoencoder parameters fixed
    if retrain_ss_flag == True:
        # # generate spatial spectrum training dataset
        data_train_ss = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta,
                                                  NUM_REPEAT_SS, grid_ss, NUM_GRID_SS, MC_mtx, AP_mtx, pos_para)

        tf.reset_default_graph()
        enmod_2 = Ensemble_Model(input_size_sf=input_size_sf,
                               hidden_size_sf=hidden_size_sf,
                               output_size_sf=output_size_sf,
                               SF_NUM=SF_NUM,
                               learning_rate_sf=learning_rate_sf,
                               input_size_ss=input_size_ss,
                               hidden_size_ss=hidden_size_ss,
                               output_size_ss=output_size_ss,
                               learning_rate_ss=learning_rate_ss,
                               reconstruct_nn_flag=False,
                               train_sf_flag=False,
                               train_ss_flag=True,
                               model_path_nn=model_path_nn,
                               model_path_sf=model_path_sf,
                               model_path_ss=model_path_ss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('spectrum estimating...')

            # train
            for epoch in range(num_epoch_ss):
                [data_batches, label_batches] = generate_spec_batches(data_train_ss, batch_size_ss, noise_flag_ss)
                for batch_idx in range(len(data_batches)):
                    data_batch = data_batches[batch_idx]
                    label_batch = label_batches[batch_idx]
                    feed_dict = {enmod_2.data_train_: data_batch, enmod_2.label_ss_: label_batch}
                    _, loss_ss = sess.run([enmod_2.train_op_ss, enmod_2.loss_ss], feed_dict=feed_dict)

                    print('Epoch: {}, Batch: {}, loss: {:g}'.format(epoch, batch_idx, loss_ss))

            var_dict_ss = {}
            for var in tf.trainable_variables():
                value = sess.run(var)
                var_dict_ss[var.name] = value
            np.save(model_path_ss, var_dict_ss)


    # # test
    tf.reset_default_graph()
    enmod_3 = Ensemble_Model(input_size_sf=input_size_sf,
                           hidden_size_sf=hidden_size_sf,
                           output_size_sf=output_size_sf,
                           SF_NUM=SF_NUM,
                           learning_rate_sf=learning_rate_sf,
                           input_size_ss=input_size_ss,
                           hidden_size_ss=hidden_size_ss,
                           output_size_ss=output_size_ss,
                           learning_rate_ss=learning_rate_ss,
                           reconstruct_nn_flag=False,
                           train_sf_flag=False,
                           train_ss_flag=False,
                           model_path_nn=model_path_nn,
                           model_path_sf=model_path_sf,
                           model_path_ss=model_path_ss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('testing...')

        # # test
        est_DOA = []
        MSE_rho = np.zeros([test_K, ])
        for epoch in range(num_epoch_test):
            test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx, AP_mtx, pos_para)
            data_batch = np.expand_dims(test_cov_vector, axis=-1)
            feed_dict = {enmod_3.data_train: data_batch}
            ss_output = sess.run(enmod_3.output_ss, feed_dict=feed_dict)
            ss_min = np.min(ss_output)
            ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]

            est_DOA_ii = get_DOA_estimate(ss_output, test_DOA, doa_min, grid_ss)
            est_DOA.append(est_DOA_ii)
            MSE_rho += np.square(est_DOA_ii - test_DOA)
        RMSE_rho = np.sqrt(MSE_rho / num_epoch_test)
        RMSE.append(RMSE_rho)

np.save(rmse_path, RMSE)

plt.figure()
for kk in range(test_K):
    RMSE_kk = [rmse[kk] for rmse in RMSE]
    plt.plot(Rho, RMSE_kk)
plt.show()