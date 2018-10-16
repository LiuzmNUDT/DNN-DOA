
import tensorflow as tf
import numpy as np

class Ensemble_Model():
    def __init__(self, input_size_sf, hidden_size_sf, output_size_sf, SF_NUM, learning_rate_sf, input_size_ss, hidden_size_ss,
                 output_size_ss, learning_rate_ss, reconstruct_nn_flag, train_sf_flag, train_ss_flag, model_path_nn, model_path_sf, model_path_ss):
        # # auto-encoder for spatial filtering
        # place holder
        self.data_train_ = tf.placeholder(tf.float32, shape=[None, input_size_sf])
        self.label_sf_ = tf.placeholder(tf.float32, shape=[None, output_size_sf * SF_NUM])
        self.data_train = tf.transpose(self.data_train_)
        self.label_sf = tf.transpose(self.label_sf_)

        # load nn parameter files
        if reconstruct_nn_flag == False:
            var_dict_nn = np.load(model_path_nn).item()
        if train_sf_flag == False:
            var_dict_sf = np.load(model_path_sf).item()
            if train_ss_flag == False:
                var_dict_ss = np.load(model_path_ss).item()

        # nn parameters
        if reconstruct_nn_flag == True:
            # encoder parameters
            self.W_ec = tf.Variable(initial_value=tf.random_uniform([hidden_size_sf, input_size_sf], minval=-0.1, maxval=0.1),
                                    trainable=True, name='W_ec')
            self.b_ec = tf.Variable(initial_value=tf.random_uniform([hidden_size_sf, 1], minval=-0.1, maxval=0.1),
                                    trainable=True, name='b_ec')

            # decoder parameters
            self.W_dc = tf.Variable(initial_value=tf.random_uniform([output_size_sf * SF_NUM, hidden_size_sf], minval=-0.1, maxval=0.1),
                                    trainable=True, name='W_dc')
            self.b_dc = tf.Variable(initial_value=tf.random_uniform([output_size_sf * SF_NUM, 1], minval=-0.1, maxval=0.1),
                                    trainable=True, name='b_dc')
        elif train_sf_flag == True:
            self.W_ec = tf.Variable(initial_value=var_dict_nn['W_ec:0'],
                              trainable=True, name='W_ec')
            self.b_ec = tf.Variable(initial_value=var_dict_nn['b_ec:0'],
                              trainable=True, name='b_ec')
            self.W_dc = tf.Variable(initial_value=var_dict_nn['W_dc:0'],
                               trainable=True, name='W_dc')
            self.b_dc = tf.Variable(initial_value=var_dict_nn['b_dc:0'],
                               trainable=True, name='b_dc')
        else:
            # load variable dictionary
            self.W_ec = tf.Variable(initial_value=var_dict_sf['W_ec:0'],
                                    trainable=False, name='W_ec')
            self.b_ec = tf.Variable(initial_value=var_dict_sf['b_ec:0'],
                                    trainable=False, name='b_ec')
            self.W_dc = tf.Variable(initial_value=var_dict_sf['W_dc:0'],
                                    trainable=False, name='W_dc')
            self.b_dc = tf.Variable(initial_value=var_dict_sf['b_dc:0'],
                                    trainable=False, name='b_dc')

        # output prediction
        self.h_sf = tf.matmul(self.W_ec, self.data_train) + self.b_ec
        self.output_pred_sf = tf.matmul(self.W_dc, self.h_sf) + self.b_dc

        # output target
        self.output_target_sf = self.label_sf

        # loss and train
        self.error_sf = self.output_target_sf - self.output_pred_sf
        self.loss_sf = tf.reduce_mean(tf.square(self.error_sf)) * (output_size_sf * SF_NUM)
        if train_sf_flag == True:
            self.train_op_sf = tf.train.RMSPropOptimizer(learning_rate=learning_rate_sf).minimize(self.loss_sf)



        # # spatial spectrum estimation
        self.label_ss_ = tf.placeholder(tf.float32, shape=[None, output_size_ss * SF_NUM])
        self.label_ss = tf.transpose(self.label_ss_)
        # get input from spatial filters
        self.sf_output_list = []
        for sf_idx in range(SF_NUM):
            sf_output_curr = self.output_pred_sf[sf_idx * output_size_sf : (sf_idx + 1) * output_size_sf]
            self.sf_output_list.append(sf_output_curr)

        # input-to-hidden, hidden-to-hidden, hidden-to-output parameters
        self.Whi_list = []
        self.bhi_list = []
        self.Whh_list = []
        self.bhh_list = []
        self.Woh_list = []
        self.boh_list = []
        if reconstruct_nn_flag == True:
            for sf_idx in range(SF_NUM):
                Whi_curr = tf.Variable(
                    initial_value=tf.random_uniform([hidden_size_ss[0], input_size_ss], minval=-0.1, maxval=0.1),
                    trainable=True, name='Whi_' + str(sf_idx))
                self.Whi_list.append(Whi_curr)
                bhi_curr = tf.Variable(
                    initial_value=tf.random_uniform([hidden_size_ss[0], 1], minval=-0.1, maxval=0.1),
                    trainable=True, name='bhi_' + str(sf_idx))
                self.bhi_list.append(bhi_curr)
                Whh_curr = tf.Variable(
                    initial_value=tf.random_uniform([hidden_size_ss[1], hidden_size_ss[0]], minval=-0.1, maxval=0.1),
                    trainable=True, name='Whh_' + str(sf_idx))
                self.Whh_list.append(Whh_curr)
                bhh_curr = tf.Variable(
                    initial_value=tf.random_uniform([hidden_size_ss[1], 1], minval=-0.1, maxval=0.1),
                    trainable=True, name='bhh_' + str(sf_idx))
                self.bhh_list.append(bhh_curr)
                Woh_curr = tf.Variable(
                    initial_value=tf.random_uniform([output_size_ss, hidden_size_ss[1]], minval=-0.1, maxval=0.1),
                    trainable=True, name='Woh_' + str(sf_idx))
                self.Woh_list.append(Woh_curr)
                boh_curr = tf.Variable(
                    initial_value=tf.random_uniform([output_size_ss, 1], minval=-0.1, maxval=0.1),
                    trainable=True, name='boh_' + str(sf_idx))
                self.boh_list.append(boh_curr)
                # self.Woo = tf.Variable(
                #     initial_value=tf.random_uniform([output_size_ss * SF_NUM, output_size_ss * SF_NUM], minval=-0.1, maxval=0.1),
                #     trainable=True, name='Woo')
                # self.boo = tf.Variable(
                #     initial_value=tf.random_uniform([output_size_ss * SF_NUM, 1], minval=-0.1,
                #                                     maxval=0.1),
                #     trainable=True, name='boo')
        elif (train_ss_flag == True) or (train_sf_flag == True and train_ss_flag == False):
            for sf_idx in range(SF_NUM):
                Whi_curr = tf.Variable(
                    initial_value=var_dict_nn['Whi_' + str(sf_idx) + ':0'],
                    trainable=True, name='Whi_' + str(sf_idx))
                self.Whi_list.append(Whi_curr)
                bhi_curr = tf.Variable(
                    initial_value=var_dict_nn['bhi_' + str(sf_idx) + ':0'],
                    trainable=True, name='bhi_' + str(sf_idx))
                self.bhi_list.append(bhi_curr)
                Whh_curr = tf.Variable(
                    initial_value=var_dict_nn['Whh_' + str(sf_idx) + ':0'],
                    trainable=True, name='Whh_' + str(sf_idx))
                self.Whh_list.append(Whh_curr)
                bhh_curr = tf.Variable(
                    initial_value=var_dict_nn['bhh_' + str(sf_idx) + ':0'],
                    trainable=True, name='bhh_' + str(sf_idx))
                self.bhh_list.append(bhh_curr)
                Woh_curr = tf.Variable(
                    initial_value=var_dict_nn['Woh_' + str(sf_idx) + ':0'],
                    trainable=True, name='Woh_' + str(sf_idx))
                self.Woh_list.append(Woh_curr)
                boh_curr = tf.Variable(
                    initial_value=var_dict_nn['boh_' + str(sf_idx) + ':0'],
                    trainable=True, name='boh_' + str(sf_idx))
                self.boh_list.append(boh_curr)
                # self.Woo = tf.Variable(
                #     initial_value=var_dict_nn['Woo:0'],
                #     trainable=True, name='Woo')
                # self.boo = tf.Variable(
                #     initial_value=var_dict_nn['boo:0'],
                #     trainable=True, name='boo')
        else:
            # load variable dictionary
            for sf_idx in range(SF_NUM):
                Whi_curr = tf.Variable(
                    initial_value=var_dict_ss['Whi_' + str(sf_idx) + ':0'],
                    trainable=False, name='Whi_' + str(sf_idx))
                self.Whi_list.append(Whi_curr)
                bhi_curr = tf.Variable(
                    initial_value=var_dict_ss['bhi_' + str(sf_idx) + ':0'],
                    trainable=False, name='bhi_' + str(sf_idx))
                self.bhi_list.append(bhi_curr)
                Whh_curr = tf.Variable(
                    initial_value=var_dict_ss['Whh_' + str(sf_idx) + ':0'],
                    trainable=False, name='Whh_' + str(sf_idx))
                self.Whh_list.append(Whh_curr)
                bhh_curr = tf.Variable(
                    initial_value=var_dict_ss['bhh_' + str(sf_idx) + ':0'],
                    trainable=False, name='bhh_' + str(sf_idx))
                self.bhh_list.append(bhh_curr)
                Woh_curr = tf.Variable(
                    initial_value=var_dict_ss['Woh_' + str(sf_idx) + ':0'],
                    trainable=False, name='Woh_' + str(sf_idx))
                self.Woh_list.append(Woh_curr)
                boh_curr = tf.Variable(
                    initial_value=var_dict_ss['boh_' + str(sf_idx) + ':0'],
                    trainable=False, name='boh_' + str(sf_idx))
                self.boh_list.append(boh_curr)
                # self.Woo = tf.Variable(
                #     initial_value=var_dict_ss['Woo:0'],
                #     trainable=False, name='Woo')
                # self.boo = tf.Variable(
                #     initial_value=var_dict_ss['boo:0'],
                #     trainable=False, name='boo')

        # feed-forward
        self.output_ss_ = []
        for sf_idx in range(SF_NUM):
            input_ss_curr = self.sf_output_list[sf_idx]
            Whi = self.Whi_list[sf_idx]
            bhi = self.bhi_list[sf_idx]
            Whh = self.Whh_list[sf_idx]
            bhh = self.bhh_list[sf_idx]
            Woh = self.Woh_list[sf_idx]
            boh = self.boh_list[sf_idx]
            h1_ = tf.matmul(Whi, input_ss_curr) + bhi
            h1 = tf.tanh(h1_)
            h2_ = tf.matmul(Whh, h1) + bhh
            h2 = tf.tanh(h2_)
            output_ss_curr = tf.matmul(Woh, h2) + boh
            self.output_ss_.append(output_ss_curr)
        self.output_ss_ = tf.concat(self.output_ss_, axis=0)
        # self.output_ss = tf.matmul(self.Woo, tf.tanh(self.output_ss_)) + self.boo
        self.output_ss = self.output_ss_

        # loss and optimizer
        self.error_ss = self.label_ss - self.output_ss
        self.loss_ss = tf.reduce_mean(tf.norm(tf.square(self.error_ss), ord=1))
        if train_ss_flag == True:
            self.train_op_ss = tf.train.RMSPropOptimizer(learning_rate=learning_rate_ss).minimize(self.loss_ss)
