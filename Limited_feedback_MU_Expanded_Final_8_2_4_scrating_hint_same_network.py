
# library import
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tenforflow as tf

import random
import numpy as np
from scipy import io
import os, sys
from tensorflow.python.framework import ops
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import h5py

# Load test data
with h5py.File('./channel_tmp_data_MU_complex_8_2_4_8.mat', 'r') as f:
    vec_H_ch = f['vec_H_ch'][()]
    vec_H_ch = vec_H_ch.transpose()

    y_train = f['y_train'][()]
    y_train = y_train.transpose()

    n_f = f['n_f'][()]
    n_f = n_f.transpose()

    Pilot_mtx_2 = f['Pilot_mtx_2'][()]
    Pilot_mtx_2 = Pilot_mtx_2.transpose()

    Noise = f['Noise'][()]
    Noise = Noise.transpose()

# Load validation data
with h5py.File('./channel_val_data_MU_complex_8_2_4_8.mat', 'r') as f:
    valid_vec_H_ch = f['vec_H_ch'][()]
    valid_vec_H_ch = valid_vec_H_ch.transpose()

    valid_y_train = f['y_train'][()]
    valid_y_train = valid_y_train.transpose()


# Parameters setting

training_iters = 50000
batch_size = 1000
display_step = 1000
test_step = 1000
learning_rate = 0.0002
decay_steps = 100000
decay_rate = 0.95
l2_regul = 0.001
test_num = 100000

# Network Parameters
n_tx = 8
n_rx = 2
n_user = int(n_tx / n_rx)
Bit = int(24 / n_user)
complex_size = 2
dim_feature = n_tx * n_rx * complex_size
L_ = 8
dim_feature_L_ = L_ * n_rx * complex_size

Pilot_mtx = Pilot_mtx_2[:, :, 0] + 1j * Pilot_mtx_2[:, :, 1]
Noise = Noise[:, :, :, :, 0] + 1j * Noise[:, :, :, :, 1]

with tf.name_scope("input") as scope:
    # tf Graph input
    x = tf.placeholder(tf.complex64, [None, L_, n_rx, n_user])
    x_ch = tf.placeholder(tf.complex64, [None, n_tx, n_rx, n_user])

    PdB_tensor = tf.placeholder(tf.complex64)
    lr = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)


# In[4]:
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)



# Receiver network
def Dnn_Encoder(x, weights, biases, train_flag):
    with tf.name_scope("full_layer1") as scope:
        fc1 = tf.add(tf.matmul(x, weights['wf1']), biases['bf1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("full_layer2") as scope:
        fc2 = tf.add(tf.matmul(fc1, weights['wf2']), biases['bf2'])
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("full_layer3") as scope:
        fc3 = tf.add(tf.matmul(fc2, weights['wf3']), biases['bf3'])
        fc3 = tf.nn.relu(fc3)

    with tf.name_scope("full_layer4") as scope:
        fc4_ = tf.add(tf.matmul(fc3, weights['wf4']), biases['bf4'])
        fc4 = tf.tanh(fc4_)
        fc4_binarized = binarize(fc4)
    return fc4_binarized, fc4_


# Transmitter network connected with quantization layers
def Dnn_Decoder_low(x, weights, biases, train_flag):

    with tf.name_scope("full_layer1") as scope:
        fc1 = tf.add(tf.matmul(x, weights['wf1']), biases['bf1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("full_layer2") as scope:
        fc2 = tf.add(tf.matmul(fc1, weights['wf2']), biases['bf2'])
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("full_layer3") as scope:
        fc3 = tf.add(tf.matmul(fc2, weights['wf3']), biases['bf3'])
        fc3 = tf.nn.relu(fc3)

    with tf.name_scope("full_layer4") as scope:
        fc4 = tf.add(tf.matmul(fc3, weights['wf4']), biases['bf4'])

        # Normalization
        for i in range(n_tx):
            fc4_temp = fc4[:, n_tx * complex_size * i:n_tx * complex_size * (i + 1)]

            fc4_complex = tf.expand_dims(tf.complex(fc4_temp[:, 0:n_tx], fc4_temp[:, n_tx:]), axis=2)
            fc4_norm = tf.expand_dims(tf.norm(fc4_complex, ord=2, axis=1), axis=2)


            if i == 0:
                fc4_final = fc4_complex / fc4_norm
            else:
                fc4_final = tf.concat([fc4_final, fc4_complex / fc4_norm], 2)

            print(fc4_final)

    return fc4_final

# Transmitter network connected with layer right before quantization
def Dnn_Decoder_full(x, weights, biases, train_flag):
    with tf.name_scope("full_layer1") as scope:
        fc1 = tf.add(tf.matmul(x, weights['full_wf1']), biases['full_bf1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("full_layer2") as scope:
        fc2 = tf.add(tf.matmul(fc1, weights['full_wf2']), biases['full_bf2'])
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("full_layer3") as scope:
        fc3 = tf.add(tf.matmul(fc2, weights['full_wf3']), biases['full_bf3'])
        fc3 = tf.nn.relu(fc3)

    with tf.name_scope("full_layer4") as scope:
        fc4 = tf.add(tf.matmul(fc3, weights['full_wf4']), biases['full_bf4'])

        # Normalization
        for i in range(n_tx):
            fc4_temp = fc4[:, n_tx * complex_size * i:n_tx * complex_size * (i + 1)]
            fc4_complex = tf.expand_dims(tf.complex(fc4_temp[:, 0:n_tx], fc4_temp[:, n_tx:]), axis=2)
            fc4_norm = tf.expand_dims(tf.norm(fc4_complex, ord=2, axis=1), axis=2)


            if i == 0:
                fc4_final = fc4_complex / fc4_norm
            else:
                fc4_final = tf.concat([fc4_final, fc4_complex / fc4_norm], 2)

            print(fc4_final)

    return fc4_final


# Set network layer size
num_filter_1 = 2 * 20 * n_tx * n_rx
num_filter_2 = 2 * 15 * n_tx * n_rx
num_filter_3 = 2 * 10 * n_tx * n_rx

# Store layers weight & bias
weights_encoder1 = {
    'wf1': tf.Variable(tf.random_normal([dim_feature_L_, num_filter_1], mean=0.0, stddev=np.sqrt(2 / (dim_feature_L_))),
                       name='wf1'),
    'wf2': tf.Variable(tf.random_normal([num_filter_1, num_filter_2], mean=0.0, stddev=np.sqrt(2 / (num_filter_1))),
                       name='wf2'),
    'wf3': tf.Variable(tf.random_normal([num_filter_2, num_filter_3], mean=0.0, stddev=np.sqrt(2 / (num_filter_2))),
                       name='wf3'),
    'wf4': tf.Variable(tf.random_normal([num_filter_3, Bit], mean=0.0, stddev=np.sqrt(2 / (num_filter_3))), name='wf4'),
}

biases_encoder1 = {
    'bf1': tf.Variable(tf.zeros([num_filter_1]), name='bf1'),
    'bf2': tf.Variable(tf.zeros([num_filter_2]), name='bf2'),
    'bf3': tf.Variable(tf.zeros([num_filter_3]), name='bf3'),
    'bf4': tf.Variable(tf.zeros([Bit]), name='bf4'),
}

# Set network layer size
num_filter_1 = 2 * 20 * n_tx * n_tx
num_filter_2 = 2 * 15 * n_tx * n_tx
num_filter_3 = 2 * 10 * n_tx * n_tx

weights_decoder1 = {
    'wf1': tf.Variable(
        tf.random_normal([int(Bit * n_user), num_filter_3], mean=0.0, stddev=np.sqrt(2 / (Bit * n_user))), name='wf1'),
    'wf2': tf.Variable(tf.random_normal([num_filter_3, num_filter_2], mean=0.0, stddev=np.sqrt(2 / (num_filter_3))),
                       name='wf2'),
    'wf3': tf.Variable(tf.random_normal([num_filter_2, num_filter_1], mean=0.0, stddev=np.sqrt(2 / (num_filter_2))),
                       name='wf3'),
    'wf4': tf.Variable(
        tf.random_normal([num_filter_1, int(dim_feature * n_user)], mean=0.0, stddev=np.sqrt(2 / (num_filter_1))),
        name='wf4'),
}

biases_decoder1 = {
    'bf1': tf.Variable(tf.zeros([num_filter_3]), name='bf1'),
    'bf2': tf.Variable(tf.zeros([num_filter_2]), name='bf2'),
    'bf3': tf.Variable(tf.zeros([num_filter_1]), name='bf3'),
    'bf4': tf.Variable(tf.zeros([dim_feature * n_user]), name='bf4'),
}

weights_decoder2 = {
    'full_wf1': tf.Variable(
        tf.random_normal([int(Bit * n_user), num_filter_3], mean=0.0, stddev=np.sqrt(2 / (Bit * n_user))),
        name='full_wf1'),
    'full_wf2': tf.Variable(
        tf.random_normal([num_filter_3, num_filter_2], mean=0.0, stddev=np.sqrt(2 / (num_filter_3))), name='full_wf2'),
    'full_wf3': tf.Variable(
        tf.random_normal([num_filter_2, num_filter_1], mean=0.0, stddev=np.sqrt(2 / (num_filter_2))), name='full_wf3'),
    'full_wf4': tf.Variable(
        tf.random_normal([num_filter_1, int(dim_feature * n_user)], mean=0.0, stddev=np.sqrt(2 / (num_filter_1))),
        name='full_wf4'),
}

biases_decoder2 = {
    'full_bf1': tf.Variable(tf.zeros([num_filter_3]), name='full_bf1'),
    'full_bf2': tf.Variable(tf.zeros([num_filter_2]), name='full_bf2'),
    'full_bf3': tf.Variable(tf.zeros([num_filter_1]), name='full_bf3'),
    'full_bf4': tf.Variable(tf.zeros([dim_feature * n_user]), name='full_bf4'),
}

# Construct model
for u_ii in range(n_user):
    x_reshape = tf.reshape(x[:, :, :, u_ii], [-1, L_ * n_rx])
    x_user = tf.concat([tf.real(x_reshape), tf.imag(x_reshape)], axis=1)

    # Receiver DNN
    Out_encoder_low, Out_encoder_full = Dnn_Encoder(x_user, weights_encoder1, biases_encoder1, train_flag)

    if u_ii == 0:
        Out_encoder_low_cat = Out_encoder_low
        Out_encoder_full_cat = Out_encoder_full
    else:
        Out_encoder_low_cat = tf.concat([Out_encoder_low_cat, Out_encoder_low], 1)
        Out_encoder_full_cat = tf.concat([Out_encoder_full_cat, Out_encoder_full], 1)

# Transmitter DNN with quantization
Out_decoder_low = Dnn_Decoder_low(Out_encoder_low_cat, weights_decoder1, biases_decoder1, train_flag)

# Transmitter DNN without quantization (For KD)
Out_decoder_full = Dnn_Decoder_full(Out_encoder_full_cat, weights_decoder2, biases_decoder2, train_flag)

w_esti_low = tf.reshape(Out_decoder_low, [-1, n_tx, n_rx, n_user])
w_esti_full = tf.reshape(Out_decoder_full, [-1, n_tx, n_rx, n_user])

# Calculate loss function (-sum rate) without KD
H_complex = x_ch
w_complex = w_esti_low


snr_lin = 10 ** (PdB_tensor / 10)
objective_ft_low = 0
for ii_ in range(n_user):
    interference = 0
    for jj_ in range(n_user):

        H_transpose = tf.transpose(H_complex[:, :, :, ii_], perm=[0, 2, 1], conjugate=True)
        w_transpose = tf.transpose(w_complex[:, :, :, jj_], perm=[0, 2, 1], conjugate=True)

        mul_result = tf.matmul(tf.matmul(H_transpose, w_complex[:, :, :, jj_]),
                               tf.matmul(w_transpose, H_complex[:, :, :, ii_]))

        if ii_ != jj_:
            interference = interference + snr_lin / n_tx * mul_result
        else:
            sinal = snr_lin / n_tx * mul_result

    objective_ft_low = objective_ft_low + (1 / np.log(2)) * (tf.log(
        tf.real(tf.matrix_determinant(tf.eye(n_rx, dtype=tf.complex64) + interference + sinal))) - tf.log(
        tf.real(tf.matrix_determinant(tf.eye(n_rx, dtype=tf.complex64) + interference))))

cost_low = -tf.reduce_mean(objective_ft_low)

# Calculate loss function (-sum rate) with KD
H_complex = x_ch
w_complex = w_esti_full

snr_lin = 10 ** (PdB_tensor / 10)
objective_ft_full = 0
for ii_ in range(n_user):
    interference = 0
    for jj_ in range(n_user):

        H_transpose = tf.transpose(H_complex[:, :, :, ii_], perm=[0, 2, 1], conjugate=True)
        w_transpose = tf.transpose(w_complex[:, :, :, jj_], perm=[0, 2, 1], conjugate=True)

        mul_result = tf.matmul(tf.matmul(H_transpose, w_complex[:, :, :, jj_]),
                               tf.matmul(w_transpose, H_complex[:, :, :, ii_]))

        if ii_ != jj_:
            interference = interference + snr_lin / n_tx * mul_result
        else:
            sinal = snr_lin / n_tx * mul_result

    objective_ft_full = objective_ft_full + (1 / np.log(2)) * (tf.log(
        tf.real(tf.matrix_determinant(tf.eye(n_rx, dtype=tf.complex64) + interference + sinal))) - tf.log(
        tf.real(tf.matrix_determinant(tf.eye(n_rx, dtype=tf.complex64) + interference))))

cost_full = -tf.reduce_mean(objective_ft_full)


optimizer_low = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                       beta2=0.999, epsilon=1e-8).minimize(cost_low)
optimizer_full = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                        beta2=0.999, epsilon=1e-8).minimize(cost_full)


# Initializing the variables, Session begin
for pdB_set in range(30,-5,-10):
    best_loss = 0

    init = tf.global_variables_initializer()

    # Save path
    model_path = "./weight_" + str(n_tx) + "_" + str(n_rx) + "_" + str(n_user) + "_" + str(
        L_) + "_Channel_hint_same_network_" + str(Bit) + "bit_" + str(pdB_set) + "/1"
    saver = tf.train.Saver()
    print("Start_training: %s" % model_path)
    ## Launch the graph

    sess = tf.Session()

    # Initialize network
    sess.run(init)
    step = 0

    while step <= training_iters:

        # Generate training data
        batch_x = (1 / np.sqrt(2)) * (
                    np.random.normal(0, 1, [batch_size, n_tx, n_rx, n_user]) + 1j * np.random.normal(0, 1,
                                                                                                     [batch_size, n_tx,
                                                                                                      n_rx, n_user]))
        batch_x = np.reshape(batch_x, [batch_size, n_tx, n_rx, n_user])
        batch_x_trans = np.transpose(batch_x, (0, 2, 3, 1))
        batch_x_reshape = np.reshape(batch_x_trans, [batch_size * n_rx * n_user, n_tx], order="F")

        batch_Noise = (1 / np.sqrt(2)) * (
                    np.random.normal(0, 1, [batch_size * n_rx * n_user, L_]) + 1j * np.random.normal(0, 1, [
                batch_size * n_rx * n_user, L_]))


        y_train_c = np.sqrt(10) * np.matmul(batch_x_reshape, Pilot_mtx) + batch_Noise

        y_train_c = np.reshape(y_train_c, [batch_size, n_rx, n_user, L_], order="F")
        y_train_c = np.transpose(y_train_c, (0, 3, 1, 2))

        # Set power
        PdB = pdB_set
        decayed_learning_rate = learning_rate

        # Reduce learning rate
        if step <= 30000:
            decayed_learning_rate = learning_rate
        elif step <= 40000:
            decayed_learning_rate = learning_rate / 10
        elif step <= 50000:
            decayed_learning_rate = learning_rate / 100

        # Optimize netowrk alternatively
        sess.run(optimizer_low,
                 feed_dict={x: y_train_c, x_ch: batch_x, PdB_tensor: PdB, lr: decayed_learning_rate, train_flag: True,
                            keep_prob: 0.5})
        sess.run(optimizer_full, feed_dict={x: y_train_c, x_ch:batch_x, PdB_tensor:PdB, lr: decayed_learning_rate, train_flag: True,keep_prob :0.5})

        # Validation start
        if step % display_step == 0:
            loss_low_val = 0
            loss_full_val = 0
            PdB = pdB_set

            for i in range(int(test_num / batch_size)):
                batch_x = valid_vec_H_ch[batch_size * i:batch_size * (i + 1), :, :, :, 0] + 1j * valid_vec_H_ch[
                                                                                                 batch_size * i:batch_size * (
                                                                                                             i + 1), :,
                                                                                                 :, :, 1]
                batch_x = np.reshape(batch_x, [batch_size, n_tx, n_rx, n_user])


                batch_valid_y_train = valid_y_train[batch_size * i:batch_size * (i + 1), :, :, :,
                                      0] + 1j * valid_y_train[batch_size * i:batch_size * (i + 1), :, :, :, 1]

                inst_loss_low, inst_loss_full = sess.run([cost_low, cost_full],
                                                         feed_dict={x: batch_valid_y_train, x_ch: batch_x,
                                                                    PdB_tensor: PdB, train_flag: False, keep_prob: 1})
                loss_low_val = loss_low_val + inst_loss_low
                loss_full_val = loss_full_val + inst_loss_full
            print("Iter " + str(step) + ", Validation Loss low= " + "{:.6f}".format(
                loss_low_val / float(test_num / batch_size)))
            print("Iter " + str(step) + ", Validation Loss full= " + "{:.6f}".format(
                loss_full_val / float(test_num / batch_size)))

        # Test start
        if step % test_step == 0:
            loss_low = 0
            loss_full = 0
            PdB = pdB_set

            for i in range(int(test_num / batch_size)):
                batch_x = vec_H_ch[batch_size * i:batch_size * (i + 1), :, :, :, 0] + 1j * vec_H_ch[
                                                                                           batch_size * i:batch_size * (
                                                                                                       i + 1), :, :, :,
                                                                                           1]
                batch_x = np.reshape(batch_x, [batch_size, n_tx, n_rx, n_user])
                batch_y_train = y_train[batch_size * i:batch_size * (i + 1), :, :, :, 0] + 1j * y_train[
                                                                                                batch_size * i:batch_size * (
                                                                                                            i + 1), :,
                                                                                                :, :, 1]

                inst_loss_low, inst_loss_full, w_esti_low_, Out_encoder_low_cat_, Out_encoder_full_cat_ = sess.run(
                    [cost_low, cost_full, w_esti_low, Out_encoder_low_cat, Out_encoder_full_cat],
                    feed_dict={x: batch_y_train, x_ch: batch_x, PdB_tensor: PdB, train_flag: False, keep_prob: 1})
                loss_low = loss_low + inst_loss_low
                loss_full = loss_full + inst_loss_full

                if i == 0:
                    w_esti_final = w_esti_low_
                    H_test = batch_x
                    y_train_final = batch_y_train
                else:
                    w_esti_final = np.concatenate([w_esti_final, w_esti_low_], axis=0)
                    H_test = np.concatenate([H_test, batch_x], axis=0)
                    y_train_final = np.concatenate([y_train_final, batch_y_train], axis=0)
            if best_loss > loss_low_val / float(test_num / batch_size):
                best_loss = loss_low_val / float(test_num / batch_size)

                best_test_loss = loss_low / float(test_num / batch_size)
                w_esti_best = w_esti_final
                save_path = saver.save(sess, model_path)
                io.savemat("DNN_result_" + str(n_tx) + "_" + str(n_rx) + "_" + str(n_user) + "_" + str(
                    L_) + "_Channel_hint_same_network_" + str(Bit) + "bit_" + str(pdB_set) + "dB.mat",
                           mdict={'H_test': H_test, 'y_train_final': y_train_final, 'w_esti_best': w_esti_best,
                                  'loss': best_test_loss})

        # Display log
        if step % 5000 == 0:
            print("Iter " + str(step) + ", Test Loss low= " + "{:.6f}".format(
                loss_low / float(test_num / batch_size)))
            print("Iter " + str(step) + ", Test Loss full= " + "{:.6f}".format(
                loss_full / float(test_num / batch_size)))
            print("Iter " + str(step) + ", Best Test Loss= " + "{:.6f}".format(best_test_loss))


        step += 1

    print("Optimization Finished!")




