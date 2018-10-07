import os
import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import layers
import tensorboard
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
import matplotlib.pyplot as plt
from lib.model_io import save_variables
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
np.set_printoptions(threshold=np.nan)

Patient = "7"
file_prefix = "C:\\Users\\Aris\\Desktop\\Epilepsy\\data\\sp\\Patient_"+Patient+"\\"
path, dirs, files = next(os.walk(file_prefix))
dataset_length = len(files)

# We know that images have 128 pixels in each dimension.
spectro_size = 128

# Images are stored in one-dimensional arrays of this length.
spectro_size_flat = spectro_size * spectro_size

##__________________________Read Input__________________________
data_images = []
labels = []

for num_data in range(dataset_length):

    with open(str(file_prefix + files[num_data]), 'rb') as fid:
        data = np.fromfile(fid, dtype=np.float32, count=-1)
        data = data.reshape((int(len(data) / spectro_size_flat), spectro_size, spectro_size))
        data = np.transpose(data, [2, 1, 0])
        data_images.append(data)
        del data
    if (files[num_data][11] == 'n'):
        labels.append([0, 0, 1])
    else:
        if(num_data<15):
            labels.append([1, 0, 0])
        else:
            labels.append([0, 1, 0])
    fid.close()

data_images = np.array(data_images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

# The number of channels of the spectrogram image.
channels = int(np.shape(data_images)[3])

spectro_image_flat = spectro_size_flat * channels

# Tuple with height and width of images used to reshape arrays.
channel_spectro_shape = (spectro_size, spectro_size)
specto_shape = (spectro_size, spectro_size, channels)

# Shuffling the training dataset at random.
random_suffle = np.random.choice(dataset_length, dataset_length, replace=False)
data_images = data_images[random_suffle]
data_images = np.log10(data_images)
labels = labels[random_suffle]

early_epileptic_seizures = 0
late_epileptic_seizures = 0
normal_brain_activity = 0
for i in range(dataset_length):
    if(np.argmax(labels[i])==0):
        early_epileptic_seizures += 1
    elif(np.argmax(labels[i])==1):
        late_epileptic_seizures += 1
    else:
        normal_brain_activity += 1

print('Number of early epileptic seizures in the data: ', early_epileptic_seizures)
print('Number of late epileptic seizures in the data: ', late_epileptic_seizures)
print('Number of normal brain activity in the data: ', normal_brain_activity)
print('Data Was Successfully Loaded')
print('')

##______________________Create the training, validation and test set____________________
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in stratSplit.split(data_images, labels):
    x_train, X_test = data_images[train_idx], data_images[test_idx]
    y_train, Y_test = labels[train_idx], labels[test_idx]
del data_images,labels
for train_idx, test_idx in stratSplit.split(x_train, y_train):
    X_train, X_val = x_train[train_idx], x_train[test_idx]
    Y_train, Y_val = y_train[train_idx], y_train[test_idx]
del x_train, y_train

training_set_size = np.shape(Y_train)[0]
validation_set_size = np.shape(Y_val)[0]
test_set_size = np.shape(Y_test)[0]

true = np.argmax(Y_val, axis=1)
true_V_E = (true/np.maximum(true, 1)).astype(np.int32)
true_V_S = np.maximum(true-1, 0).astype(np.int32)
true = np.argmax(Y_test, axis=1)
true_T_E = (true/np.maximum(true, 1)).astype(np.int32)
true_T_S = np.maximum(true-1, 0).astype(np.int32)
del true

##_________________________CNN Implementation___________________
weight_decay_coef = 0.00001
with tf.device('/gpu:0'):

    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, 128, 128, channels], name='x')
        Y = tf.placeholder(tf.float32, shape=[None, 3], name='y')
        prior_prob = tf.constant([[early_epileptic_seizures/dataset_length, \
                                   late_epileptic_seizures/dataset_length,\
                                   normal_brain_activity/dataset_length]], \
                                 dtype=tf.float32, name='Prior_Class_Probabilities')
        prob = tf.placeholder_with_default(1.0, shape=(), name='hold_probability')
        phase = tf.placeholder_with_default(False, shape=(), name='training_phase')
        lr = tf.placeholder(tf.float32, name='Learning_Rate')

    with tf.name_scope('Convolution_Weights'):
        # Variables to be learned by the model
        CW1 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, channels, 72], dtype=tf.float32, name='conv_weights_1')
        CW2 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 72, 100], dtype=tf.float32, name='conv_weights_2')
        CW3 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 100, 128], dtype=tf.float32, name='conv_weights_3')
        CW4 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 128, 150], dtype=tf.float32, name='conv_weights_4')
        CW5 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 150, 220], dtype=tf.float32, name='conv_weights_5')
        CW6 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 220, 256], dtype=tf.float32, name='conv_weights_6')

    with tf.name_scope('Dense_Layer_Weights'):
        W1 = tf.get_variable(initializer=xavier_initializer(), shape=[2 * 2 * 256, 256], dtype=tf.float32, name='weights_1')
        W2 = tf.get_variable(initializer=xavier_initializer(), shape=[256, 512], dtype=tf.float32, name='weights_2')
        W3 = tf.get_variable(initializer=xavier_initializer(), shape=[512, 3], dtype=tf.float32, name='weights_3')
        b = tf.get_variable(initializer=xavier_initializer(), shape=[3], dtype=tf.float32, name='biases')

    with tf.name_scope('Network'):
        # Convolutional Layer 1.
        conv1 = tf.nn.conv2d(input=X, filter=CW1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
        b_conv1 = layers.batch_normalization(inputs=conv1, training=phase, name="b_conv1")
        a1 = tf.nn.elu(b_conv1, name="a1")
        pool1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")

        # Convolutional Layer 2.
        conv2 = tf.nn.conv2d(input=pool1, filter=CW2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        b_conv2 = layers.batch_normalization(inputs=conv2, training=phase, name="b_conv2")
        a2 = tf.nn.elu(b_conv2, name="a2")
        pool2 = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
        dropout1 = tf.nn.dropout(pool2, keep_prob=prob, name="dropout1")

        # Convolutional Layer 3.
        conv3 = tf.nn.conv2d(input=dropout1, filter=CW3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        b_conv3 = layers.batch_normalization(inputs=conv3, training=phase, name="b_conv3")
        a3 = tf.nn.elu(b_conv3, name="a3")
        pool3 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
        dropout2 = tf.nn.dropout(pool3, keep_prob=prob, name="dropout2")

        # Convolutional Layer 4.
        conv4 = tf.nn.conv2d(input=dropout2, filter=CW4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        b_conv4 = layers.batch_normalization(inputs=conv4, training=phase, name="b_conv4")
        a4 = tf.nn.elu(b_conv4, name="a4")
        pool4 = tf.nn.max_pool(a4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool4")
        dropout3 = tf.nn.dropout(pool4, keep_prob=prob, name="dropout3")

        # Convolutional Layer 5.
        conv5 = tf.nn.conv2d(input=dropout3, filter=CW5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        b_conv5 = layers.batch_normalization(inputs=conv5, training=phase, name="b_conv5")
        a5 = tf.nn.elu(b_conv5, name="a5")
        pool5 = tf.nn.max_pool(a5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool5")
        dropout4 = tf.nn.dropout(pool5, keep_prob=prob, name="dropout4")

        # Convolutional Layer 6.
        conv6 = tf.nn.conv2d(input=dropout4, filter=CW6, strides=[1, 1, 1, 1], padding="SAME", name="conv6")
        b_conv6 = layers.batch_normalization(inputs=conv6, training=phase, name="b_conv6")
        a6 = tf.nn.elu(b_conv6, name="a6")
        pool6 = tf.nn.max_pool(a6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool6")
        dropout5 = tf.nn.dropout(pool6, keep_prob=prob, name="dropout5")

        # Hidden layer 1.
        layer1 = tf.matmul(tf.reshape(dropout5, [-1, 2 * 2 * 256]), W1, name="layer1")
        b_layer1 = layers.batch_normalization(layer1, training=phase, name="b_layer1")
        a7 = tf.nn.elu(b_layer1, name="a7")
        dropout6 = tf.nn.dropout(a7, keep_prob=prob, name="dropout6")

        # Hidden layer 2.
        layer2 = tf.matmul(dropout6, W2, name="layer2")
        b_layer2 = layers.batch_normalization(layer2, training=phase, name="b_layer2")
        a8 = tf.nn.elu(b_layer2, name="a8")
        dropout7 = tf.nn.dropout(a8, keep_prob=prob, name="dropout7")

        # Output layer.
        logits = tf.add(tf.matmul(dropout7, W3), b, name="logits")

        #### Predictions ####
        pred = tf.nn.softmax(logits, name='pred')
        pred_cls = tf.argmax(pred, axis=1, name='pred_cls')

        EE_pred = tf.cast(pred_cls/tf.maximum(pred_cls, 1), dtype=tf.int32)
        ES_pred = tf.cast(tf.maximum(pred_cls-1,0), dtype=tf.int32)

    with tf.name_scope('loss'):
        # Loss function
        regularization = weight_decay_coef * ((tf.norm(W1, ord=2)**2/2)\
                                             +(tf.norm(W2, ord=2)**2/2)\
                                             +(tf.norm(W3, ord=2)**2/2))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        loss = tf.add(tf.reduce_mean(cross_entropy), regularization, name='loss')

    # Optimization
    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            train = optimizer.minimize(loss)

    # Performance Measures
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(pred_cls, tf.argmax(Y, axis=1))
        corrects = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("loss", loss)

    # merge all summaries into a single "operation" which we can execute in a session
    merged = tf.summary.merge_all()

##____________________________________Training the Network_________________________________

batch_sz = 64

epoch = 0
min_my_metric = sys.float_info.max
max_epochs = 1000
n_early_stop_epochs = 50
early_stop_counter = 0

decay_factor = 1.0005
lr0 = 0.5
l_min = 1e-6
learning_rate = lr0
data_fitted = 0

accuracy = 0
best_epoch = 0
best_f1_score = 0
best_accuracy = 0

# Gathering stats
stat_train_loss = []
stat_valid_loss = []
stat_valid_tp_E = []
stat_valid_fp_E = []
stat_valid_fn_E = []
stat_valid_f1_E_score = []
stat_valid_tp_S = []
stat_valid_fp_S = []
stat_valid_fn_S = []
stat_valid_f1_S_score = []
stat_valid_f1_score = []
stat_valid_AUC_E_score = []
stat_valid_AUC_S_score = []
stat_valid_AUC_score = []
stat_test_tp_E = []
stat_test_fp_E = []
stat_test_fn_E = []
stat_test_f1_E_score = []
stat_test_tp_S = []
stat_test_fp_S = []
stat_test_fn_S = []
stat_test_f1_S_score = []
stat_test_f1_score = []
stat_test_AUC_E_score = []
stat_test_AUC_S_score = []
stat_test_AUC_score = []

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:

    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)

    sess.run(tf.global_variables_initializer())
    print("Variables initilialized.")
    print("Starting training.")
    print("")

    while (epoch < max_epochs):
        epoch += 1

        # Shuffling the training dataset at random.
        random_suffle = np.random.choice(training_set_size, training_set_size, replace=False)
        X_train = X_train[random_suffle]
        Y_train = Y_train[random_suffle]

        #_______________________Train Epoch_____________________
        pointer = 0
        total_batches = 0
        train_loss = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > training_set_size):
                batch = training_set_size % batch_sz
            else:
                batch = batch_sz

            # Training on batch.
            mean_loss, _ = sess.run([loss, train], feed_dict={X: X_train[pointer:pointer + batch], \
                                                              Y: Y_train[pointer:pointer + batch], \
                                                              lr: learning_rate, prob: 0.5, phase: True })

            # Gathering info.
            if math.isnan(mean_loss):
                print('train cost is NaN1')
                break
            train_loss += mean_loss
            total_batches += 1
            pointer += batch
            data_fitted += batch

            if(learning_rate>l_min):
                if(batch==batch_sz):
                    learning_rate = lr0 * (1 / decay_factor) ** (data_fitted / batch_sz)
                    if (learning_rate < l_min):
                        learning_rate = l_min

            # Epoch termination condition.
            if (pointer == training_set_size):
                break

        if (total_batches > 0):
            train_loss /= total_batches
        stat_train_loss.append(train_loss)

        # _______________________Validation Epoch_____________________
        pointer = 0
        total_batches = 0
        valid_loss = 0
        tp_E = 0
        fp_E = 0
        fn_E = 0
        tp_S = 0
        fp_S = 0
        fn_S = 0
        F1_E = 0
        F1_S = 0
        V_F1 = 0
        AUC_E = 0
        AUC_S = 0
        V_AUC = 0
        V_EE = np.zeros((validation_set_size,)).astype(np.int32)
        V_ES = np.zeros((validation_set_size,)).astype(np.int32)
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > validation_set_size):
                batch = validation_set_size % batch_sz
            else:
                batch = batch_sz

            # Validation on batch.
            mean_loss = sess.run(loss, feed_dict={X: X_val[pointer:pointer + batch], \
                                                  Y: Y_val[pointer:pointer + batch]})
            # Getting the predictions for calculating F1 and AUC scores.
            V_EE[pointer:pointer + batch], \
            V_ES[pointer:pointer + batch] = sess.run([EE_pred, ES_pred],
                                                     feed_dict={X: X_val[pointer:pointer + batch]})
            # Gathering info.
            if math.isnan(mean_loss):
                print('train cost is NaN3')
                break
            valid_loss += mean_loss
            total_batches += 1
            pointer += batch

            # Epoch termination condition.
            if (pointer == validation_set_size):
                break
        # Calculate mean loss on the validation set
        if (total_batches > 0):
            valid_loss /= total_batches
        stat_valid_loss.append(valid_loss)

        # Calculate F1 and AUC scores on the validation set
        for i in range(validation_set_size):
            if true_V_E[i]==0:
                if V_EE[i]==0:
                    tp_E += 1
                else:
                    fn_E += 1
            else:
                if V_EE[i]==0:
                    fp_E += 1
        if ((2 * tp_E + fp_E + fn_E) > 0):
            F1_E = (2 * tp_E) / (2 * tp_E + fp_E + fn_E)
        for i in range(validation_set_size):
            if true_V_S[i]==0:
                if V_ES[i]==0:
                    tp_S += 1
                else:
                    fn_S += 1
            else:
                if V_ES[i]==0:
                    fp_S += 1
        if ((2 * tp_S + fp_S + fn_S) > 0):
            F1_S = (2 * tp_S) / (2 * tp_S + fp_E + fn_S)
        V_F1 = 0.5 * (F1_E + F1_S)
        AUC_E = roc_auc_score(true_V_E, V_EE)
        AUC_S = roc_auc_score(true_V_S, V_ES)
        V_AUC = 0.5 * (AUC_E + AUC_S)

        # Appending the stats to lists to write them to the csv file.
        stat_valid_tp_E.append(tp_E)
        stat_valid_fp_E.append(fp_E)
        stat_valid_fn_E.append(fn_E)
        stat_valid_f1_E_score.append(F1_E)
        stat_valid_tp_S.append(tp_S)
        stat_valid_fp_S.append(fp_S)
        stat_valid_fn_S.append(fn_S)
        stat_valid_f1_S_score.append(F1_S)
        stat_valid_f1_score.append(V_F1)
        stat_valid_AUC_E_score.append(AUC_E)
        stat_valid_AUC_S_score.append(AUC_S)
        stat_valid_AUC_score.append(V_AUC)

        # _______________________Accuracy calculation_____________________
        pointer = 0
        tp_E = 0
        fp_E = 0
        fn_E = 0
        tp_S = 0
        fp_S = 0
        fn_S = 0
        F1_E = 0
        F1_S = 0
        T_F1 = 0
        AUC_E = 0
        AUC_S = 0
        T_AUC = 0
        T_EE = np.zeros((test_set_size,)).astype(np.int32)
        T_ES = np.zeros((test_set_size,)).astype(np.int32)
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > test_set_size):
                batch = test_set_size % batch_sz
            else:
                batch = batch_sz

            # Getting the predictions for calculating F1 and AUC scores.
            T_EE[pointer:pointer + batch], \
            T_ES[pointer:pointer + batch] = sess.run([EE_pred, ES_pred],
                                                     feed_dict={X: X_test[pointer:pointer + batch]})
            pointer += batch

            # Epoch termination condition.
            if (pointer == test_set_size):
                break

        # Calculate F1 and AUC scores on the validation set
        for i in range(test_set_size):
            if true_T_E[i] == 0:
                if T_EE[i] == 0:
                    tp_E += 1
                else:
                    fn_E += 1
            else:
                if T_EE[i] == 0:
                    fp_E += 1
        if ((2 * tp_E + fp_E + fn_E) > 0):
            F1_E = (2 * tp_E) / (2 * tp_E + fp_E + fn_E)
        for i in range(test_set_size):
            if true_T_S[i] == 0:
                if T_ES[i] == 0:
                    tp_S += 1
                else:
                    fn_S += 1
            else:
                if T_ES[i] == 0:
                    fp_S += 1
        if ((2 * tp_S + fp_S + fn_S) > 0):
            F1_S = (2 * tp_S) / (2 * tp_S + fp_E + fn_S)
        T_F1 = 0.5 * (F1_E + F1_S)
        AUC_E = roc_auc_score(true_T_E, T_EE)
        AUC_S = roc_auc_score(true_T_S, T_ES)
        T_AUC = 0.5 * (AUC_E + AUC_S)

        # Appending the stats to lists to write them to the csv file.
        stat_test_tp_E.append(tp_E)
        stat_test_fp_E.append(fp_E)
        stat_test_fn_E.append(fn_E)
        stat_test_f1_E_score.append(F1_E)
        stat_test_tp_S.append(tp_S)
        stat_test_fp_S.append(fp_S)
        stat_test_fn_S.append(fn_S)
        stat_test_f1_S_score.append(F1_S)
        stat_test_f1_score.append(T_F1)
        stat_test_AUC_E_score.append(AUC_E)
        stat_test_AUC_S_score.append(AUC_S)
        stat_test_AUC_score.append(T_AUC)

        if (epoch-1 != len(stat_train_loss)):
            file = "C:\\Users\\Aris\\Desktop\\Epilepsy\\logs\\graphing\\Patient_"+Patient+"\\log_"+str('log10_')+str('priors')+".csv"
            log = open(file, 'a')
            for i in range(epoch-1, len(stat_train_loss)):
                log.write( str(stat_train_loss[i]) + "," + str(stat_valid_loss[i]) + "," + \
                           str(stat_valid_tp_E[i]) + "," + str(stat_valid_fp_E[i]) + "," + \
                           str(stat_valid_fn_E[i]) + "," + str(stat_valid_f1_E_score[i]) + "," + \
                           str(stat_valid_tp_S[i]) + "," + str(stat_valid_fp_S[i]) + "," + \
                           str(stat_valid_fn_S[i]) + "," + str(stat_valid_f1_S_score[i]) + "," + \
                           str(stat_valid_f1_score[i]) + "," + str(stat_valid_AUC_E_score[i]) + "," + \
                           str(stat_valid_AUC_S_score[i]) + "," + str(stat_valid_AUC_score[i]) + "," + \
                           str(stat_test_tp_E[i]) + "," + str(stat_test_fp_E[i]) + "," + \
                           str(stat_test_fn_E[i]) + "," + str(stat_test_f1_E_score[i]) + "," + \
                           str(stat_test_tp_S[i]) + "," + str(stat_test_fp_S[i]) + "," + \
                           str(stat_test_fn_S[i]) + "," + str(stat_test_f1_S_score[i]) + "," + \
                           str(stat_test_f1_score[i]) + "," + str(stat_test_AUC_E_score[i]) + "," + \
                           str(stat_test_AUC_S_score[i]) + "," + str(stat_test_AUC_score[i])
                         )
                log.write("\n")
            log.close()

        ##_______________________________Early Stopping______________________________
        my_metric = 0.5*valid_loss + 0.5*(1-V_F1)

        if my_metric < min_my_metric:
            print('Best epoch = ' + str(epoch))
            min_my_metric = my_metric
            best_epoch = epoch
            early_stop_counter = 0
            save_variables(sess, saver, epoch, Patient)
            best_epoch = epoch
            best_f1_score = T_F1
            best_accuracy = accuracy
        else:
            early_stop_counter += 1

        ##______________________________Print Epoch Info____________________________
        print('Epoch : ', epoch)
        print('Early Stopping : ' + str(early_stop_counter) + "/" + str(n_early_stop_epochs))
        print('Learning Rate : ', learning_rate)
        print('Train : ', train_loss)
        print('Valid : ', valid_loss)
        print('Valid F1-score : ', V_F1)
        print('AUC : ', V_AUC)
        print('####################################')
        print('TP Early : ', tp_E)
        print('FP Early : ', fp_E)
        print('FN Early : ', fn_E)
        print('TP Seizure : ', tp_S)
        print('FP Seizure : ', fp_S)
        print('FN Seizure : ', fn_S)
        print('F1-score : ', T_F1)
        print('AUC : ', T_AUC)
        print('####################################')
        print('')

        if early_stop_counter > n_early_stop_epochs :
            # too many consecutive epochs without surpassing the best model
            print('stopping early')
            break

print('Best epoch : ', best_epoch)
print('Best accuracy : ', best_accuracy)
print('Best F1-score : ', best_f1_score)