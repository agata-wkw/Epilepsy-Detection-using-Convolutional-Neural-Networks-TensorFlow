import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

file_prefix = "/home/aris/Desktop/Epilepsy/data/sp/Patient_7/"
path, dirs, files = next(os.walk(file_prefix));
file_count = len(files);
print(files[0:9]);
##__________________________Read Input__________________________
data_images = [];
labels = [];

for num_data in range(10):

    with open(str(file_prefix + files[num_data]), 'rb') as fid:
        data_images.append(np.fromfile(fid, dtype=np.float32, count=-1));
    if (files[num_data][11] == 'n'):
        labels.append([0, 1]);
    else:
        labels.append([1, 0]);
    fid.close();
##_________________________Post Processing of Data___________________


data_images = np.array(data_images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)
num_of_chan = np.shape(data_images)[1];
pixels = int(num_of_chan / (128 * 128));
print(pixels)

##_________________________CNN Implementation___________________

X = tf.placeholder(tf.float32, shape=[None, num_of_chan], name='x')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
x_image = tf.reshape(X, [-1, 128, 128, pixels])

prob = tf.placeholder(tf.float32, name='Dropout_Keep_Probability')

# Variables to be learned by the model
CW1 = tf.Variable(tf.random_normal(shape=[3, 3, pixels, 64]), dtype=tf.float32, name='conv_weights_1')
Cb1 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32, name='conv_biases_1')

CW2 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 100]), dtype=tf.float32, name='conv_weights_2')
Cb2 = tf.Variable(tf.random_normal(shape=[100]), dtype=tf.float32, name='conv_biases_2')

CW3 = tf.Variable(tf.random_normal(shape=[3, 3, 100, 128]), dtype=tf.float32, name='conv_weights_3')
Cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32, name='conv_biases_3')

CW4 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 150]), dtype=tf.float32, name='conv_weights_4')
Cb4 = tf.Variable(tf.random_normal(shape=[150]), dtype=tf.float32, name='conv_biases_4')

CW5 = tf.Variable(tf.random_normal(shape=[3, 3, 150, 220]), dtype=tf.float32, name='conv_weights_5')
Cb5 = tf.Variable(tf.random_normal(shape=[220]), dtype=tf.float32, name='conv_biases_5')

CW6 = tf.Variable(tf.random_normal(shape=[3, 3, 220, 256]), dtype=tf.float32, name='conv_weights_6')
Cb6 = tf.Variable(tf.random_normal(shape=[256]), dtype=tf.float32, name='conv_biases_6')

W1 = tf.Variable(tf.random_normal(stddev=0.05, shape=[2 * 2 * 256, 100]), dtype=tf.float32, name='weights_1')
W2 = tf.Variable(tf.random_normal(shape=[100, 2]), dtype=tf.float32, name='weights_2')
b2 = tf.Variable(tf.random_normal(shape=[2]), dtype=tf.float32, name='biases_2')

# Convolutional Layer 1.
cl1 = tf.nn.relu(tf.add(tf.nn.conv2d(input=x_image, filter=CW1, strides=[1, 1, 1, 1], padding="SAME"), Cb1))
pool1 = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolutional Layer 2.
cl2 = tf.nn.relu(tf.add(tf.nn.conv2d(input=pool1, filter=CW2, strides=[1, 1, 1, 1], padding="SAME"), Cb2))
pool2 = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolutional Layer 3.
cl3 = tf.nn.relu(tf.add(tf.nn.conv2d(input=pool2, filter=CW3, strides=[1, 1, 1, 1], padding="SAME"), Cb3))
pool3 = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolutional Layer 4.
cl4 = tf.nn.relu(tf.add(tf.nn.conv2d(input=pool3, filter=CW4, strides=[1, 1, 1, 1], padding="SAME"), Cb4))
pool4 = tf.nn.max_pool(cl4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolutional Layer 4.
cl5 = tf.nn.relu(tf.add(tf.nn.conv2d(input=pool4, filter=CW5, strides=[1, 1, 1, 1], padding="SAME"), Cb5))
pool5 = tf.nn.max_pool(cl5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolutional Layer 4.
cl6 = tf.nn.relu(tf.add(tf.nn.conv2d(input=pool5, filter=CW6, strides=[1, 1, 1, 1], padding="SAME"), Cb6))
pool6 = tf.nn.max_pool(cl6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Hidden layer 1.
layer1 = tf.add(tf.matmul(tf.reshape(pool6, [-1, 2 * 2 * 256]), W1))
layer1_b = batch_norm(layer1, center=True, scale=True, is_training=phase, scope='bn1')
layer1_o = tf.nn.relu(layer1_b)

# Dropout Layer.
dropout = tf.nn.dropout(layer1_o, keep_prob=prob)

# Output layer.
logits = tf.add(tf.matmul(dropout, W2), b2)

#### Predictions ####
pred = tf.nn.softmax(logits)
pred_cls = tf.argmax(pred, axis=1)

# Loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(cross_entropy)

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

# Performance Measures
correct_prediction = tf.equal(pred_cls, tf.argmax(Y, axis=1))
corrects = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

from sklearn.model_selection import train_test_split;

train_x, test_x, train_y, test_y = train_test_split(data_images, labels, test_size=0.2, random_state=42);

# # Shuffling the training data at random.
# random_suffle = np.random.choice(np.shape(train_x)[0], np.shape(test_x)[0], replace=False);
# data_images = data_images[random_suffle];
# labels = labels[random_suffle];


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42);

with tf.Session() as sess:
    tf.set_random_seed(1234);
    sess.run(tf.global_variables_initializer());
    print(np.shape(sess.run(pool1, feed_dict={X: data_images})));
    print(np.shape(sess.run(pool2, feed_dict={X: data_images})));
    print(np.shape(sess.run(pool3, feed_dict={X: data_images})));
    print(np.shape(sess.run(pool4, feed_dict={X: data_images})));
    print(np.shape(sess.run(pool5, feed_dict={X: data_images})));
    print(np.shape(sess.run(pool6, feed_dict={X: data_images})));

# with open(filename, 'rb') as fid:
#     data = np.fromfsile(fid, dtype=np.float32, count=-1)
# fid.close()

# n_channels = int(len(data)/(128*128)) 
# data = data.reshape((n_channels, 128, 128))
# data = np.transpose(data, [2, 1, 0])

# print(np.shape(data))


