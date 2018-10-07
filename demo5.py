#coding:utf-8


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing


# dataprocess
dataframe = pd.read_csv('seq100x.txt',header=None, names = ['y','x'] )


trainingset_x=list(dataframe.x)
trainingset_y=list(dataframe.y)
for i in range(len(trainingset_x)):
    trainingset_x[i] = list(str(trainingset_x[i]))
    temp = []
    for j in range(60):
        if j<len(trainingset_x[i]):
            if trainingset_x[i][j] == 'A':
                temp.append([1,0,0,0])
            elif trainingset_x[i][j] == 'C':
                temp.append([0,1,0,0])
            elif trainingset_x[i][j] == 'G':
                temp.append([0,0,1,0])
            else:
                temp.append([0,0,0,1])
        else:
            temp.append([0,0,0,0])
    trainingset_x[i] = temp
trainingset_x = np.reshape(trainingset_x,(len(trainingset_x),60,4))
trainingset_y = np.array(trainingset_y)

encoder1 = preprocessing.LabelBinarizer()
trainingset_y = encoder1.fit_transform(trainingset_y)

# hyperparameters
lr = 0.0001
training_iters = len(trainingset_x)
batch_size = 100
epoch_num = 100
n_inputs = 4   # equals to embedding dim
n_steps = 60    # time steps, length of single DNA seq
n_hidden_units = 4   # neurons in hidden layer,it equal to output length ,also the n_classes.
n_hidden_layers = 2  # depth of network
n_classes = len(trainingset_y[0])    # number of different y



x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


#initialize weights and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)



def LSTMdemo(X):

    X = tf.reshape(X, [-1, n_inputs])
    w_in = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
    b_in = bias_variable([n_hidden_units, ])
    X_in = tf.matmul(X, w_in) + b_in
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    #lstm
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=0.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_hidden_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)


    #dense layer
    fc1_in = tf.reshape(outputs,[batch_size,n_inputs*n_steps])
    w_fc1 = weight_variable([n_inputs*n_steps,1024])
    w_fc2 = weight_variable([1024, 1024])
    fc1 = tf.matmul(fc1_in, w_fc1)
    fc2 = tf.matmul(fc1, w_fc2)

    #softmax layer
    b_out = bias_variable([n_classes, ])
    w_sf = weight_variable([1024,n_classes])
    softmax = tf.matmul(fc2,w_sf) +b_out


    results = softmax
    return results




pred = LSTMdemo(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver(max_to_keep=5)


def next(self, batch_size):
    batch_data = (self[step * batch_size:min(step*batch_size +
                                         batch_size, len(self.data))])
    return batch_data


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epoch_num):
        lr=10*lr
        step = 0
        a = []
        print('we are running the epoch %d'%epoch)
        while (step+1) * batch_size < training_iters:
            batch_xs = next(trainingset_x,batch_size)
            batch_ys = next(trainingset_y,batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            a.append(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
            step += 1
        print('the accuracy of this epoch is:',np.mean(a))   # each batch will record its accuracy into a[],and each epoch we will
        if epoch % 20 == 0: saver.save(sess,'ckpt/demo5.ckpt',global_step= epoch)