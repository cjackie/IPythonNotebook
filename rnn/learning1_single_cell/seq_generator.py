import tensorflow as tf
import numpy as np
from scipy import stats

__verbose__ = False


if '__main__' == __name__:
    with tf.Session() as sess:
        # make cell
        symbol_size = 5
        batch_size = 1
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(symbol_size, activation=tf.sigmoid)
        state = single_cell.zero_state(batch_size, tf.float32)
        x = tf.placeholder(tf.float32, shape=[batch_size,1], name='x')
        cell = single_cell(x, state)

        sess.run(tf.global_variables_initializer()) # init
        
        alphabets = [chr(ord('a')+i) for i in  range(26)]
        alphabets = alphabets[0:symbol_size]
        seq_n = 10
        cur_x = alphabets[0]
        seq = [cur_x]
        for i in range(seq_n-1):
            o = sess.run(cell, feed_dict={ x:[[ord(cur_x)]] })
            if __verbose__:
                print("===== state of the cell =====")
                print(o[1].c)  # print out states
                print("=============================")
            probabilities = sess.run(tf.nn.softmax(o[0]))
            if __verbose__:
                print('===== probabilities =====')
                print(probabilities)
                print('=========================')
            index = stats.rv_discrete(values=(range(probabilities.shape[1]), probabilities[0,:])).rvs()
            cur_x = alphabets[index]
        seq.append(cur_x)
        print('===== result =====')
        print(reduce(lambda r, c: r+c, seq, ''))
        print('==================')

