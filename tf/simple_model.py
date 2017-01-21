import tensorflow as tf
import numpy as np

# a model: shape [1] -> shape [2].
sess = tf.Session()
verbose = True

input_num = 1
input_size = 2
out_size = 1
hn = 10
input_in = tf.placeholder(tf.float32, shape=[input_num, input_size])
out_in = tf.placeholder(tf.float32, shape=[input_num,out_size])
# hidden layer 1
with tf.variable_scope('hidden'):
    hw = tf.get_variable('weights', shape=[input_size, hn], 
                dtype=tf.float32, initializer=tf.random_normal_initializer())
    hidden = tf.nn.sigmoid(tf.matmul(input_in, hw))
# output layer
with tf.variable_scope('linear_output'):
    lw = tf.get_variable('weights', shape=[hn, out_size],
                dtype=tf.float32, initializer=tf.random_normal_initializer())
    out = tf.matmul(hidden,lw)

sess.run(tf.global_variables_initializer())

loss = tf.reduce_mean(tf.square(tf.subtract(out, out_in)))
optimizer = tf.train.GradientDescentOptimizer(0.001)
training = optimizer.optimize(loss)

steps = 100
for _ in range(steps):
    thetas = np.random.rand((input_size, out_size))
    x1 = 100*np.random.rand(input_num, input_size)
    y1 = x1*thetas;

    feed_dict = {
        input_in: x1,
        out_in: y1
    }
    sess.run(training, feed_dict=feed_dict)
    print(sess.run(loss, feed_dict=feed_dict))