import tensorflow as tf
import numpy as np
from scipy import stats

class motion_basis_learning():
    '''
    input data has to be shape of (1, 3, n, 1), where n is length. for example,
    accelerometer, 3 represents 3 axis, and n represent time points.
    '''

    def __init__(self, k=50, filter_width=5, pooling_size=4, axis_num=3):
        self.k = k
        self.filter_width = filter_width
        self.pooling_size = pooling_size
        self.axis_num = axis_num

        self.param_scope_name = 'crbm_'+str(id(self))
        with tf.variable_scope(self.param_scope_name) as crbm_scope:
            self.w = tf.get_variable('weights', shape=(axis_num, filter_width, 1, k), dtype=tf.float32, 
                        initializer=tf.random_normal_initializer())
            self.w_r = tf.reverse_v2(self.w, [0,1])
            self.hb = tf.get_variable('hidden_biase', shape=(k,), dtype=tf.float32,
                         initializer=tf.random_normal_initializer())
            self.vb = tf.get_variable('visible_biase', shape=(1,), dtype=tf.float32,
                         initializer=tf.random_normal_initializer())
        self.sess = tf.Session()
        # initialize parameters
        self.sess.run(tf.global_variables_initializer())

    def build_training_model(self, training_data):
        '''
        @training_data: np.array. shape of (1, 3, n, 1)
        '''

        filter_width = self.filter_width
        k = self.k
        sess = self.sess
        param_scope_name = self.param_scope_name

        v_len = training_data.shape[2]
        axis_num = training_data.shape[1]
        h_shape = [v_len-filter_width+1, k]
        v_shape = list(training_data.shape)
        with tf.variable_scope(param_scope_name) as crbm_scope:
            crbm_scope.reuse_variables()    
            w = tf.get_variable('weights', shape=(3, filter_width, 1, k), dtype=tf.float32)
            w_r = tf.reverse_v2(w, [0,1])
            hb = tf.get_variable('hidden_biase', shape=(k,), dtype=tf.float32)
            vb = tf.get_variable('visible_biase', shape=(1,), dtype=tf.float32)
        h_real_in = tf.placeholder(tf.float32, shape=[v_len-filter_width+1, k]) # after full convolution
        v_real_in = tf.placeholder(tf.float32, shape=training_data.shape)
        convolution_real = tf.nn.convolution(v_real_in, w, 'VALID', strides=[1,1])
        energy_real = -tf.reduce_sum(h_real_in*convolution_real[0,0,:,:]) \
                        - tf.reduce_sum(hb*tf.reduce_sum(h_real_in, axis=0)) \
                        - tf.reduce_sum(vb*tf.reduce_sum(v_real_in, axis=0))
        
        h_fantasy_in = tf.placeholder(tf.float32, shape=[v_len-filter_width+1, k]) # after full convolution
        v_fantasy_in = tf.placeholder(tf.float32, shape=training_data.shape)
        convolution_fantasy = tf.nn.convolution(v_fantasy_in, w, 'VALID', strides=[1,1])
        energy_fantasy = -tf.reduce_sum(h_fantasy_in*convolution_fantasy[0,0,:,:]) \
                            - tf.reduce_sum(hb*tf.reduce_sum(h_fantasy_in, axis=0)) \
                            - tf.reduce_sum(vb*tf.reduce_sum(v_fantasy_in, axis=0))

        # regularization
        reg = tf.reduce_mean(tf.nn.sigmoid(convolution_real[0,0,:,:] + hb))
        # loss
        loss = tf.reduce_mean(energy_real - energy_fantasy) + reg

        learning_rate_in = tf.placeholder(tf.float32)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_in)
        training = optimizer.minimize(loss)

        self.h_shape = h_shape
        self.v_shape = v_shape
        self.h_real_in = h_real_in
        self.v_real_in = v_real_in
        self.h_fantasy_in = h_fantasy_in
        self.v_fantasy_in = v_fantasy_in
        self.loss = loss
        self.reg = reg
        self.training_data = training_data
        self.learning_rate_in = learning_rate_in
        self.training = training

    def train(self, steps=100, convergence_point=None, learning_rate=0.001, sigma=2, verbose=False, gibb_steps=1):
        training_data = self.training_data
        h_shape = self.h_shape
        v_shape = self.v_shape
        h_real_in = self.h_real_in
        v_real_in = self.v_real_in
        h_fantasy_in = self.h_fantasy_in
        v_fantasy_in = self.v_fantasy_in
        sess = self.sess
        loss = self.loss
        reg = self.reg
        learning_rate_in = self.learning_rate_in
        training = self.training

        for _ in range(steps):
            #gibb sampling
            v_real = training_data
            h_real = self._gen_h(tf.convert_to_tensor(v_real, tf.float32))

            h_fantasy = h_real.copy()
            for _ in range(gibb_steps):
                v_fantasy = self._gen_v(tf.convert_to_tensor(h_fantasy, tf.float32), sigma)
                h_fantasy = self._gen_h(tf.convert_to_tensor(v_fantasy, tf.float32))

            feed_dict = {
                h_real_in: h_real, 
                v_real_in: v_real,
                h_fantasy_in: h_fantasy,
                v_fantasy_in: v_fantasy,
                learning_rate_in: learning_rate
            }
            sess.run(training, feed_dict=feed_dict)
            if verbose:
                print(str(sess.run(loss, feed_dict=feed_dict)) + ',' + str(sess.run(reg, feed_dict=feed_dict)))

    def _gen_h(self, v):
        '''
        @v: tensor
        '''
        pooling_size = self.pooling_size
        h_shape = self.h_shape
        w = self.w
        hb = self.hb
        sess = self.sess

        h = np.zeros(h_shape)
        convoluted = sess.run(tf.nn.convolution(v, w, 'VALID', strides=[1,1]) + hb)
        for i in range(convoluted.shape[3]):
            for j in range(convoluted.shape[2]/pooling_size):
                convoluted_j = convoluted[0,0,j*pooling_size:(j+1)*pooling_size,i]
                # how to deal with overflow?
                prob_not_normalized = np.power(np.e, np.concatenate((convoluted_j, [1]), axis=0)) 
                prob = prob_not_normalized / sum(prob_not_normalized)
                # h_j = stats.rv_discrete(values=(range(len(prob)), prob)).rvs() # bottle neck.
                h_j = np.random.choice(range(pooling_size+1), p=prob)
                if h_j == pooling_size:
                    # none of h_real[j*pooling_size:j*(pooling_size+1),i] be 1
                    pass
                else:
                    h[j*pooling_size+h_j,i] = 1
        return h

    def _gen_v(self, h, sigma):
        '''
        @h: tensor
        '''
        v_shape = self.v_shape
        axis_num = self.axis_num
        filter_width = self.filter_width
        w_r = self.w_r
        vb = self.vb
        sess = self.sess

        v = np.zeros(v_shape)
        # convolution related to reconstruction
        h_rec_in_fitted = tf.expand_dims(tf.expand_dims(tf.expand_dims(h, 0), 0), -1)
        h_rec_in_fitted = tf.pad(h_rec_in_fitted, [[0,0],[axis_num-1, axis_num-1],[filter_width-1,filter_width-1],[0,0], [0,0]])
        w_r_fitted = tf.expand_dims(tf.expand_dims(tf.squeeze(w_r), -1), -1)
        convolution_rec_raw = sess.run(tf.nn.convolution(h_rec_in_fitted, w_r_fitted, 'VALID', strides=[1,1,1]))
        convolution_rec = convolution_rec_raw[:,:,:,:,0] # same shape as `training_data`
        vb = sess.run(vb)
        for i0 in range(v_shape[0]):
            for i1 in range(v_shape[1]):
                for i2 in range(v_shape[2]):
                    for i3 in range(v_shape[3]):
                        mean = convolution_rec[i0,i1,i2,i3] + vb
                        v[i0,i1,i2,i3] = stats.norm.rvs(mean,sigma)
        return v




