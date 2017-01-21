import tensorflow as tf
import numpy as np
from scipy import stats

class motion_basis_learning():
    '''
    input data has to be shape of (m, 3, n, 1), where n is length, and m number of batch.
    for example, accelerometer, 3 represents 3 axis, and n represent time points.
    '''
    SAVE_PARAMS_PATH = './save_variables/params'
    SUMMARY_DIR = './summaries/'

    def __init__(self, k=50, filter_width=5, pooling_size=4, axis_num=3):
        self.k = k
        self.filter_width = filter_width
        self.pooling_size = pooling_size
        self.axis_num = axis_num
        self.accumulated_steps = 0

        self.param_scope_name = 'motion_basis_learning'
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


    def build_training_model(self, training_data, restore_params_path=None, enable_save=True, 
                save_interval=10, enable_summary=True, summary_flush_secs =10):
        '''
        @training_data: np.array. shape of (m, 3, n, 1)
        @restore_params_path: string. if not None, restore variables. 
            save
        '''

        filter_width = self.filter_width
        k = self.k
        sess = self.sess
        param_scope_name = self.param_scope_name
        batch_size = training_data.shape[0]

        if restore_params_path != None:
            # restore parameters
            tf.train.Saver().restore(sess, restore_params_path)

        v_len = training_data.shape[2]
        axis_num = training_data.shape[1]
        h_shape = [batch_size, v_len-filter_width+1, k]
        v_shape = list(training_data.shape)
        with tf.variable_scope(param_scope_name) as crbm_scope:
            crbm_scope.reuse_variables()    
            w = tf.get_variable('weights', shape=(3, filter_width, 1, k), dtype=tf.float32)
            w_r = tf.reverse_v2(w, [0,1])
            hb = tf.get_variable('hidden_biase', shape=(k,), dtype=tf.float32)
            vb = tf.get_variable('visible_biase', shape=(1,), dtype=tf.float32)

        h_real_in = tf.placeholder(tf.float32, shape=[batch_size, v_len-filter_width+1, k]) # after full convolution
        v_real_in = tf.placeholder(tf.float32, shape=training_data.shape)
        convolution_real = tf.nn.convolution(v_real_in, w, 'VALID', strides=[1,1])
        energy_real = -tf.reduce_sum(h_real_in*convolution_real, axis=[1,2,3]) \
                        - tf.reduce_sum(hb*tf.reduce_sum(h_real_in, axis=[1]), axis=[1]) \
                        - tf.reduce_sum(vb*tf.reduce_sum(v_real_in, axis=[1,2,3]))
        
        h_fantasy_in = tf.placeholder(tf.float32, shape=[batch_size, v_len-filter_width+1, k]) # after full convolution
        v_fantasy_in = tf.placeholder(tf.float32, shape=training_data.shape)
        convolution_fantasy = tf.nn.convolution(v_fantasy_in, w, 'VALID', strides=[1,1])
        energy_fantasy = -tf.reduce_sum(h_fantasy_in*convolution_fantasy, axis=[1,2,3]) \
                            - tf.reduce_sum(hb*tf.reduce_sum(h_fantasy_in, axis=[1]), axis=[1]) \
                            - tf.reduce_sum(vb*tf.reduce_sum(v_fantasy_in, axis=[1,2,3]))

        # to prevent weight "explosion" during learning.
        data_len = training_data.shape[2]
        energy_real = energy_real / data_len
        energy_fantasy = energy_fantasy / data_len

        # regularization
        reg = tf.reduce_mean(tf.nn.sigmoid(convolution_real + hb))
        # loss
        loss = tf.reduce_mean(energy_real - energy_fantasy) + reg

        learning_rate_in = tf.placeholder(tf.float32)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_in)
        training = optimizer.minimize(loss)

        summary_file = None
        summaries = None
        if enable_summary:
            summary_file = tf.summary.FileWriter(motion_basis_learning.SUMMARY_DIR, 
                    sess.graph, flush_secs=summary_flush_secs)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('probability', reg)
            w_gradient = optimizer.compute_gradients(loss, var_list=[w])
            tf.summary.histogram('weights_gradient', w_gradient)
            hb_gradient = optimizer.compute_gradients(loss, var_list=[hb])
            tf.summary.histogram('hidden_biases', hb_gradient)
            vb_gradient = optimizer.compute_gradients(loss, var_list=[vb])
            tf.summary.histogram('visible_biases', vb_gradient)
            summaries = tf.summary.merge_all()

        params_saver = None
        if enable_save:
            params_saver = tf.train.Saver(var_list=[w, vb, hb])

        self.batch_size = batch_size
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

        self.enable_summary = enable_summary
        self.enable_save = enable_save
        self.save_interval = save_interval
        self.params_saver = params_saver
        self.enable_summary = enable_summary
        self.summaries = summaries
        self.summary_file = summary_file

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

        enable_summary = self.enable_summary
        summaries = self.summaries
        summary_file = self.summary_file
        enable_save = self.enable_save
        save_interval = self.save_interval
        params_saver = self.params_saver
        accumulated_steps = self.accumulated_steps

        for s in range(steps):
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
            if enable_summary:
                summary_file.add_summary(sess.run(summaries, feed_dict=feed_dict), accumulated_steps+s)
            if enable_save and (accumulated_steps+s) % save_interval == 0:
                params_saver.save(sess, motion_basis_learning.SAVE_PARAMS_PATH, global_step=accumulated_steps+s)

        self.accumulated_steps += steps


    def _gen_h(self, v):
        '''
        @v: tensor
        '''
        pooling_size = self.pooling_size
        h_shape = self.h_shape
        w = self.w
        hb = self.hb
        sess = self.sess
        batch_size = self.batch_size

        h = np.zeros(h_shape)
        convoluted = sess.run(tf.nn.convolution(v, w, 'VALID', strides=[1,1]) + hb)
        for batch_i in range(batch_size):
            for i in range(convoluted.shape[3]):
                for j in range(convoluted.shape[2]/pooling_size):
                    convoluted_j = convoluted[batch_i,0,j*pooling_size:(j+1)*pooling_size,i]
                    # how to deal with overflow?
                    prob_not_normalized = np.power(np.e, np.concatenate((convoluted_j, [1]), axis=0)) 
                    prob = prob_not_normalized / sum(prob_not_normalized)
                    # h_j = stats.rv_discrete(values=(range(len(prob)), prob)).rvs() # bottle neck.
                    h_j = np.random.choice(range(pooling_size+1), p=prob)
                    if h_j == pooling_size:
                        # none of h_real[j*pooling_size:j*(pooling_size+1),i] be 1
                        pass
                    else:
                        h[batch_i,j*pooling_size+h_j,i] = 1
        return h

    def _gen_v(self, h, sigma):
        '''
        @h: tensor
        @sigma: float. perturbation.
        '''
        v_shape = self.v_shape
        axis_num = self.axis_num
        filter_width = self.filter_width
        w_r = self.w_r
        vb = self.vb
        sess = self.sess

        v = np.zeros(v_shape)
        # convolution related to reconstruction
        h_rec_in_fitted = tf.expand_dims(tf.expand_dims(h, 1), -1)
        h_rec_in_fitted = tf.pad(h_rec_in_fitted, [[0,0],[axis_num-1, axis_num-1],[filter_width-1,filter_width-1],[0,0],[0,0]])
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




