import tensorflow as tf
import numpy as np


class ab_seq():

    def __init__(self, session, batch_size = 100, lstm_units = 10, seq_len = 10, verbose = True):
        self.lstm_units = lstm_units
        self.seq_len = seq_len
        self.session = session
        self.batch_size = batch_size
        self.verbose = verbose
        self.__step_state = None

        # training model
        with tf.variable_scope("lstm") as lstm_scope:
            cell = self.cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
            training_seq_in = self.training_seq_in = tf.placeholder(tf.float32, shape=[seq_len, batch_size, 1])
            cell_init_state = self.cell.zero_state(batch_size, tf.float32)
            training_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                        inputs=training_seq_in,
                        initial_state=cell_init_state,
                        time_major=True,
                        scope=lstm_scope,
                        dtype=tf.float32)
            training_output = training_outputs[-1,:,:] # only cares about last output

        s_size = 2
        with tf.variable_scope('training_logits') as scope:
            weights = self.weights = tf.get_variable('weights', [lstm_units, s_size],
                initializer=tf.random_normal_initializer())
            biases = self.biases = tf.get_variable('biases', [s_size],
                initializer=tf.random_normal_initializer())
            training_logits = tf.matmul(training_output, weights) + biases


        with tf.variable_scope('softmax') as softmax:
            training_softmax = self.training_softmax = tf.nn.softmax(training_logits)

        with tf.variable_scope('loss') as loss_scope:
            labels_in = self.labels_in = tf.placeholder(tf.float32, shape=[batch_size, s_size])
            loss = self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(training_logits, labels_in))

        optimizer = tf.train.AdamOptimizer(0.001)
        self.training = optimizer.minimize(loss)

        # # to initialize individual variables
        # session.run(self.weights.initializer)
        # session.run(self.biases.initializer)

        # init all variables
        session.run(tf.global_variables_initializer())

    def train_model(self, data, labels, max_step=10000, break_point=0.01):
        loss = self.loss
        training = self.training
        training_seq_in = self.training_seq_in
        loss = self.loss
        labels_in = self.labels_in
        session = self.session

        feed_dict = {
            training_seq_in: data,
            labels_in: labels
        }
        for i in range(max_step):
            session.run(training, feed_dict=feed_dict)
            if self.verbose:
                print('current loss is : ' + str(session.run(loss, feed_dict=feed_dict)))
            if (session.run(loss, feed_dict=feed_dict) < break_point):
                break

    def predict(self, seq):
        '''
        @seq : string. char of string is either a or b
        '''
        with tf.variable_scope("lstm") as lstm_scope:
            lstm_scope.reuse_variables()

            seq_len = len(seq)
            if seq_len > self.seq_len:
                raise Exception('sequence length is greater than trained length')
            cell = self.cell
            cell_init_state = cell.zero_state(1, tf.float32)
            seq_in = np.zeros((seq_len, 1, 1))
            seq_in[:,0,0] = map(lambda s: ord(s), seq)
            seq_in = tf.convert_to_tensor(seq_in, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                        inputs=seq_in,
                        initial_state=cell_init_state,
                        time_major=True,
                        scope=lstm_scope,
                        dtype=tf.float32)
            output = outputs[-1,:,:]  # only cares about last output

        with tf.variable_scope("softmax") as softmax:
            softmax.reuse_variables()

            weights = self.weights
            biases = self.biases
            logits = tf.matmul(output, weights) + biases
            softmax = tf.nn.softmax(logits)

        return self.session.run(softmax)

    def predict_with_training_model(self, seq):
        training_softmax = self.training_softmax
        training_seq_in = self.training_seq_in
        batch_size = self.batch_size
        seq_len = self.seq_len

        seq_in = np.zeros([seq_len, batch_size, 1])
        seq_in[:,0,0] = map(lambda s: ord(s), seq)

        feed_dict = {
            training_seq_in: seq_in
        }
        result = self.session.run(training_softmax, feed_dict=feed_dict)
        return result

    def step(self, char_in):
        '''
        run a single step
        @char_in :char.
        '''
        cell = self.cell
        if self.__step_state == None:
            self.__step_state = self.cell.zero_state(1, tf.float32)
        prev_state = self.__step_state

        with tf.variable_scope("lstm") as lstm_scope:
            lstm_scope.reuse_variables()

            char_in_tensor = tf.convert_to_tensor([[ord(char_in)]], tf.float32)
            output, next_state = cell(char_in_tensor, prev_state)
            self.__step_state = next_state

        with tf.variable_scope("softmax") as softmax:
            softmax.reuse_variables()

            weights = self.weights
            biases = self.biases
            logits = tf.matmul(output, weights) + biases
            softmax = tf.nn.softmax(logits)

        return self.session.run(softmax)
