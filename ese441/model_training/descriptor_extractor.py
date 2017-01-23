import tensorflow as tf
import numpy as np

class DescriptorExtractor():

	def __init__(self, motion_basis_learned):
		'''
		@motion_basis_learned: MotionBasisLearner. MotionBasisLearner 
			after training is done.
		'''
		self.w = motion_basis_learned.w
		self.hb = motion_basis_learned.vb
		self.axis_num = motion_basis_learned.axis_num
		self.sess = motion_basis_learned.sess

	def extract_descriptor(self, data):
		'''
		@data, numpy array. shape (@self.axis_num, len), where len is greater than
			w.shape[1]
		@return: numpy array. shape is (n,) where n is number of 
			features.
		'''
		descriptor_raw = self._extract_descriptor_raw(data)
		f = descriptor_raw.shape[0]
		t = descriptor_raw.shape[1]

		descriptor = np.zeros(f)
		for k in range(f):
			descriptor[k] = np.sum(descriptor_raw[k, :])/float(t)
		return descriptor

	def _extract_descriptor_raw(self, data):
		'''
		@data, numpy array. shape (@self.axis_num, len), where len is greater than
			w.shape[1]
		@return: numpy array. shape (k, m). k is number of features
		'''
		axis_num = self.axis_num
		hb = self.hb
		w = self.w
		sess = self.sess

		assert(axis_num == data.shape[0] and len(data.shape) == 2)

		data_fitted = tf.convert_to_tensor(data, dtype=tf.float32)
		data_fitted = tf.expand_dims(tf.expand_dims(data_fitted, axis=0), axis=-1)
		convolution = tf.nn.convolution(data_fitted, w, 'VALID', strides=[1,1])
		probabilities = tf.squeeze(tf.nn.sigmoid(convolution + hb))
		probabilities = sess.run(probabilities)

		descriptor_raw = np.zeros((probabilities.shape[1], probabilities.shape[0]))
		for k in range(probabilities.shape[1]):
			for i in range(probabilities.shape[0]):
				p = probabilities[i, k]
				descriptor_raw[k, i] = np.random.choice([0,1], p=[1-p, p])
		return descriptor_raw





		