####################################################
# use this script to get parameters for extracting 
# descriptor
##################################

from model_training.motion_basis_learning import MotionBasisLearner
import numpy as np

config = {
	'time_length': 600, 
	'k': 50, 
	'filter_width': 5, 
	'pooling_size': 4,

	'accelerometer_restore_path': None,
	'gyroscope_restore_path': None,
	'accelerometer_files': ['data/accelerometer_1.csv', 'data/accelerometer_2.csv'],
	'accelerometer_variable_path': 'variables_saved/accel/variables',
	'accelerometer_summaries_dir': 'summaries/accel/',
	'gyroscope_files': ['data/gyroscope_1.csv', 'data/gyroscope_2.csv'],
	'gyroscope_variable_path': 'variables_saved/gyro/variables',
	'gyroscope_summaries_dir': 'summaries/gyro/'
}

def get_data(data_raw, start_i, length):
	'''
	@data_raw: array. each element is a string encode comma seperated values.
	   assume each element is x,y,z,time
	@start_i: int.
	@length: int.
	@return: numpy array. shape of (3, config['time_length'])
	'''
	data = np.ndarray(shape=(3, length))
	for i in range(length):
		line = data_raw[start_i+i]
		tokens = line.split(',')
		data[0,i] = float(tokens[0])
		data[1,i] = float(tokens[1])
		data[2,i] = float(tokens[2])
	return data

def get_data_from_files(filename_list, config):
	'''
	@filename_list: list of string. each file csv formatted of "x,y,z,time"
	@return: ndarray. shape (n, 3, config['time_length'], 1))
	'''
	data_is_empty = True
	for fn in filename_list:
		f = open(fn, 'r')
		f.readline()
		lines = f.read().strip().split('\n')
		f.close()

		number_seq = len(lines) / config['time_length']
		for i in range(number_seq):
			data_i = get_data(lines, i*config['time_length'], config['time_length'])
			data_i = np.expand_dims(np.expand_dims(data_i, 0), -1)
			if data_is_empty:
				data = data_i
				data_is_empty = False
			else:
				data = np.concatenate((data, data_i), axis=0)
	return data


# getting data
accelerometer_data = get_data_from_files(config['accelerometer_files'], config)
gyroscope_data = get_data_from_files(config['gyroscope_files'], config)

# init model
accel_basis = MotionBasisLearner(k=config['k'], filter_width=config['filter_width'],
		pooling_size=config['pooling_size'], save_params_path=config['accelerometer_variable_path'],
		summary_dir=config['accelerometer_summaries_dir'], param_scope_name=config['accelerometer_variable_path'])
gyro_basis = MotionBasisLearner(k=config['k'], filter_width=config['filter_width'],
		pooling_size=config['pooling_size'], save_params_path=config['gyroscope_variable_path'],
		summary_dir=config['gyroscope_summaries_dir'], param_scope_name=config['gyroscope_variable_path'])

# init training
summary_flush_secs = 10
save_interval = 100
accel_basis.build_training_model(accelerometer_data, restore_params_path=config['accelerometer_restore_path'], 
		summary_flush_secs=summary_flush_secs, save_interval=save_interval)
gyro_basis.build_training_model(gyroscope_data, restore_params_path=config['gyroscope_restore_path'],
		summary_flush_secs=summary_flush_secs, save_interval=save_interval)

# start training
learning_rate = 0.01
steps = 10
verbose = True
while True:
	if verbose:
		print("learning acceleromenter basis:")
	accel_basis.train(verbose=verbose, steps=steps, learning_rate=learning_rate)

	if verbose:
		print("learning gyroscope basis:")
	gyro_basis.train(verbose=verbose, steps=steps, learning_rate=learning_rate)

	