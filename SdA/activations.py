import theano.tensor as T

def activation_none(x):
	return T.nnet.relu(x, 1)

def tanh_lecun(x):
	return 1.7159 * T.tanh((2/3) * x)

def const_tanh_lecun(x):
	return T.tanh(x) + 0.01 * x

def get_activation_function(activation_name):
	functions = {'sigmoid': T.nnet.sigmoid,
			'tanh': T.tanh,
			'relu': T.nnet.relu,
			'hard_sigmoid': T.nnet.hard_sigmoid,
			'ultra_fast_sigmoid': T.nnet.ultra_fast_sigmoid,
			'tanh_lecun': tanh_lecun,
			'const_tanh_lecun': const_tanh_lecun,
			'none': activation_none}
	return functions[activation_name]

