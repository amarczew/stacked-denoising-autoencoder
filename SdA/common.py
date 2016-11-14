import theano 

_FLOATX = theano.config.floatX
_EPSILON = 10e-8

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
