import numpy as np
import theano
import theano.tensor as T

from common import _FLOATX, _EPSILON

def variable(value, dtype=_FLOATX, name=None):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)

class Optimizer(object):

    def __init__(
	self,
	epsilon = _EPSILON):
	self.epsilon = epsilon
	self.updates = []
	self.weights = []


class SGD(Optimizer):
    
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False):
        super(SGD, self).__init__()
        self.iterations = variable(.0)
        self.nesterov = nesterov
	if type(lr) == float:
	    self.lr = variable(lr)
	else:
	    self.lr = lr
        self.momentum = variable(momentum)
        self.decay = variable(decay)

    def get_updates(self, params, gparams):
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        # momentum
        self.weights = [variable(np.zeros(p.get_value().shape)) for p in params]
        for p, g, m in zip(params, gparams, self.weights):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, new_p))
        return self.updates

class Adam(Optimizer):
    
    def __init__(
	self, lr=0.001, b1=0.9, b2=0.999, epsilon=_EPSILON):
	super(Adam, self).__init__(epsilon = epsilon)
	self.iterations = variable(0.)
	if type(lr) == float:
	    self.lr = variable(lr)
	else:
	    self.lr = lr
	self.b1 = b1
	self.b2 = b2
   
    def get_updates(self, params, gparams):
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * T.sqrt(1. - T.pow(self.b2, t)) / (1. - T.pow(self.b1, t))

        ms = [variable(np.zeros(p.get_value().shape)) for p in params]
        vs = [variable(np.zeros(p.get_value().shape)) for p in params]
        self.weights = ms + vs

        for p, g, m, v in zip(params, gparams, ms, vs):
            m_t = (self.b1 * m) + (1. - self.b1) * g
            v_t = (self.b2 * v) + (1. - self.b2) * T.sqr(g)
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))

            new_p = p_t
            self.updates.append((p, new_p))
        return self.updates 
