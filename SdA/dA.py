import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils.utils import tile_raster_images
from optims import SGD, Adam
from common import _FLOATX

try:
	import PIL.Image as Image
except ImportError:
	import Image


class dA(object):

	def __init__(
		self,
		numpy_rng,
		n_visible,
		n_hidden,
		theano_rng=None,
		input=None,
		W=None,
		bhid=None,
		bvis=None,
		activation_in = T.nnet.sigmoid,
		activation_reconstruction = T.nnet.sigmoid
	):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.activation_in = activation_in
		self.activation_reconstruction = activation_reconstruction

		# create a Theano random generator that gives symbolic random values
		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# note : W' was written as `W_prime` and b' as `b_prime`
		if not W:
			initial_W = numpy.asarray(
				numpy_rng.uniform(
					low= - numpy.sqrt(6. / (n_hidden + n_visible)),
					high= numpy.sqrt(6. / (n_hidden + n_visible)),
					size=(n_visible, n_hidden)
				),
				dtype=_FLOATX
			)
			if self.activation_in == theano.tensor.nnet.sigmoid:
				initial_W *= 4
			W = theano.shared(value=initial_W, name='W', borrow=True)
		
		if not bvis:
			bvis = theano.shared(
				value=numpy.zeros(
					n_visible,
					dtype=_FLOATX
				),
				borrow=True
			)

		if not bhid:
			bhid = theano.shared(
				value=numpy.zeros(
					n_hidden,
					dtype=_FLOATX
				),
				name='b',
				borrow=True
			)

		self.W = W
		# b corresponds to the bias of the hidden
		self.b = bhid
		# b_prime corresponds to the bias of the visible
		self.b_prime = bvis
		# tied weights, therefore W_prime is W transpose
		self.W_prime = self.W.T
		self.theano_rng = theano_rng
		# if no input is given, generate a variable representing the input
		if input is None:
			# we use a matrix because we expect a minibatch of several
			# examples, each example being a row
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

		self.params = [self.W, self.b, self.b_prime]

	def get_corrupted_input(self, input, corruption_level):
		return self.theano_rng.binomial(size=input.shape, n=1,
			p=1 - corruption_level, dtype=_FLOATX) * input

	def get_hidden_values(self, input):
		""" Computes the values of the hidden layer """
		return self.activation_in(T.dot(input, self.W) + self.b)

	def get_reconstructed_input(self, hidden):
		"""Computes the reconstructed input given the values of the
		hidden layer """
		return self.activation_reconstruction(T.dot(hidden, self.W_prime) + self.b_prime)

	def get_cost_updates(self, corruption_level, learning_rate):
		""" This function computes the cost and the updates for one trainng
		step of the dA """

		tilde_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)
		# note : we sum over the size of a datapoint; if we are using
		#		 minibatches, L will be a vector, with one entry per
		#		 example in minibatch
		L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
		#L = T.sum((self.x - z) ** 2, axis=1) / 2 
		# note : L is now a vector, where each element is the
		#		 cross-entropy cost of the reconstruction of the
		#		 corresponding example of the minibatch. We need to
		#		 compute the average of all these to get the cost of
		#		 the minibatch
		cost = T.mean(L) 

		# compute the gradients of the cost of the `dA` with respect
		# to its parameters
		gparams = T.grad(cost, self.params)
		# generate the list of updates
		op = SGD(learning_rate, decay = .0)
		updates = op.get_updates(self.params, gparams)

		return (cost, updates)

def test_dA(learning_rate=0.1, training_epochs=15,
			dataset='mnist.pkl.gz',
			batch_size=20, output_folder='dA_plots'):

	"""
	This demo is tested on MNIST

	:type learning_rate: float
	:param learning_rate: learning rate used for training the DeNosing
						  AutoEncoder

	:type training_epochs: int
	:param training_epochs: number of epochs used for training

	:type dataset: string
	:param dataset: path to the picked dataset

	"""
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()    # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images

	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)
	os.chdir(output_folder)

	####################################
	# BUILDING THE MODEL NO CORRUPTION #
	####################################

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	da = dA(
		numpy_rng=rng,
		theano_rng=theano_rng,
		input=x,
		n_visible=28 * 28,
		n_hidden=500
	)

	cost, updates = da.get_cost_updates(
		corruption_level=0.,
		learning_rate=learning_rate
	)

	train_da = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size]
		}
	)

	start_time = time.clock()

	############
	# TRAINING #
	############

	# go through training epochs
	for epoch in xrange(training_epochs):
		# go through trainng set
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_da(batch_index))

		print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

	end_time = time.clock()

	training_time = (end_time - start_time)

	print >> sys.stderr, ('The no corruption code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((training_time) / 60.))
	image = Image.fromarray(
		tile_raster_images(X=da.W.get_value(borrow=True).T,
						   img_shape=(28, 28), tile_shape=(10, 10),
						   tile_spacing=(1, 1)))
	image.save('filters_corruption_0.png')

	#####################################
	# BUILDING THE MODEL CORRUPTION 30% #
	#####################################

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	da = dA(
		numpy_rng=rng,
		theano_rng=theano_rng,
		input=x,
		n_visible=28 * 28,
		n_hidden=500
	)

	cost, updates = da.get_cost_updates(
		corruption_level=0.3,
		learning_rate=learning_rate
	)

	train_da = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size]
		}
	)

	start_time = time.clock()

	############
	# TRAINING #
	############

	# go through training epochs
	for epoch in xrange(training_epochs):
		# go through trainng set
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_da(batch_index))

		print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

	end_time = time.clock()

	training_time = (end_time - start_time)

	print >> sys.stderr, ('The 30% corruption code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % (training_time / 60.))

	image = Image.fromarray(tile_raster_images(
		X=da.W.get_value(borrow=True).T,
		img_shape=(28, 28), tile_shape=(10, 10),
		tile_spacing=(1, 1)))
	image.save('filters_corruption_30.png')

	os.chdir('../')


if __name__ == '__main__':
	test_dA()
