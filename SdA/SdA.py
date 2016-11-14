import os
import sys
import time

import numpy as np
import cPickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from optims import SGD, Adam

class SdA(object):

	def __init__(
		self,
		numpy_rng,
		n_ins,
		hidden_layers_sizes,
		n_outs,
		theano_rng=None,
		activation = T.nnet.sigmoid,
		autoencoder_in_activation = T.nnet.sigmoid,
		autoencoder_reconstruction_activation = T.nnet.sigmoid
	):

		self.sigmoid_layers = []
		self.dA_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)

		assert self.n_layers > 0

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		# allocate symbolic variables for the data
		self.x = T.matrix('x')	# the data is presented as rasterized images
		self.y = T.ivector('y')  # the labels are presented as 1D vector of
								 # [int] labels

		# The SdA is an MLP, for which all weights of intermediate layers
		# are shared with a different denoising autoencoders
		# We will first construct the SdA as a deep multilayer perceptron,
		# and when constructing each sigmoidal layer we also construct a
		# denoising autoencoder that shares weights with that layer
		# During pretraining we will train these autoencoders (which will
		# lead to chainging the weights of the MLP as well)
		# During finetunining we will finish training the SdA by doing
		# stochastich gradient descent on the MLP

		for i in xrange(self.n_layers):
			# construct the sigmoidal layer

			# the size of the input is either the number of hidden units of
			# the layer below or the input size if we are on the first layer
			if i == 0:
				input_size = n_ins
			else:
				input_size = hidden_layers_sizes[i - 1]

			# the input to this layer is either the activation of the hidden
			# layer below or the input of the SdA if you are on the first
			# layer
			if i == 0:
				layer_input = self.x
			else:
				layer_input = self.sigmoid_layers[-1].output


			sigmoid_layer = HiddenLayer(rng=numpy_rng,
										input=layer_input,
										n_in=input_size,
										n_out=hidden_layers_sizes[i],
										activation=activation)
			# add the layer to our list of layers
			self.sigmoid_layers.append(sigmoid_layer)
			# its arguably a philosophical question...
			# but we are going to only declare that the parameters of the
			# sigmoid_layers are parameters of the StackedDAA
			# the visible biases in the dA are parameters of those
			# dA, but not the SdA
			self.params.extend(sigmoid_layer.params)

			# Construct a denoising autoencoder that shared weights with this
			# layer
			dA_layer = dA(numpy_rng=numpy_rng,
						  theano_rng=theano_rng,
						  input=layer_input,
						  n_visible=input_size,
						  n_hidden=hidden_layers_sizes[i],
						  W=sigmoid_layer.W,
						  bhid=sigmoid_layer.b,
						  activation_in = autoencoder_in_activation,
						  activation_reconstruction = autoencoder_reconstruction_activation)
			self.dA_layers.append(dA_layer)
		# We now need to add a logistic layer on top of the MLP
		self.logLayer = LogisticRegression(
			input=self.sigmoid_layers[-1].output,
			n_in=hidden_layers_sizes[-1],
			n_out=n_outs
		)

		self.params.extend(self.logLayer.params)
		# construct a function that implements one step of finetunining

		# compute the cost for second phase of training,
		# defined as the negative log likelihood
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		# compute the gradients with respect to the model parameters
		# symbolic variable that points to the number of errors made on the
		# minibatch given by self.x and self.y
		self.errors = self.logLayer.errors(self.y)

	def pretraining_functions(self, pretrain_set_x, batch_size):
		
        # index to a [mini]batch
		index = T.lscalar('index')	# index to a minibatch
		corruption_level = T.scalar('corruption')  # % of corruption to use
		learning_rate = T.scalar('lr')	# learning rate to use
		# begining of a batch, given `index`
		batch_begin = index * batch_size
		# ending of a batch given `index`
		batch_end = batch_begin + batch_size

		pretrain_fns = []
		for dA in self.dA_layers:
			# get the cost and the updates list
			cost, updates = dA.get_cost_updates(corruption_level,
												learning_rate)
			# compile the theano function
			fn = theano.function(
				inputs=[
					index,
					theano.In(corruption_level, value=0.2),
					theano.In(learning_rate, value=0.1)
				],
				outputs=cost,
				updates=updates,
				givens={
					self.x: pretrain_set_x[batch_begin: batch_end]
				}
			)
			# append `fn` to the list of functions
			pretrain_fns.append(fn)

		return pretrain_fns

	def build_finetune_functions(self, datasets, batch_size, learning_rate):
		
		(pretrain_set_x, pretrain_set_y) = datasets[0]
		(train_set_x, train_set_y) = datasets[1]
		(valid_set_x, valid_set_y) = datasets[2]
		(test_set_x, test_set_y) = datasets[3]
		(new_representation_set_x, _) = datasets[4]
		
		# compute number of minibatches for training, validation and testing
		n_train_batches = train_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_valid_batches /= batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_test_batches /= batch_size

		index = T.lscalar('index')	# index to a [mini]batch

		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.finetune_cost, self.params)

		# compute list of fine-tuning updates
		opt = SGD(learning_rate, decay=0.)
		updates = opt.get_updates(self.params, gparams)

		# finetune function
		train_fn = theano.function(
			inputs=[index],
			outputs=self.finetune_cost,
			updates=updates,
			givens={
			self.x: train_set_x[
					index * batch_size: (index + 1) * batch_size
				],
			self.y: train_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			},
			name='train'
		)
		
		# Return % of error on train set
		train_score_i = theano.function(
			[index],
			self.errors,
			givens={
				self.x: train_set_x[
					index * batch_size: (index + 1) * batch_size
				],
				self.y: train_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)

		# Return % of error on valid set
		valid_score_i = theano.function(
			[index],
			self.errors,
			givens={
				self.x: valid_set_x[
					index * batch_size: (index + 1) * batch_size
				],
				self.y: valid_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)

		# Return % of error on test set
		test_score_i = theano.function(
			[index],
			self.errors,
			givens={
				self.x: test_set_x[
					index * batch_size: (index + 1) * batch_size
				],
				self.y: test_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)
		
		predict_train = theano.function(
			[index],
			outputs=(self.logLayer.y_pred, self.y),
			givens={
			self.x: train_set_x[
					index * batch_size: (index + 1) * batch_size
				],
			self.y: train_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)

		predict_test = theano.function(
			[index],
			outputs=(self.logLayer.y_pred, self.y),
			givens={
			self.x: test_set_x[
					index * batch_size: (index + 1) * batch_size
				],
			self.y: test_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)

		predict_valid = theano.function(
			[index],
			outputs=(self.logLayer.y_pred, self.y),
			givens={
			self.x: valid_set_x[
					index * batch_size: (index + 1) * batch_size
				],
			self.y: valid_set_y[
					index * batch_size: (index + 1) * batch_size
				]
			}
		)

		# Create a function that scans the entire train set
		def train_score():
			return [train_score_i(i) for i in xrange(n_train_batches + 1)]
		
		# Create a function that scans the entire validation set
		def valid_score():
			return [valid_score_i(i) for i in xrange(n_valid_batches + 1)]

		# Create a function that scans the entire test set
		def test_score():
			return [test_score_i(i) for i in xrange(n_test_batches + 1)]
		
		# Create a function that generates all the predicted y and y from train set
		def predict_train_y():
			return [predict_train(i) for i in xrange(n_train_batches + 1)]

		# Create a function that generates all the predicted y and y from valid set
		def predict_valid_y():
			return [predict_valid(i) for i in xrange(n_valid_batches + 1)]

		# Create a function that generates all the predicted y and y from test set
		def predict_test_y():
			return [predict_test(i) for i in xrange(n_test_batches + 1)]

		def new_representation_X_pretrain(hidden_layer):
			new_representation_pretrain = theano.function(
				inputs = [index],
				outputs = self.sigmoid_layers[hidden_layer].linear_output,
				givens = {
					self.x: pretrain_set_x[
						index: index + 1
					]
				}
			)
			pretrain_lines = pretrain_set_x.get_value(borrow=True).shape[0]
			return [new_representation_pretrain(i) for i in xrange(pretrain_lines)]
		
		def new_representation_X_train(hidden_layer):
			new_representation_train = theano.function(
				inputs = [index],
				outputs = self.sigmoid_layers[hidden_layer].linear_output,
				givens = {
					self.x: train_set_x[
						index: index + 1
					]
				}
			)
			train_lines = train_set_x.get_value(borrow=True).shape[0]
			return [new_representation_train(i) for i in xrange(train_lines)]

		def new_representation_X_valid(hidden_layer):
			new_representation_valid = theano.function(
				inputs = [index],
				outputs = self.sigmoid_layers[hidden_layer].linear_output,
				givens = {
					self.x: valid_set_x[
						index: index + 1
					]
				}
			)
			valid_lines = valid_set_x.get_value(borrow=True).shape[0]
			return [new_representation_valid(i) for i in xrange(valid_lines)]

		def new_representation_X_test(hidden_layer):
			new_representation_test = theano.function(
				inputs = [index],
				outputs = self.sigmoid_layers[hidden_layer].linear_output,
				givens = {
					self.x: test_set_x[
						index: index + 1
					]
				}
			)
			test_lines = test_set_x.get_value(borrow=True).shape[0]
			return [new_representation_test(i) for i in xrange(test_lines)]

		def new_representation_X(hidden_layer):
			if new_representation_set_x is None:
				return None
			new_representation = theano.function(
				inputs = [index],
				outputs = self.sigmoid_layers[hidden_layer].linear_output,
				givens = {
					self.x: new_representation_set_x[
						index: index + 1
					]
				}
			)
			new_representation_lines = new_representation_set_x.get_value(borrow=True).shape[0]
			return [new_representation(i) for i in xrange(new_representation_lines)]

		return train_fn, train_score, valid_score, test_score, predict_train_y, predict_valid_y, \
			predict_test_y, new_representation_X_pretrain, new_representation_X_train, \
			new_representation_X_valid, new_representation_X_test, new_representation_X


def run_SdA(datasets, numpy_rng, n_ins, n_outs, hidden_layers_sizes,
			corruption_levels, finetune_lr=0.1,
			pretraining_epochs=20, pretrain_lr=0.001, training_epochs=100,
			batch_size=10, activation = T.nnet.sigmoid, autoencoder_in_activation = T.nnet.sigmoid,
			autoencoder_reconstruction_activation = T.nnet.sigmoid,
			save_pretrain_new_representation = False, save_train_new_representation = False,
			save_valid_new_representation = False, save_test_new_representation = False
		   ):

	results = {}
	results['pretrain_log'] = []
		# results['pretrain_log'].append({'layer':, 'epoch':, 'cost':})
	results['train_log'] = []
		# results['train_log'].append({'epoch':, 'minibatch_index':, 'n_train_batches':, 
		#		'train_score_error':, 'valid_score_error':, 'test_score_error':})
	
	results['summary'] = {'epoch_best_test_score': None, 
			'best_test_score': None,
			'epoch_best_valid_score': None,
			'best_valid_score': None,
			'test_score_best_valid_score': None}

	results['final_predictions'] = {'train': None, 'valid': None, 'test': None}

	results['new_representations'] = {'pretrain': None, 'train': None, 'valid': None, 'test': None,
			'new_representation': None}
 
	pretrain_set_x, pretrain_set_y = datasets[0]
	train_set_x, train_set_y = datasets[1]
	valid_set_x, valid_set_y = datasets[2]
	test_set_x, test_set_y = datasets[3]
	new_representation_set_x, _ = datasets[4]

	# compute number of minibatches for pretraining, training, validation and testing
	n_pretrain_batches = pretrain_set_x.get_value(borrow=True).shape[0]
	n_pretrain_batches /= batch_size
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_valid_batches /= batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_test_batches /= batch_size
	
	print '......... building the model'
	# construct the stacked denoising autoencoder class
	sda = SdA(
		numpy_rng=numpy_rng,
		n_ins=n_ins,
		hidden_layers_sizes=hidden_layers_sizes,
		n_outs=n_outs,
		activation = activation,
		autoencoder_in_activation = autoencoder_in_activation,
		autoencoder_reconstruction_activation = autoencoder_reconstruction_activation
	)

	#########################
	# PRETRAINING THE MODEL #
	#########################
	print '......... getting the pretraining functions'
	pretraining_fns = sda.pretraining_functions(pretrain_set_x=pretrain_set_x,
												batch_size=batch_size)
	print '......... pre-training the model'
	start_time = time.clock()
	## Pre-train layer-wise
	for i in range(sda.n_layers):
		# go through pretraining epochs
		for epoch in range(pretraining_epochs):
			# go through the training set
			c = []
			for batch_index in range(n_pretrain_batches):
				c.append(pretraining_fns[i](index=batch_index,
					corruption=corruption_levels[i], lr=pretrain_lr))
			m = np.mean(c)
			print '............ pre-training layer %i, epoch %d, cost %f' % (i, epoch, m)
			results['pretrain_log'].append({'layer': i, 'epoch': epoch, 'cost': m})

	end_time = time.clock()

	print ('......... the pretraining ran for %.2fm' % ((end_time - start_time) / 60.))

	# get the training, validation, testing, and additional functions for the model
	print '......... getting the finetuning functions'
	
	train_fn, model_score_on_train, model_score_on_valid, model_score_on_test, \
		predict_train, predict_valid, predict_test, new_representation_X_pretrain, \
		new_representation_X_train, new_representation_X_valid, \
		new_representation_X_test, new_representation_X = sda.build_finetune_functions(
			datasets=datasets,
			batch_size=batch_size,
			learning_rate=finetune_lr
		)

	########################
	# FINETUNING THE MODEL #
	########################

	print '......... finetunning the model'
	# early-stopping parameters
	patience = 10 * n_train_batches  # look as this many examples regardless
	patience_increase = 2.	# wait this much longer when a new best is
							# found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_valid_score = np.inf
	best_test_score = np.inf
	test_score_best_valid = np.inf
	test_score = 0.
	start_time = time.clock()

	done_looping = False
	epoch = 0

	b_iter = 0
	while (epoch < training_epochs) and (not done_looping):
		epoch = epoch + 1
		overall_cost = 0.
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_fn(minibatch_index)
			overall_cost += minibatch_avg_cost
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				train_score_error = None
				valid_loss = model_score_on_valid()
				valid_score_error = np.nanmean(valid_loss) * 100.
				test_score_error = None

				print('............ epoch %i, minibatch %i/%i, validation error %f %%,'
						' total cost %f' % (epoch, minibatch_index + 1, 
						n_train_batches, valid_score_error, overall_cost/(iter+1)))
				
				# if we got the best validation score until now
				if valid_score_error < best_valid_score:
					# saving params from the best model so far
					best_model_params = []
					for param in sda.params:
						best_model_params.append(param.get_value(borrow = True))

					# improve patience if loss improvement is good enough
					if valid_score_error < best_valid_score * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# test it on the test set and on train set
					train_loss = model_score_on_train()
					train_score_error = np.nanmean(train_loss) * 100.
					test_losses = model_score_on_test()
					test_score_error = np.nanmean(test_losses) * 100.
					print(('............... epoch %i, minibatch %i/%i, train error of best '
								'model %f %%, test error of best model %f %%') %
								(epoch, minibatch_index + 1, n_train_batches,
								train_score_error, test_score_error))
					
					# save best validation score and epoch
					if valid_score_error < best_valid_score:
						best_valid_score = valid_score_error
						best_valid_epoch = epoch
						b_iter = iter
						test_score_best_valid = test_score_error
					
					# save best test score and epoch
					if test_score_error < best_test_score:
						best_test_score = test_score_error
						best_test_iter = iter
						best_test_epoch = epoch

				result = {}
				result['epoch'] = epoch
				result['minibatch_index'] = minibatch_index
				result['n_train_batches'] = n_train_batches
				result['train_score_error'] = train_score_error
				result['valid_score_error'] = valid_score_error
				result['test_score_error'] = test_score_error
				results['train_log'].append(result)
			if patience <= iter:
				 done_looping = True
				 break

	# setting params for the best model saved in finetune process
	param_index = 0
	for param in sda.params:
		param.set_value(best_model_params[param_index], borrow = True)
		param_index += 1
	
	end_time = time.clock()
	print (
		(
			'......... optimization complete with best validation score of %f %%, '
			'on iteration %i, epoch %i, '
			'with test performance %f %%'
			'. The best test score was %f %% on iteration %i, epoch %i'
		)
		% (best_valid_score, b_iter + 1, best_valid_epoch, test_score_best_valid, 
			best_test_score, best_test_iter + 1, best_test_epoch)
	)
	results['summary'] = {
			'epoch_best_test_score': best_test_epoch, 
			'best_test_score': best_test_score,
			'epoch_best_valid_score': best_valid_epoch,
			'best_valid_score': best_valid_score,
			'test_score_best_valid_score': test_score_best_valid
			}


	print ('......... the training code for file ran for %.2fm' % ((end_time - start_time) / 60.))

	if save_pretrain_new_representation:
		results['new_representations']['pretrain'] = new_representation_X_pretrain
	if save_train_new_representation:
		results['new_representations']['train'] = new_representation_X_train
	if save_valid_new_representation:
		results['new_representations']['valid'] = new_representation_X_valid
	if save_test_new_representation:
		results['new_representations']['test'] = new_representation_X_test
	if new_representation_set_x is not None:
		results['new_representations']['new_representation'] = new_representation_X

	results['final_predictions'] = {'train': predict_train(), 'valid': predict_valid(),
			'test': predict_test()}

	return results

def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
				pretrain_lr=0.001, training_epochs=1000,
				dataset='mnist.pkl.gz', batch_size=100):
	
	ds = load_data(dataset)
	datasets = [ds[0], ds[0], ds[1], ds[2], (None, None)]

	result = run_SdA(datasets = datasets,
		numpy_rng = np.random.RandomState(89677),
		n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10,
		batch_size = batch_size, finetune_lr = finetune_lr, pretrain_lr = pretrain_lr,
		pretraining_epochs = pretraining_epochs, training_epochs = training_epochs,
		corruption_levels = [.1, .2, .3]
	)
	
	print "pretrain length: {0}, train length: {1}, last train log: {2}".format(
		len(result['pretrain_log']), len(result['train_log']), 
		result['train_log'][len(result['train_log']) - 1])

if __name__ == '__main__':
	print "Testing SdA"
	test_SdA()
