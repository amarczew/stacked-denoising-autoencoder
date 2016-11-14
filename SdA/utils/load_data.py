"""
Created on Fri May	8 14:28:27 2015

@author: amarczew
"""
from django.utils import encoding
import arff
import os
import cPickle
import time
import sys
import theano
import theano.tensor as T

import numpy as np
from common import _FLOATX
		   
def shared_dataset(data_xy, borrow=True):
	""" Function that loads the dataset into shared variables
	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x,
						dtype=_FLOATX),
						borrow=borrow)
	if data_y is None:
		return shared_x, None

	shared_y = theano.shared(np.asarray(data_y,
						 dtype=_FLOATX),
						borrow=borrow)
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, T.cast(shared_y, 'int32')

def load_file(dataset_path):
	st = time.clock()
	print("...... loading file: " + dataset_path)
	data_dir, data_file = os.path.split(dataset_path)
	dt = None
	pickleFile = dataset_path + ".pickle"
	if not os.path.isfile(pickleFile):
		print("...... loading data from arff file")
		content_unicode = open(dataset_path, 'rb').read().decode('utf-8')
		content = encoding.smart_str(content_unicode, encoding='ascii', errors='ignore')
		data = arff.load(content)
		dt = data['data']
		print("...... saving data in pickle format")
		f = open(pickleFile, 'wb')
		cPickle.dump(dt, f, protocol = cPickle.HIGHEST_PROTOCOL)
		f.close()
	else:
		print("...... loading data from pickle")
		f = file(pickleFile, 'rb')	
		dt = cPickle.load(f)
		f.close()
	print ('...... loading file done: %.2fm' % ((time.clock() - st) / 60.))
	return dt

def split_x_y(data):
	np_data = np.asarray(data)
	y_vector = np_data[:, -1].flatten()
	# Ignoring first column
	return np_data[:, 1:-1], y_vector 
	 
def load_data(dataset_path, class_map = None):
	data = []
	for filename in dataset_path:
		data.extend(load_file(filename))

	x, y = split_x_y(data)
	if class_map is not None:
		for i in range(len(class_map)):
			y[y == class_map[i]] = i
	else:
		y = None
	return shared_dataset((x, y))
	
def load_data_arff(pretrain, train, valid, test, new_representation = None, class_map = None):
	st = time.clock()
	print "loading all datasets"
	print "... loading pretrain_set"
	pretrain_set = load_data(pretrain)
	print "...... x shape: {0}".format(pretrain_set[0].get_value().shape)	  

	print "... loading train_set"
	train_set = load_data(train, class_map)
	print "...... x shape: {0}".format(train_set[0].get_value().shape)	
	print "... loading valid_set"
	valid_set = load_data(valid, class_map)
	print "...... x shape: {0}".format(valid_set[0].get_value().shape)
	
	print "... loading test_set"
	test_set = load_data(test, class_map)
	print "...... x shape: {0}".format(test_set[0].get_value().shape) 

	new_representation_set = (None, None)
	if new_representation is not None:
		print "... loading new_representation_set"
		new_representation_set = load_data(new_representation)
		print "...... x shape: {0}".format(new_representation_set[0].get_value().shape)

	print ('loading all datasets - done: %.2fm\n' % ((time.clock() - st) / 60.))

	return [pretrain_set, train_set, valid_set, test_set, new_representation_set]
