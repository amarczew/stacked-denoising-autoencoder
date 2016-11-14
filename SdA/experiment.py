import sys
import os
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import SdA
from logistic_sgd import load_data

from logger import Logger
from activations import get_activation_function

from texttable import Texttable
from common import _FLOATX, str2bool
from utils.load_data import load_data_arff


class Experiment(object):

	def __init__(self, path_config_file):
		
		self.cont_configs = 0

		# processing input config files
		self.process_config_file(path_config_file)

		# directories
		self.logs_dir					= "%s/logs/" % \
												(self.options['output-directory'])
		self.config_logs_dir			= "%s/config-logs/" % \
												(self.options['output-directory'])
		self.new_representations_dir	= "%s/new-representations/" % \
												(self.options['output-directory'])
		self.predictions_dir			= "%s/predictions/" % \
												(self.options['output-directory'])
		self.experiment_summaries_dir	= "%s/experiment-summaries/" % \
												(self.options['output-directory'])

		self.options['cache-directory'] = self.options['output-directory']
		self.params_pretrained_dir		= "%s/sda-params/params-pretrained/pretrained-" % \
												(self.options['cache-directory'])
		self.params_trained_dir			= "%s/sda-params/params-trained/trained-" % \
												(self.options['cache-directory'])

		# running experiments
		self.initialize_output_log_files()
		self.run()
		self.close_output_log_files()

	def initialize_output_log_files(self):
		# Creating log files
		sys.stdout = Logger("%s%s.log" % (self.logs_dir, self.options['execution-name']))
		header = os.path.isfile("%s%s.log" % 
			(self.experiment_summaries_dir, self.options['execution-name']))
		self.log_experiment_summaries = open("%s%s.log" % 
			(self.experiment_summaries_dir, self.options['execution-name']), "a", 0)

		# Initialize headers
		if not header:
			print >> self.log_experiment_summaries, "experiment_config_name | batch_size | " + \
				"pretrain_lr | pretraining_epochs | " + \
				"optimization_autoencoder | finetune_lr | training_epochs | " + \
				"optimization_logistic | #hidden_layers | hidden_layers_sizes | " + \
				"corruption_levels | autoencoder_in_activation | " + \
				"autoencoder_reconstruction_activation | activation | epoch_best_test_score | " + \
				"best_test_score |	epoch_best_valid_score | best_valid_score | " + \
				"test_score_best_valid_score"


	def close_output_log_files(self):
		sys.stdout.reboot()
		self.log_experiment_summaries.close()


	def run(self): 
		print("Starting at %s the execution %s" % (time.strftime("%Y-%m-%d %H:%M"), 
			self.options['execution-name']))
		begin_time = time.clock()

		self.datasets = self.process_and_load_dataset(self.options['dataset-config-file'], 
				self.options['class-map'])
		
		print("... running experiments")
		for config in sorted(self.configs.keys()):
			for replication in range(self.options['replications']):
				begin_config_time = time.clock()
				experiment_config_name = "{0};{1};rep-{2}.{3}".format(
						self.options['execution-name'], 
						config, 
						replication + 1, self.options['replications'])

				print("...... config {0}".format(experiment_config_name))
				
				"""
				pretrained_pickle_filename = self.params_pretrained_dir + \
												experiment_config_name + ".pickle"
				trained_pickle_filename    = self.params_trained_dir + \
												experiment_config_name + ".pickle"
				"""

				result = SdA.run_SdA(datasets = self.datasets,
					numpy_rng = self.options['numpy_rng'],
					n_ins = self.options['input-size'], n_outs = len(self.options['class-map']), 
					batch_size = self.options['batch-size'],
					finetune_lr = self.options['learning-finetune'], 
					pretrain_lr = self.options['learning-pretrain'],
					pretraining_epochs = self.options['pretrain-epochs'], 
					training_epochs = self.options['train-epochs'],
					hidden_layers_sizes = self.configs[config].hidden_layers_sizes,
					corruption_levels = self.configs[config].corruption_levels, 
					activation = self.options['activation_obj'],
					autoencoder_in_activation = self.options['autoencoder-in-activation_obj'],
					autoencoder_reconstruction_activation = 
						self.options['autoencoder-reconstruction-activation_obj'],
					save_pretrain_new_representation = 
						self.options['save_pretrain_new_representation'],
					save_train_new_representation = 
						self.options['save_train_new_representation'],
					save_valid_new_representation = 
						self.options['save_valid_new_representation'],
					save_test_new_representation = 
						self.options['save_test_new_representation']
				)
				
				self.process_result_experiment_config(experiment_config_name, 
					self.configs[config], result)
				print("...... config ran for %.2fm" % ((time.clock() - begin_config_time)
					/ 60.))
		print ("==> Total Execution Time: %.2fm\n" % ((time.clock() - begin_time) / 60.))

	def process_result_experiment_config(self, experiment_config_name, config, result):
		self.save_new_representations(experiment_config_name, config,
				**result['new_representations'])
		self.save_experiment_config_summary(experiment_config_name, config,
				result['summary'])

	def save_experiment_config_summary(self, experiment_config_name, config, summary):
		print >> self.log_experiment_summaries, "{0} | {1} | {2} | {3} | {4} | {5} | {6} | " \
			"{7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} | " \
			"{18}".format(experiment_config_name, self.options['batch-size'], 
			self.options['learning-pretrain'], self.options['pretrain-epochs'], 
			self.options['optimization-autoencoder'], self.options['learning-finetune'],
			self.options['train-epochs'], self.options['optimization-logistic'],
			len(config.hidden_layers_sizes), config.hidden_layers_sizes, config.corruption_levels,
			self.options['autoencoder-in-activation'], 
			self.options['autoencoder-reconstruction-activation'], self.options['activation'],
			summary['epoch_best_test_score'], summary['best_test_score'], 
			summary['epoch_best_valid_score'], summary['best_valid_score'], 
			summary['test_score_best_valid_score'])


	def process_config_file(self, path_config_file):
		self.options = {}
		self.configs = {}
		with open(path_config_file) as fp:
			for line in fp:
				option = line.strip().split(":")
				if len(option) < 2 or option[0][0] == '#':
					continue
				if option[0][:6] == "config":
					self.cont_configs += 1
					self.configs[option[0]] = self.Config(option[1])
				elif option[0] == "class-map":
					self.options[option[0]] = option[1].split(",")
				else:
					self.options[option[0]] = option[1]
		
			# numpy random generator
			if int(self.options['replications']) == 1:
				self.options['numpy_rng'] = np.random.RandomState(89677)
			else:
				self.options['numpy_rng'] = np.random.RandomState(None)
			
			self.options['execution-name'] = self.options['dataset-name-file']	+ ";" + \
												self.options['experiment-name']

			self.options['input-size'] = int(self.options['input-size'])
			self.options['batch-size'] = int(self.options['batch-size'])
			self.options['replications'] = int(self.options['replications'])
			self.options['pretrain-epochs'] = int(self.options['pretrain-epochs'])
			self.options['train-epochs'] = int(self.options['train-epochs'])
					
			self.options['learning-finetune'] = float(self.options['learning-finetune']) 
			self.options['learning-pretrain'] = float(self.options['learning-pretrain'])

			self.options['save_pretrain_new_representation'] = \
					str2bool(self.options['save_pretrain_new_representation'])
			self.options['save_train_new_representation'] = \
					str2bool(self.options['save_train_new_representation'])
			self.options['save_valid_new_representation'] = \
					str2bool(self.options['save_valid_new_representation'])
			self.options['save_test_new_representation'] = \
					str2bool(self.options['save_test_new_representation'])

			self.options['activation_obj'] = get_activation_function(self.options['activation'])
			self.options['autoencoder-in-activation_obj'] = get_activation_function(
					self.options['autoencoder-in-activation'])
			self.options['autoencoder-reconstruction-activation_obj'] = get_activation_function(
					self.options['autoencoder-reconstruction-activation'])


	def process_and_load_dataset(self, path_dataset_file, class_map = None):
		if path_dataset_file == "MNIST":
			ds = load_data('mnist.pkl.gz')
			datasets = [ds[0], ds[0], ds[1], ds[2], (None, None)]
			return datasets
		else:
			files = {}
			files['ds_pretrain'] = []
			files['ds_train'] = []
			files['ds_valid'] = []
			files['ds_test'] = []
			files['ds_new_representation'] = []
			with open(path_dataset_file) as fp:
				for line in fp:
					ds = line.strip().split(":")
					if len(ds) < 2 or ds[0][0] == '#':
						continue
					files[ds[0]].append(ds[1])
				if len(files['ds_new_representation']) == 0:
					files['ds_new_representation'] = None
					
			return load_data_arff(files['ds_pretrain'], files['ds_train'], files['ds_valid'], 
					files['ds_test'], files['ds_new_representation'], class_map)


	def save_new_representations(self, experiment_config_name, config, 
			pretrain = None, train = None, valid = None, test = None, new_representation = None):
		if pretrain is not None:
			self.save_new_representation(experiment_config_name, config, 
					"pretrain", pretrain)
		if train is not None:
			self.save_new_representation(experiment_config_name, config, 
					"train", train)
		if valid is not None:
			self.save_new_representation(experiment_config_name, config, 
					"valid", valid)
		if test is not None:
			self.save_new_representation(experiment_config_name, config,  
					"test", test)
		if new_representation is not None:
			self.save_new_representation(experiment_config_name, config, 
					"new_representation", new_representation)
		

	def save_new_representation(self, experiment_config_name, config, dataset_name, 
			new_representation_X):
		print "......... saving new representations for {0} dataset  in the file " \
				"{1};{0};layer-N.txt, directory {2}".format(dataset_name, 
					experiment_config_name, self.new_representations_dir)
		new_representation_files = []
		# Creating new representation files
		for i in range(len(config.hidden_layers_sizes)):
			f = open("{0}{1};{2};layer-{3}.{4}.txt".format(
				self.new_representations_dir, experiment_config_name, 
				dataset_name, i + 1, len(config.hidden_layers_sizes)), "w")
			new_representation_files.append(f)

		for i in range(len(config.hidden_layers_sizes)):
			print "............ layer {0}.{1}".format(i + 1, 
				len(config.hidden_layers_sizes))
			new_repre = new_representation_X(i)
			for line in new_repre:
				for value in line[0]:
					print >> new_representation_files[i], str(value) + ",",
				print >> new_representation_files[i]

		# Closing new representation files
		for f in new_representation_files:
			f.close()



	class Config():
		def __init__(self, options):
			ops = options.split(",");
			qnt_hidden_layers = int(ops[0])
			self.hidden_layers_sizes = []
			self.corruption_levels = []
			for z in xrange(qnt_hidden_layers):
				self.hidden_layers_sizes.append(int(ops[1 + z]))
				self.corruption_levels.append(float(ops[1 + z + qnt_hidden_layers]))

