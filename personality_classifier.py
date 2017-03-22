# label a status update for personality

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm, preprocessing, metrics
import feature_extractor as fe
import numpy
import re
import sys
import csv
import codecs
import argparse
import pickle
from subprocess import call

class_idx = [7, 8, 9, 10, 11]

def load_data(datafile):
	with codecs.open(datafile, encoding="latin-1") as f:
		reader = csv.reader(f)
		return next(reader), list(reader)

def load_conf_file(conffile):
	conf = set(line.strip() for line in open(conffile))
	return conf

def predict_trait(X, Y):
	scores = cross_val_score(svm.SVC(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

def handle_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--datafile', help='file containing data for training or testing', required=True, dest='datafile')
	parser.add_argument('-c', '--conffile', help='file containing list of features to be extracted', dest='conffile', required=True)
	parser.add_argument('-e', '--expdir', help='directory for storing conf file and models associated with this experiment', required=True)
	parser.add_argument('-l', '--load', action='store_true', help='include to load models from <expdir> instead of training new')
	return vars(parser.parse_args())

if __name__ == "__main__":
	args = handle_args()
	print(args)
	header, data = load_data(args['datafile'])
	labels = numpy.asarray([[line[i] for i in class_idx] for line in data]).T.tolist()

	conf = load_conf_file(args['conffile'])
	features = fe.extract_features([line[1] for line in data], conf)	# this will only pass the status update text to the feature extractor

	if not args['load']:
		# train new models, evaluate, store
		for i in range(len(class_idx)):
			trait = header[class_idx[i]]
			clf = svm.SVC().fit(features, labels[i])
			predicted = cross_val_predict(clf, features, labels[i], cv=10)
			print("%s: %.2f" % (header[class_idx[i]], metrics.accuracy_score(labels[i], predicted)))
			with open("%s/%s.pkl" % (args['expdir'], trait), 'wb') as f:
				pickle.dump(clf, f)
	else:
		for i in range(len(class_idx)):
			trait = header[class_idx[i]]
			with open("%s/%s.pkl" % (args['expdir'], trait), 'rb') as f:
				clf = pickle.load(f)
			print("%s: %.2f" % (trait, clf.score(features, labels[i])))
		
