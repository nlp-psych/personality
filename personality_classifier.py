# label a status update for personality

from sklearn.model_selection import cross_val_score
from sklearn import svm
import feature_extractor as fe
import numpy
import re
import sys
import csv
import codecs

datafile = sys.argv[1]
conffile = sys.argv[2]

ndata = -1 # for testing feature extraction: optional arg to control how much of data to use. won't work for testing classification because it just takes the first n -- all one class
if len(sys.argv) > 3:
	ndata = int(sys.argv[3])

def load_data():
	with codecs.open(datafile, encoding="latin-1") as f:
		return list(csv.reader(f))[1:ndata]

def load_conf_file():
	conf = set(line.strip() for line in open(conffile))
	return conf

def predict_trait(X, Y):
	scores = cross_val_score(svm.SVC(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

if __name__ == "__main__":
	data = load_data()
	conf = load_conf_file()
	labels = numpy.asarray([line[7:11] for line in data]).T.tolist()
	print(numpy.array(labels).shape)
	features = fe.extract_features([line[1] for line in data], conf)	# this will only pass the status update text to the feature extractor

	for i in range(len(labels)):
		print (predict_trait(features, labels[i]))
