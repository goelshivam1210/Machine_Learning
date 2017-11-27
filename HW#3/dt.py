'''
author: Shivam Goel
WSUID# 11483916
Reference 
http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

'''
import numpy as np
import math
import random
import time
import csv
from pprint import pprint

class ID3():

	def __init__(self, path1, path2, path3, path4, path5):


		self.train_set_label,self.test_set_label, self.vocabulary, self.validation_set_label = self.parser(path1, path2, path3, path4)
		# calls the make feature set function to make the feature vectors of the input training data
		self.train_set, self.validation_set = self.make_feature_set(path2)
		# calls the make feature set function to make the feature vectors of the input testing data
		self.test_set = self.make_feature_set(path5, split = False)
		self.tree = None
		print len(self.train_set)
	

		
	def parser (self, path1, path2, path3, path4):

		# make the library, library is the list of words in the alphabetical order that doesnt have the stop words
		# First lets extract the stop words
		stop = []
		vocab = []
		label = []
		label2 = []
		f = open(path1)
		for line in f :
			line = line.strip()
			stop.append(line)

		# now make the list of all the words from the training set.
		f = open(path2)
		for line in f:
			line = line.strip()
			# line = line[:-1]
			
			g = line.split(" ")
			if len(g) > 1:

				for i in range(len(g)):
					if g[i] in stop:
						continue
					else:
						if g[i] not in vocab:
							vocab.append(g[i])
		vocab = sorted(vocab)
		print len(vocab)

		# parse the label set (train label set)
		f = open(path3)
		for line in f:
			line = line.strip()
			label.append(line)
		# parse the label set (test label set)
		f = open(path4)
		for line in f:
			line = line.strip()
			label2.append(line)
		
		return np.array(label[:int(0.8*len(label))]), np.array(label2), vocab, np.array(label[int(0.8* len(label)):])

	
	def make_feature_set(self, path, split = True):

		set_t = []
				
		# path of the file of the training  set or testing set
		f = open(path)
		for line in f:
			line = line.strip()
			g = line.split(" ")

			set_temp = []
			for voc in self.vocabulary:
				if voc in g:
					set_temp.append(1)
				else:
					set_temp.append(0)


			set_t.append(set_temp)

		if split is False:
			return np.array(set_t)
		else:
			return np.array(set_t[ :int(0.8*len(set_t))]),np.array(set_t[int(0.8*len(set_t)):]) 
		# after parsing all the lines the variable train_set will be a matrix of shape = nXd, n = number of training examples and d = number of features.

	# return {c: (a==c).nonzero()[0] for c in np.unique(a)}
	def partition (self, a, b, attr):
		# create an empty list for populating the right and left branches of the tree
		x_0 = []
		y_0 = []
		x_1 = []
		y_1 = []
		for i in range (np.shape(b)[0]):
			if  a[i][attr] == 0:
				x_0.append(a[i])
				y_0.append(b[i])
			else:
				x_1.append(a[i])
				y_1.append(b[i])

		return np.array(x_0), np.array(y_0), np.array(x_1), np.array(y_1)


	def entropy (self, s):
		res = 0
		val, counts = np.unique(s, return_counts = True)
		freqs = counts.astype('float64')/len(s)
	
		for p in freqs:
			if p != 0.0:
				# sum of -p log p
				res -= p*np.log2(p)
	
		return res


	def info_gain(self, y, x):

		# here we need to calculate the entropy of all the feature set and that is basically 
		res = self.entropy(y)
	

		val, counts = np.unique(x, return_counts = True)
		freqs = counts.astype('float64') / len(x)

				
		for p, v in zip(freqs, val):
	
			y_new = []
			for i in range (len(x)):
				if x[i] == v:
					# add that label to the list
					y_new.append(y[i])
	

			z = self.entropy(y_new)
	
	
			res -= p * z


		return res

	def purity(self, s):
		# now purity is checked by the label, if the given child nodes examples all are same labels, then it is a pure node. else impure

		return len(set(s)) == 1
	
	
	def create_tree(self): 
		self.tree = self._create_tree(self.train_set, self.train_set_label)

	def _create_tree (self, x, y):
		if self.purity(y):
			return y[0]

		gain = np.array([self.info_gain(y, x_attr) for x_attr in x.T])
		selected_attr = np.argmax(gain)
		tree_dict = {selected_attr :{}}

		(left_x, left_y, right_x, right_y) = self.partition(x, y, selected_attr)
	
		# create the left node of the tree by recursively calling the create_tree function
		tree_dict[selected_attr][0] = self._create_tree(left_x, left_y)
		# create the right node of the tree
		tree_dict[selected_attr][1] = self._create_tree(right_x, right_y)

	
		return tree_dict

	def test_tree(self, dataset, dataset_label):
		pprint(self.tree)

		correct = 0
		for i in range(len(dataset)):
			x = dataset[i]

			y = dataset_label[i]


			y_ = self.predict(x, self.tree)
			if y == y_:
				correct += 1
		return correct, len(dataset_label)

	def predict(self, x, tree):

		# this traverses through the dictionary
		# if the current splitting in the tree reveals a leaf node, ie., not a dictionary
		if not isinstance(tree[tree.keys()[0]][x[tree.keys()[0]]] , dict):
			# return the label of the instance
			return tree[tree.keys()[0]][x[tree.keys()[0]]]
		# recursively again iterate through the dictionary till we find the leaf node
		return self.predict(x, tree[tree.keys()[0]][x[tree.keys()[0]]])

	# def prune_tree(self, tree):

def main():

	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#3/fortunecookiedata/stoplist.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#3/fortunecookiedata/traindata.txt"
	path3 = "/home/goelshivam12/Desktop/ML_Homework/HW#3/fortunecookiedata/trainlabels.txt"
	path4 = "/home/goelshivam12/Desktop/ML_Homework/HW#3/fortunecookiedata/testdata.txt"
	path5 = "/home/goelshivam12/Desktop/ML_Homework/HW#3/fortunecookiedata/testlabels.txt"
	# instantiate the class
	decision = ID3(path1, path2, path3, path5, path4)
	decision.create_tree()
	accurate, total = decision.test_tree(decision.validation_set, decision.validation_set_label)
	# print accurate, total
	accurate2, total2 = decision.test_tree(decision.test_set, decision.test_set_label)
	print (float(accurate2)/total2) * 100

	

if __name__ == '__main__':

	main()
		
