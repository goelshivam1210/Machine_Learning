from svmutil import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import table
from pandas_ml import ConfusionMatrix

'''
support_vectors = model.get_SV() 
svm_save_model('heart_scale.model', m)
m = svm_load_model('heart_scale.model')
nr_sv = model.get_nr_sv()
'''

class SVM():
	def __init__(self, path1, path2, validate):

		self.train_list_letter, self.train_list_word, _ = self.parse(path1, validate)
		self.validate_list_letter, self.validate_list_word, _ = self.parse(path1, validate)
		self.test_list_letter, self.test_list_word, _ = self.parse(path2, validate)



	def parse(self, file, validate):

		# list of the x's


		list_word = []
		# list of the y's
		list_letter = []
		#holds the whole list
		list_main = []

		f = open(file)		#code for parsing

		for line in f:
			# to remove the blank lines
			if line.strip():
				g = line.split("\t")
				#convert each element in the list to a list
				#extract the label in each x
				letter = g[2]
				# map the letter to a number 
				# a -> 1, b -> 2, c -> 3, ...
				letter = ord(letter) - 97
				#extract the x for each input data
				word = list(g[1][2:])
				word = (map(int, word))
				# print word
				# if validation set is being computed
				list_word.append(word)
				list_letter.append(letter)
				list_main.append(g)
		# If validaion is being computed
		if validate == 1:
			#only return the 20% of the data
			return list_letter[int(0.8*(len(list_word))):], list_word[int(0.8*(len(list_word))):], list_main 

		# if training is being computed
		elif validate == 2:
			# oonly return firsr 80 percent of the data
			return list_letter[ : int(0.8*(len(list_word)))], list_word[ :int(0.8*(len(list_word)))], list_main

		else:
			return list_letter, list_word, list_main


	def classifier_train (self, linear, c_param):

		# for train
		#use functions from the svm library to learn the classifier using SVM
		# print self.train_list_word[0:2]
		# print ""
		# print ""
		# print ""
		# print "           {}".format(self.train_list_letter[0:2])
		parameter = "-t 0 -c {} -b 1".format(c_param)
		# print parameter
		prob = svm_problem(self.train_list_letter, self.train_list_word)
		# print parameter
		# param = svm_parameter('-t {} -c {}').format(linear, c_param)
		m = svm_train(prob, parameter)

		_, p_acc, _ = svm_predict(self.train_list_letter, self.train_list_word, m, '-b 1')
		# plot the p_acc


		# this returns the model and the accuracy of the traininng set
		return m, p_acc


	def classifier_test(self, type, model_name):
		# for testing set
		if type == 1:
			y, x = self.test_list_letter, self.test_list_word
		# for validation
		else:
			y, x = self.validate_list_letter, self.validate_list_word		

		# Now we need to load the learned model and check the accuracy in validation or testing data

		m = svm_load_model(model_name)
		number = m.get_nr_sv()
		y_label, p_acc, _ = svm_predict (y, x, m, '-b 1')
		# print y_label
		y_pred = pd.Series(y_label, name = 'Predicted')
		y_actual = pd.Series(self.test_list_letter, name = 'Actual')
		df_CF = ConfusionMatrix(y_actual, y_pred)
		df_CF.print_stats()
		df_CF.plot(color = 'b')
		plt.show()

		return p_acc, number

		




def main():
	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"

	# C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	C = [0.1]	
	accuracy_train = []
	validate = 3
	linear = SVM(path1 , path2, 3)

	# for training accuracy
	for i in range (len(C)):
		m, p = linear.classifier_train(0, C[i])
		# plot(m)
		svm_save_model('model{}'.format(i+1), m)
		accuracy_train.append(p[0])
		print p
	print accuracy_train
	# This plots the training accuracy of the data set in SVM for all the given C's 
	# plt.plot(accuracy_train)



	# for testing accuracy
	accuracy_test = []
	numbers = []
	linear = SVM(path1, path2, 3)
	for i in range(len(C)):
		model_name = "model{}".format(i+1)
		p_acc, number = linear.classifier_test(1, model_name)
		# save the accuracies in a list
		accuracy_test.append(p_acc[0])
		numbers.append(number)


	# plots the testing accuracy of the dataset
	plt.plot(accuracy_test)
	print ("Number of SV")
	print numbers


	# for validation accuracy
	# accuracy_validation = []
	# linear = SVM(path1, path2, 1)
	# for i in range(len(C)):
	# 	model_name = "model{}".format(i+1)
	# 	p_acc = linear.classifier_test(2, model_name)
	# 	# save the accuracies in a list
	# 	accuracy_validation.append(p_acc[0])

	# # plots the testing accuracy of the dataset
	# plt.plot(accuracy_validation)

	plt.xlabel("C's", fontsize = 15)
	plt.ylabel(" Accuracy", fontsize = 15)
	plt.title("Accuracy Curve (SVM)", fontsize = 25)
	# plt.ylim([0.1, 0.8])
	plt.grid(True)
	plt.legend(['Training', ' Testing', 'Validation'])
	plt.show()  


if __name__ == '__main__':

	main()






