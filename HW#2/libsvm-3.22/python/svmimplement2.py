from svmutil import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt


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

	# def plotter (self, linear, c_param):


	def classifier_train (self, linear, c_param, degree):

		# for train
		#use functions from the svm library to learn the classifier using SVM
		parameter = "-t 1 -d {} -c {} -b 1".format(degree, c_param)
		# print parameter
		prob = svm_problem(self.train_list_letter, self.train_list_word)
		m = svm_train(prob, parameter)

		p_label, p_acc, _ = svm_predict(self.train_list_letter, self.train_list_word, m, '-b 1')

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
		_, p_acc, _ = svm_predict (y, x, m, '-b 1')

		return p_acc, number



def main():
	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"

	# C = [0.1]
	# set of values of C
	C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	validate = 1
	# D = [2]
	# set of degrees
	D = [1, 2, 3, 4]
	accuracy_train = np.zeros(shape = (len(D), len(C)), dtype = float)
	linear = SVM(path1 , path2, 2)

	# for training accuracy
	for i in range (len(D)):
		
		for k in range(len(C)):

			m, p = linear.classifier_train(0, C[k], D[i])
			# plot(m)
			svm_save_model('model{}_{}'.format(k+1, i), m)
			accuracy_train[i][k] = p[0]
			# print p
	
	print accuracy_train
	# This plots the training accuracy of the data set in SVM for all the given C's 
	# plt.plot(accuracy_train)




	# for testing accuracy
	accuracy_test = np.zeros(shape = (len(D), len(C)), dtype = float)
	numbers = np.zeros(shape = (len(D), len(C)), dtype = float)
	linear = SVM(path1, path2, 3)
	for i in range (len(D)):

		for k in range(len(C)):

			model_name = "model{}_{}".format((k+1), i)
			p_acc, number = linear.classifier_test(1, model_name)
			# save the accuracies in a list
			accuracy_test[i][k] = p_acc[0]
			numbers[i][k] = number

	print accuracy_test


	# plots the testing accuracy of the dataset
	# plt.plot(accuracy_test)
	print ("Number of SV")
	print numbers


	# for validation accuracy
	accuracy_validation = np.zeros(shape = (len(D), len(C)), dtype = float)
	linear = SVM(path1, path2, 1)
	for i in range(len(D)):

		for k in range(len(C)):

			model_name = "model{}_{}".format((k+1), i)
			p_acc, _ = linear.classifier_test(2, model_name)
			# save the accuracies in a list
			print ("p_acc[o] = {}".format(p_acc[0]))
			# print p_acc[0]
			accuracy_validation[i][k] = p_acc[0]

	print accuracy_validation
	# C_plot = ['0.0001', '0.001', '0.01', '0.1', '1', '10', 100, 1000, 10000]
	# plots the testing accuracy of the dataset
	for i in range (len(D)):
		plt.plot(numbers[i], marker = 'o')
		# plt.plot(accuracy_train[i], marker = 'o')
		# plt.plot(accuracy_validation[i], marker = '>')
		# plt.plot(accuracy_test[i], marker = '<')
		# plt.xticks(['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000', '10000'])	
		plt.xlabel("C's", fontsize = 15)
		plt.ylabel(" Numbers of SV", fontsize = 15)
		plt.title("Support Vectors(SVM) Degree = {}".format(D[i]), fontsize = 25)
		# plt.ylim([0.1, 0.8])
		plt.grid(True)
		# plt.legend(['Training', ' Validation', 'Testing'])
		plt.show()  



if __name__ == '__main__':

	main()


		# compute the training classifier and each time please test its accuracy on the validation data as well as the test data 




