from svmutil import *
import numpy
import math
import random
import matplotlib as plt


'''
 support_vectors = model.get_SV() 
svm_save_model('heart_scale.model', m)
m = svm_load_model('heart_scale.model')



'''

class SVM():
	def __init__(self, path1, path2):

		self.train_list_letter, self.train_list_word, _ = self.parse(path1, validate)
		self.validate_list_letter, self.validate_list_word, _ = self.parse(path1, validate)
		self.test_list_letter, self.test_list_word, _ = self.parse(path2, validate)



	def parse(self, file):
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
				word = np.array(map(float, word))
				# if validation set is being computed
				list_word.append(word)
				list_letter.append(letter)
				list_main.append(g)
		# If validaion is being computed
		if validate == 1:
			#only return the 20% of the data
			return list_word[0.8(len(list_word)):], list_letter[0.8(len(list_word)):]

		# if trainining is being computed
		elif validate == 2:
			# oonly return firsr 80 percent of the data
			return list_word[ :0.8(len(list_word))], list_letter[ : 0.8(len(list_word))]

		else:

			return list_letter, list_word, list_main



	# def plotter (self, linear, c_param):


	def classifier_train (self, linear, param):

		# for train
		#use functions from the svm library to learn the classifier using SVM
		prob = svm_problem(self.train_list_letter, self.train_list_word)
		param = svm_parameter('-t {} -c {}').format(linear, c_param)
		m = svm_train(prob, param)

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
		_, p_acc, _ = svm_predict (y, x, m, '-b 1')

		return p_acc




def main():
	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"

	C = [10**-4,10**-3,10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
	accuracy_train = []
	linear = SVM(path1 , path2)

	# for training accuracy
	for i in range (len(C)):
		m, p = linear.classifier_train(0, C[i])
		plot(m)
		svm_save_model('model{}'.format(i+1), m)
		accuracy_train.append(p)
	# This plots the training accuracy of the data set in SVM for all the given C's 
	plt.plot(accuracy_train)


	# for testing accuracy
	accuracy_test = []
	for i in range(len(C)):
		model_name = "model{}".format(i+1)
		p_acc = linear.classifier_test(1, model_name)
		# save the accuracies in a list
		accuracy_test.append(p_acc)

	# plots the testing accuracy of the dataset
	plt.plot(accuracy_test)


	# for validation accuracy
	accuracy_validation = []
	for i in range(len(C)):
		model_name = "model{}".format(i+1)
		p_acc = linear.classifier_test(2, model_name)
		# save the accuracies in a list
		accuracy_validation.append(p_acc)

	# plots the testing accuracy of the dataset
	plt.plot(accuracy_validation)

	plt.xlabel("C's", fontsize = 15)
	plt.ylabel(" Accuracy", fontsize = 15)
	plt.title("Accuracy Curve (SVM)", fontsize = 25)
	# plt.ylim([0.1, 0.8])
	plt.grid(True)
	plt.legend(['Training', ' Testing', 'Validation'])
	plt.show()  





		# compute the training classifier and each time please test its accuracy on the validation data as well as the test data 




