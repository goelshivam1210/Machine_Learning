import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import table
import sys


'''
[1666 1240 1020  842  834  748  696  638  630  582  534  524  510  506  468
  478  462  482  464  510]
[10077  8799  7375  7171  6969  6811  6801  6558  6594  6738  6578  6533
  6930  6865  6605  6737  6917  6859  7058  6945]
4617
47535
'''
# mistakes_train = [1666, 1240, 1020,  842,  834,  748,  696,  638,  630,  582,  534,  524,  510,  506,  468,
#   478,  462,  482,  464,  510]
# mistakes_test = [22039, 18947, 17837, 22046, 18813,  15068, 15719, 15946, 14268, 14849, 14361, 13808, 13381,
  # 14228,  13442, 13722, 13572, 14281, 13772]
mistakes_test = [22039, 18947, 17837, 16557, 22046, 18813, 15068, 15719, 15946, 14268, 14849, 14361, 13808, 13381, 14228, 13442, 13722, 13572, 14281, 13772]


# print mistakes_test

mistakes_train = [2219, 1519, 1237, 1037, 862, 788, 719, 711, 613, 523, 513, 478, 455, 393, 397, 
365, 350, 321, 207, 226]
print (len(mistakes_test))
print (len(mistakes_train))
plt.matplotlib.style.use('ggplot')
plt.bar(mistakes_train, marker = 'o')
plt.xlabel("Iterations", fontsize = 15)
plt.ylabel(" No of mistakes", fontsize = 15)
plt.title("Mistakes(Kernelized Perceptron (Multi))", fontsize = 25)
plt.show()




# mistakes_test = []
total_train = 4617
total_test = 47535

mistakes_train = np.array(mistakes_train, dtype = float)
mistakes_test = np.array(mistakes_test, dtype = float)

print mistakes_test
mistakes_train = total_train - mistakes_train 
mistakes_train = mistakes_train / total_train


mistakes_test = total_test - mistakes_test
mistakes_test = mistakes_test / total_test
# print mistakes_test
# print mistakes_train

# print mistakes_test
# print mistakes_train

# import sys
# sys.exit()

# mistake = plt.figure(1)
# plt.plot(mistakes_train, marker = 'o')
# plt.xlabel("Iterations", fontsize = 15)
# plt.ylabel(" Mistakes", fontsize = 15)
# plt.title("Number of Mistakes in Training (Kernelized Perceptron(Binary))", fontsize = 25)
# # plt.ylim([0.1, 0.8])
# plt.grid(True)
# # plt.legend(['Training', ' Testing', 'Validation'])
# plt.show()

plt.matplotlib.style.use('ggplot')
# accuracy = plt.figure(2)
plt.plot(mistakes_train , marker = 'o')
plt.plot(mistakes_test, marker = '>')
plt.xlabel("Iterations", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("Training vs Testing (Kernelized Perceptron (Multi))", fontsize = 25)
# plt.ylim([0.1, 0.8])
plt.grid(True)
plt.legend(['Training', ' Testing'])
plt.show()  


