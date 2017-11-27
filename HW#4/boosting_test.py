import matplotlib.pyplot as plt


# x = [0.001, 0.01, 0.1, 1, 10]

x1 = [1163./2000., 1163./2000., 1163./2000., 1163./2000., 1163./2000., 1163./2000.]
x2 = [1301./2000., 1372./2000., 1372./2000, 1372./2000, 1372./2000, 1372./2000]
x3 = [1455./2000., 1540./2000., 1540/2000., 1540/2000., 1540/2000., 1540/2000.]

plt.plot(x1, marker = 'o')
plt.plot(x2, marker = '<')
plt.plot(x3, marker = '>')

plt.xlabel("Number of boosting iterations", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("DT (Boosting) Testing", fontsize = 25)
plt.grid(True)
plt.legend(['Depth = 1', 'Depth = 2', 'Depth = 3'])


plt.show()


