import matplotlib.pyplot as plt


# x = [0.001, 0.01, 0.1, 1, 10]

x1 = [1163./2000., 1163./2000., 1163./2000., 1163./2000., 1163./2000., 1163./2000.]
x2 = [1301./2000., 1372./2000., 1372./2000, 1372./2000, 1372./2000, 1372./2000]
x3 = [1455./2000., 1540./2000., 1540/2000., 1540/2000., 1540/2000., 1540/2000.]

plt.plot(x1, marker = o)
plt.plot(x2, marker = o)
plt.plot(x3, marker = o)

plt.xlabel("Number of trees", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("DT (Boosting) Testing", fontsize = 25)
plt.grid(True)
plt.legend(['Depth = 1', 'Depth = 2', 'Depth = 3', 'Depth = 4', 'Depth = 5'])


plt.show()


