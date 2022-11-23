import matplotlib.pyplot as plt
import numpy as np 
import sys


lines = []
dim = 5
if len(sys.argv) > 1:
    dim = int(sys.argv[1])
print("dimx: ",dim)

# Xfile = open("Xs.txt",'r')
# Xstr = Xfile.read()
# X = np.array(eval(Xstr))
# X = X.reshape((int(len(X)/dim),dim))
X = np.loadtxt("./data/X"+str(dim)+".txt")

lines+=plt.plot(X[:,2:4],'.',label="X")
# plt.show()

Zfile = open("./data/X_pred"+str(dim)+".txt",'r')
Zstr = Zfile.read()
Z = np.array(eval(Zstr))
Z = Z.reshape((int(len(Z)/dim),dim))

lines+=plt.plot(Z[:,2:4],'*',label="X_pred")

Yfile = open("./data/X1_pred"+str(dim)+".txt",'r')
Ystr = Yfile.read()
Y = np.array(eval(Ystr))
Y = Y.reshape((int(len(Y)/dim),dim))

lines+=plt.plot(Y[:,6:8],'o',label="X1_pred")

labels = [l.get_label() for l in lines]
plt.legend(lines, labels)
# plt.show()
plt.savefig('./data/compareX.pdf')
