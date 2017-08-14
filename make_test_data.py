import numpy as np
import sklearn.datasets as skl
import matplotlib.pyplot as plt
import pickle


data, cl = skl.make_blobs(n_samples=1000,centers=3)

def get_data(k):
    x = []
    y = []
    for i in range(cl.shape[0]):
        if cl[i]==k:
            x.append(data[i, 0])
            y.append(data[i, 1])
    return x, y

for k in range(3):
    x, y = get_data(k)
    v = np.random.rand(3,1)
    plt.scatter(x, y,c = tuple(v[:, 0])) #, s=area, c=colors, alpha=0.5)

plt.show()
pickle.dump(data, open("data.pickle","wb"))

#np.savetxt("data.csv", data, fmt='%.18e', delimiter=',')

m0 = np.mean(data[:,0])
m1 = np.mean(data[:,1])
weights = []
for i in range(data.shape[0]):
    weights.append(10 if data[i,0]>m0 else 1)

pickle.dump(weights, open("weights.pickle","wb"))
