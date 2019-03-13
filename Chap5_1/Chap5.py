# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from display_network import *
from scipy.spatial.distance import cdist

mndata = MNIST('../DATA')
mndata.load_testing()
X = mndata.test_images
X0 = np.asarray(X)[:1000,:]/256.0
X = X0
def updatelable(X, centers):
    return np.argmin(cdist(X, centers), axis = 1)
def updatecenter(X, lable):
    center = np.zeros((K,X.shape[1]))
    for x in range(K):
        center[x,:]= np.mean(X[lable == x,:], axis = 0)
    return center
def Stop(X, Y):
    return (X == Y).all()
def KClustermain(X):
    center = [X[np.random.choice(X.shape[0],K, replace = False)]]
    lable = []
    lable.append(updatelable(X, center[-1]))
    while True:
        center.append(updatecenter(X, lable[-1]))
        if Stop(center[-2], center[-1]):
            break
        lable.append(updatelable(X, center[-1]))
    return lable[-1], center[-1]
K = 10
kmeans = KMeans(n_clusters=K).fit(X)
fig = plt.figure()
pred_label,center = KClustermain(X)

print(type(center.T))
print(center.T.shape)
A = display_network(center.T, K, 1)

f1= fig.add_subplot(2,2,1)
f1.imshow(A, interpolation='nearest', cmap = "jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)

# plt.savefig('a1.png', bbox_inches='tight')


# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4) 
image = cmap(norm(A))

import scipy.misc
scipy.misc.imsave('../aa.png', image)

print(type(pred_label))
print(pred_label.shape)
print(type(X0))
N0 = 20;
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
    Xk = X0[pred_label == k, :]

    center_k = [center[k]]
    neigh = NearestNeighbors(N0).fit(Xk)
    dist, nearest_id  = neigh.kneighbors(center_k, N0)
    
    X1[N0*k: N0*k + N0,:] = Xk[nearest_id, :]
    X2[N0*k: N0*k + N0,:] = Xk[:N0, :]

plt.axis('off')
A = display_network(X1.T, K, N0)
f2 = fig.add_subplot(2,2,2)
f2.imshow(A, interpolation='nearest' )
plt.gray()
plt.show()