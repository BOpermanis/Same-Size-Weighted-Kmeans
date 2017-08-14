import numpy as np
import pickle
import matplotlib.pyplot as plt
import collections


# Euclidian norm squared
def dist(x,y):
    return np.sum((x-y)**2)

# Centroids initialized with uniform distribution
def ChooseInitialMeans(data,k):
    means = []
    for _ in range(k):
        random_centroid = []
        for i in range(data.shape[1]):
            a = min(data[:, i])
            b = max(data[:, i])
            random_centroid.append(np.random.uniform(a, b))
        means.append(random_centroid)
    return means


# Running algo for one initialization of centroids
def kmeansOnceWeights(data,weights,k,n,n_per_cluster):
    means = ChooseInitialMeans(data, k)

    # runs k-means iterations only 50 times
    for iter in range(50):

        if iter>0:
            means = []
            for k0 in ids:
                indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
                if len(indices) > 0:
                    cut = np.take(data, indices, axis=0)
                    means.append(np.apply_along_axis(np.mean, axis=0, arr=cut))

        # caluclates means
        clusters = dict(enumerate(means))
        ids = list(clusters.keys())

        # calculates differences
        diffs = []
        for id in ids:
            diffs.append(np.apply_along_axis(lambda x: dist(x, clusters[id]), axis=1, arr=data))
        diffs = np.asarray(diffs)

        # assigns cluster for each sample point
        clust_sizes = dict(zip(ids, np.zeros(len(ids))))
        closest_cluster = []
        for i in range(n):
            row = diffs[:, i]
            w0 = weights[i]
            inds_sorted = np.argsort(row)
            for id_opt in inds_sorted:
                if clust_sizes[id_opt] < n_per_cluster:
                    closest_cluster.append(id_opt)
                    clust_sizes[id_opt] += w0
                    break

    # calculates cluster within variances
    inner_diffs = []
    for k0 in ids:
        indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
        if len(indices) > 0:
            cut = np.take(diffs, indices, axis=1)
            inner_diffs.append(np.apply_along_axis(np.mean, axis=1, arr=cut)[k0])

    return ids, closest_cluster, sum(inner_diffs)



# runs k-mans B times with different initializations
def kmeans(data,weights,k,n=None,n_per_cluster=None,B=10):

    if n is None:
        n = data.shape[0]

    if n_per_cluster is None:
        n_per_cluster = int(np.ceil(sum(weights) / k))

    results = []
    for b in range(B):
        print(b)
        results.append(kmeansOnceWeights(data,weights, k, n, n_per_cluster))

    inner_diffs = [r[2] for r in results]
    opt = np.argmin(inner_diffs)

    counter = collections.Counter(results[opt][1])
    print(counter)

    return results[opt][0], results[opt][1]



if __name__=="__main__":

    data = pickle.load(open("data.pickle", "rb"))
    weights = pickle.load(open("weights.pickle", "rb"))

    # number of clusters
    k = 2

    # targeted sum of weights for each cluster
    n_per_cluster = int(np.ceil(sum(weights) / k))

    # times running the algo
    B = 10

    n = data.shape[0]

    #results
    ids, closest_cluster = kmeans(data, weights, k)

    # visualization
    for k0 in ids:
        indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
        cut = np.take(data, indices, axis=0)
        x, y = cut[:, 0], cut[:, 1]
        plt.scatter(x, y, c=tuple(np.random.rand(3, 1)[:, 0]))
    plt.show()