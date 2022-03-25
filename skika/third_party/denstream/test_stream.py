#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from DenStream import DenStream

data = pd.read_csv("seqs.csv", header=None)

intervals = set()
for (x, y), val in np.ndenumerate(data):
    intervals.add(val)
print(f"intervals: {intervals} interval len: {len(intervals)}")

outliers = set()
centers = set()

clusterer = DenStream(lambd=0.1, eps=10, beta=0.5, mu=10)
for (x, y), val in np.ndenumerate(data):
    label = clusterer.fit_predict([np.array([val])])[0]

    if label == -1:
        outliers.add(val)
        # print(f"prediction: {val}")
    else:
        centers.add(int(round(clusterer.p_micro_clusters[label].center()[0])))
        # print(f"prediction: {clusterer.p_micro_clusters[label].center()}")

p_micro_cluster_centers = np.array([p_micro_cluster.center()
                                for p_micro_cluster in clusterer.p_micro_clusters])

print(f"p_micro_centers: {p_micro_cluster_centers}")
print(f"outliers: {outliers} len: {len(outliers)}")
print(f"centers: {centers} len: {len(centers)}")
