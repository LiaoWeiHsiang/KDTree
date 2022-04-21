import json
import os
import math
import numpy as np
from kd_tree import *
from time import time

# maps_root = '/data3/map_by_scenes_v7_Downsampling0.1_no_ground_ieflat'
# map_name = 'singapore-hollandvillage'
# with open(os.path.join(maps_root,map_name,"coordinate_and_scenes_number.json"), 'r') as f:
#     coordinate_and_scenes_number = json.load(f)

# center_coordinates = np.array(coordinate_and_scenes_number["center_coordinate"])
# scenes_numbers = np.array(coordinate_and_scenes_number["scenes_number"])
# print(center_coordinates)

print("Testing KD Tree...")
test_times = 5
run_time_1 = run_time_2 = 0


def get_euclidean_distance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
def exhausted_search(X, Xi):
    dist_best = float('inf')
    row_best = None
    for row in X:
        dist = get_euclidean_distance(Xi, row)
        if dist < dist_best:
            dist_best = dist
            row_best = row
    return row_best
for _ in range(test_times):
    low = 0
    high = 100
    n_rows = 1000
    n_cols = 2
    X = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    y = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    # X = [[1,3],[5,8],[5,9]]
    # y = [[1,3],[5,8],[5,9]]
    Xi = [2.1,3.1]

    tree = KDTree()
    tree.build_tree(X, y)

    start = time()
    nd = tree.nearest_neighbour_search(Xi)
    run_time_1 += time() - start

    print(nd.split[0])
    ret1 = get_euclidean_distance(Xi, nd.split[0])

    start = time()
    row = exhausted_search(X, Xi)
    run_time_2 += time() - start
    ret2 = get_euclidean_distance(Xi, row)

    assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" \
        % (str(Xi), str(nd), str(row), str(tree))
print("%d tests passed!" % test_times)
print("KD Tree Search %.2f s" % run_time_1)
print("Exhausted search %.2f s" % run_time_2)
