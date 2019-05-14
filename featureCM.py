#import numpy
import numpy as np

def parse_feature_CM(path):
    #establish np array of zeros the size of the data set
    CM_Vec = np.zeros([133885, 900])
    #read in file
    CM_file = open(path).read().split('\n')

    #split data to return vector
    for i in CM_file[:-1]:
        qm9_entry = i.split()
        qm9_index = int(qm9_entry[0].replace('qm9:', ''))
        for indexj, j in enumerate(qm9_entry[1:]):
            CM_Vec[qm9_index-1][indexj] = float(j)
    return CM_Vec
