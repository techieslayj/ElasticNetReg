import numpy as np

def parse_prop(pathProp):
    qm9_prop_File = open(pathProp).read().split('\n')[1:-1]
    qm9_prop = np.zeros([133885, 13])
    qm9_index = np.zeros(133885)
    for indexi, i in enumerate(qm9_prop_File):
        temp = i.split()
        qm9_index[indexi] = int(temp[0].replace('qm9:', ''))
        for indexj in range(13):
            qm9_prop[int(temp[0][4:])-1][indexj] = float(temp[indexj+2])
    return qm9_prop, qm9_index
