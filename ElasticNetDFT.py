import numpy as np
import time

#Elastic Net regularization algorithm

# Function import
import featureCM
import prop
import folding
import fitElasticNet

fOut = open("CM-EN.txt", "w")

#  1. CM vector parsing
CM_Path = "SupplementaryMaterials/CM"
time0 = time.time()
CM_vec = featureCM.parse_feature_CM(CM_Path)
# print(CM_vec[1])
time1 = time.time()
fOut.write("Execuation time 01 - CM vector parsing: %s sec.\n" % (time1 - time0))

#  2. Property parsing
QM9_Prop, QM9_Index = prop.parse_prop("SupplementaryMaterials/qm9-mol-info-standardized-v1")
time2 = time.time()
fOut.write("Execuation time 02 - Property parsing: %s sec.\n" % (time2 - time1))

#  3. Fold parsing
Predefined_Split, Train_Index, Test_Index = folding.predef_Fold()
time3 = time.time()
fOut.write("Execuation time 03 - Fold parsing: %s sec. \n" % (time3 - time2))

#  4. Reindex to training and testing set
Train_X = np.zeros([len(Train_Index[0]), CM_vec.shape[1]])
Train_Y = np.zeros([13, len(Train_Index[0])])
Test_X = np.zeros([len(Test_Index[0]), CM_vec.shape[1]])
Test_Y = np.zeros([13, len(Test_Index[0])])
Predict_Y = np.zeros([13, len(Test_Index[0])])
for indexi, i in enumerate(Train_Index[0]):
    Train_X[indexi] = CM_vec[i-1]
    Train_Y[:,indexi] = QM9_Prop[i-1,:]
for indexi, i in enumerate(Test_Index[0]):
    Test_X[indexi] = CM_vec[i-1]
    Test_Y[:,indexi] = QM9_Prop[i-1,:]
time4 = time.time()
fOut.write("Execuation time 04 - Reindex: %s sec. \n" % (time4 - time3))

#  5. Training and Testing - Elastic Net (EN)
test_PATH = "SupplementaryMaterials/qm9-prop-stats-v1"
test_file = open(test_PATH).read().split('\n')
stdevp_vec = np.zeros(13)
MAE = np.zeros(13)
for i in test_file[2:-1]:
    tempStdevp = i.split()
    for j in range (1, np.size(tempStdevp)):
        stdevp_vec[j-1] = float(tempStdevp[j])


fOut.write("\n")
fOut.write("--- Training and Testing EN ---\n")
fOut.write("\n")
fOut.write(" Prop_Id" + "      Err_MAE" + "      Time_Train" + "  Time_Test" + "\n" )
#Only perform EN regularization on 1 value in table because of my hardware speed
for indexi in range(13):
    (Err_MAD, Err_RMSD, Time_Train, Time_Test) = \
        fitElasticNet.fit_EN(Train_X, Train_Y[indexi], Test_X, Test_Y[indexi], Predefined_Split[0])
    MAE[indexi] = Err_MAD*stdevp_vec[indexi]
    fOut.write("{:8}{:14.8f}{:11.2f}{:11.2f}\n".format(indexi, MAE[indexi], Time_Train, Time_Test))
fOut.close()
