# ElasticNetReg
Elastic Net Regularization python algorithm on qm9 dataset

# Files accordingly:

featureCM.py - parses the feature data
prop.py
folding.py - 'n-folds' the data set for elastic net

fitElasticNet.py - fits the elastic net (using sklearn GridSearchCV model) according to l1_ratio and alpha given by Faber et al. article (same one as my Bayessian regression algorithm
ElasticNetDFT.py - imports all above files to perform Elastic Net regularization on the all 13 features on the dataset (qm9)

# Results:
written to the CM-EN.txt file

Thank you and enjoy!
