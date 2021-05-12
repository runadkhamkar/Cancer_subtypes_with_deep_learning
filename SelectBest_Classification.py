
############################# IMPORT LIBRARY  #################################
seed=75
import numpy as np
import tensorflow 
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import average_precision_score, precision_recall_curve, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, auc, cohen_kappa_score, precision_recall_curve, log_loss, roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
#from sklearn.metrics.classification import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE, RFECV
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE 
from imblearn.combine import SMOTEENN, SMOTETomek
from keras.initializers import RandomNormal
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model, load_model
from FeatureExtractor import *
from fine_tunning import *
matplotlib.use('Agg')
np.random.seed(seed)


##################### ##   LOAD BREAST CANCER DATASET #######################
'''
file_1 = pd.read_csv('./data/subtype_molecular_rna_seq.csv')
data = file_1.iloc[0:20439,2:607].values  
X=data.T
       
file_2 = pd.read_csv('./data/subtype_molecular_rna_seq_label.csv', low_memory=False)
label= file_2.iloc[0,2:607].values   
y=label.T

print('Actual dataset shape {}'.format(Counter(y)))
'''
file_1 = pd.read_csv('./data/ucec_rna_seq_data.csv')
data = file_1.iloc[0:20482,2:232].values 
X=data.T

file_2 = pd.read_csv('./data/ucec_rna_seq_data_label.csv', low_memory=False)
label = file_2.iloc[0,2:232].values   #First row then column from dataset
y=label.T


count=0
aaecount=0

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=seed),
    XGBClassifier(learning_rate=0.001,max_depth=4,n_estimators=100, nthread=1, subsample=0.65),
    GradientBoostingClassifier(random_state=seed),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel='rbf', probability=True, random_state=seed),
    LogisticRegression(C=0.1, multi_class= 'multinomial', solver='sag', random_state=seed),
    MLPClassifier(hidden_layer_sizes=(500), random_state=seed, verbose=True, activation='tanh', solver='adam', alpha=0.0001, batch_size='auto'),
    VotingClassifier(estimators=[('MLP', MLPClassifier(hidden_layer_sizes=(500), random_state=seed, verbose=True, activation='tanh', solver='adam', alpha=0.0001, batch_size='auto')), 
    ('LDA', LinearDiscriminantAnalysis()),
    ('LR', LogisticRegression(C=0.1, multi_class= 'multinomial', solver='sag', random_state=seed))], voting='soft')
    ]
log_cols=["Classifier", "Accuracy", "F1-Score", "Recall", "Precision", "AUC", "MCC", "Kappa", "Log-Loss"]
log = pd.DataFrame(columns=log_cols)



def zero_mix(x, n):
    temp = np.copy(x)
    noise=n
    if 'spilt' in noise:
        frac = float(noise.split('-')[1])
    for i in temp:
        n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
        i[n] = 0
    return (temp)

def gaussian_mix(x):
    n = np.random.normal(0, 0.1, (len(x), len(x[0])))
    return (x + n)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):
       x_train, x_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       
       print('Dataset shape for Train {}'.format(Counter(y_train)))
       print('Dataset shape for Test {}'.format(Counter(y_test)))
       
           
       ################################# OVER SAMPLING ###############################
      
       sm = SMOTE(sampling_strategy='auto', random_state=seed)
       x_train, y_train = sm.fit_resample(x_train, y_train)
       
       print('Resampled dataset shape for Train {}'.format(Counter(y_train)))
       print('Resampled dataset shape for Test {}'.format(Counter(y_test)))
       
       #############################  FEATURE SCALING/NORMALIZATION ##################

       qt = QuantileTransformer(n_quantiles=10, random_state=seed)
       qt.fit(x_train)
       x_train=qt.transform(x_train)
       x_test=qt.transform(x_test)
       
       ################ VARIOUS AUTOENCODERS ###############
       selector = SelectFromModel(estimator=LinearSVC(), max_features=50).fit(x_train, y_train)
       x_train = selector.transform(x_train)
       x_test = selector.transform(x_test)
       print ('After Feature_Selection', x_train.shape)
       '''


       
       ##############  AAE  ##############
       #aae_model('./feature_extraction/AAE/'+aaenum+'/', AdversarialOptimizerSimultaneous(),
       #          xtrain=x_train,ytrain=y_train, xtest=x_test, ytest=y_train, encoded_dim=50,img_dim=x_train.shape[1], nb_epoch=100)
       path='./WithFineTunning/encoder'+aaenum+'.h5'
       model=getFeatures(path=path,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
       model.save(path)
       path1='./WithFineTunning/encoder'+aaenum+'_tunned.h5'
       fine_tune(model_given=path,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,filename=path1)
       model = load_model(path1)
       x_train = model.predict(x_train)
       print('X_Train Shape after fine_tune :', x_train.shape)
       x_test = model.predict(x_test)
       print('X_Test Shape after fine_tune :', x_test.shape)
       
       '''
       ########################    CLASSIFICATION    ##########################
       def accuracy_score(y_test,y_pred):
        a=0
        for i in range(len(y_pred)):
          if(y_pred[i]==y_test[i]):
            a+=1
        return a*100/len(y_test)
       for clf in classifiers:
           clf.fit(x_train, y_train)
           name = clf.__class__.__name__
           print("="*30)
           print(name)
           print('****Results****')
           y_pred = clf.predict(x_test)
           y_pred_proba = clf.predict_proba(x_test)
           y_test_binarize = label_binarize(y_test, classes=[0,1,2,3])
           
           acc = accuracy_score(y_test, y_pred)
           print("Accuracy: {:.4%}".format(acc))
           
           
           f1=f1_score(y_test, y_pred,average='weighted')
           print("F1 Score Weighted: {:.4%}".format(f1))
           
           
           rs=recall_score(y_test, y_pred, average='weighted')
           print("Recall Score Weighted: {:.4%}".format(rs))
           
           
           ps=precision_score(y_test, y_pred, average='weighted')
           print("Precision Weighted: {:.4%}".format(ps))
           
           
           auc=roc_auc_score(y_test_binarize,y_pred_proba, average='macro')
           print("AUC Score: {:.4%}".format(auc))
           
           
           mcc=matthews_corrcoef(y_test, y_pred)
           print("MCC Score: {:.4%}".format(mcc))
           
           
           kappa=cohen_kappa_score(y_test, y_pred, labels=None, weights=None, sample_weight=None)
           print("Kappa: {:.4%}".format(kappa))
           
           
           ll = log_loss(y_test, y_pred_proba)
           print("Log Loss: {:.4%}".format(ll))
           
           log_entry = pd.DataFrame([[name, acc, f1, rs, ps, auc, mcc, kappa, ll]], columns=log_cols)
           log = log.append(log_entry)
       
       
       print("="*30)
       print (log)
       
             
################################################################################

print('###########################################')

result_temp = pd.DataFrame(log)

result_final=result_temp.groupby('Classifier').mean()
result_final.to_csv("./results/12.5.21/result_withSelect_SVC.tsv", sep='\t')
print (result_final)

print('###########################################')
print('Result Saved Successfully')