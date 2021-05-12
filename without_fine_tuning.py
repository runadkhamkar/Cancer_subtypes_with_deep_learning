
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
'''
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from variational_autoencoder import *
from aae_architechture import *
from deep_autoencoder import *
from denoising_autoencoder import *
from shallow_autoencoder import *
'''
matplotlib.use('Agg')
np.random.seed(seed)


#######################   LOAD BREAST CANCER DATASET #######################

file_1 = pd.read_csv('./data/subtype_molecular_rna_seq.csv')
data = file_1.iloc[0:20439,2:607].values  
X=data.T
       
file_2 = pd.read_csv('./data/subtype_molecular_rna_seq_label.csv', low_memory=False)
label= file_2.iloc[0,2:607].values   
y=label.T

print('Actual dataset shape {}'.format(Counter(y)))

############################  LOAD UCEC  DATA   ###########################
'''
file_1 = pd.read_csv('./data/ucec_rna_seq.csv')
data = file_1.iloc[0:20482,2:232].values 
X=data.T

file_2 = pd.read_csv('./data/ucec_rna_seq_label.csv', low_memory=False)
label = file_2.iloc[0,2:232].values   #First row then column from dataset
y=label.T   

print('Actual dataset shape {}'.format(Counter(y)))
'''

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
# The above two functions are used to add noise in the data
# And used to train denoising autoencoder

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):

       #print("TRAIN:", train_index, "TEST:", test_index)
       x_train, x_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       #print("TRAIN:", x_train, "TEST:", x_test)
       print('Dataset shape for Train {}'.format(Counter(y_train)))
       print('Dataset shape for Test {}'.format(Counter(y_test)))
       
           
       ################################# OVER SAMPLING ###############################
      
       sm = SMOTE(sampling_strategy='auto', random_state=seed)
       x_train, y_train = sm.fit_resample(x_train, y_train)
       #oversample only traning data
       
       '''
       sm = SMOTEENN(ratio=1, random_state=seed)
       x_train, y_train = sm.fit_sample(x_train, y_train)
       '''

       '''
       sm = SMOTETomek(ratio=1, random_state=seed)
       x_train, y_train = sm.fit_sample(x_train, y_train)
       '''

       '''
       sm=RandomOverSampler(ratio=1, random_state=seed)
       x_train, y_train = sm.fit_sample(x_train, y_train)
       '''
       
       '''
       sm=ADASYN(ratio=1, random_state=seed, n_neighbors=5, n_jobs=1)
       x_train, y_train = sm.fit_sample(x_train, y_train)
       '''
       print('Resampled dataset shape for Train {}'.format(Counter(y_train)))
       print('Resampled dataset shape for Test {}'.format(Counter(y_test)))
       
       
       #use this when train denoising autoencoder
       #use either gaussian mix or zero mix

       #x_train_noisy=zero_mix(x_train, 'spilt-0.05')
       #x_test_noisy=zero_mix(x_test, 'spilt-0.05')
      
       #x_train_noisy=gaussian_mix(x_train)
       #x_test_noisy=gaussian_mix(x_test)
       #n_samples, n_features = x_train.shape

       #############################  FEATURE SCALING/NORMALIZATION ##################

       qt = QuantileTransformer(n_quantiles=10, random_state=seed)
       qt.fit(x_train)
       x_train=qt.transform(x_train)
       x_test=qt.transform(x_test)

      
       '''
       # Standart Scaling
       sc = StandardScaler()
       sc.fit(x_train)
       x_train=sc.transform(x_train)
       x_test=sc.transform(x_test)
       '''   
          
       ###############################DIMENSION REDUCTION ############################
      
       '''
       pca = PCA(n_components=50, random_state=seed)
       pca.fit(x_train, y_train)
       x_train = pca.transform(x_train)
       x_test = pca.transform(x_test)
       print ('After PCA', x_train.shape)
       '''
       
       
       ################ VARIOUS AUTOENCODERS ###############
       
       aaecount= aaecount+1
       aaenum=str(aaecount)


       ######### Shallow Autoencoder ############
       '''
       shallow_autoencoder_fit(x_train, x_test, encoding_dim=50, optimizer="adadelta",
       						   loss_function="binary_crossentropy", nb_epoch=100, 
       						   batch_size=20, path='./feature_extraction/shallowAE/'+aaenum+'/')

       #do not require fine tuning since this autoencoder does not have any hidden layer
       shallow_autoencoder = load_model('./feature_extraction/shallowAE/'+aaenum+'/shallow_encoder'+'.h5')
       x_train = shallow_autoencoder.predict(x_train)
       print('X_Train Shape after ShallowAE :', x_train.shape)
       x_test = shallow_autoencoder.predict(x_test)
       print('X_Test Shape after ShallowAE :', x_train.shape)
       '''
       

       ######### Denoising Autoencoder ############
       '''
       denoising_autoencoder_fit(x_train, x_test, x_train_noisy, x_test_noisy, encoding_dim=50, optimizer="adadelta",
       							 loss_function="binary_crossentropy", nb_epoch=50, 
       							 batch_size=20, path='./feature_extraction/denoisingAE/'+aaenum+'/')
       #after fiting the model once; it can be directly load from the saved folder hence you can comment out the above lines
       #do not require fine tuning since this autoencoder does not have any hidden layer
       denoising_autoencoder = load_model('./feature_extraction/denoisingAE/'+aaenum+'/denoising_encoder'+'.h5')

       x_train = denoising_autoencoder.predict(x_train)
       print('X_Train Shape after ShallowAE :', x_train.shape)
       x_test = denoising_autoencoder.predict(x_test)
       print('X_Test Shape after ShallowAE :', x_train.shape)
       '''
       

       ######### Deep Autoencoder  ##########
       '''
       deep_autoencoder_fit(x_train, x_test, encoding_dim=50, optimizer="adadelta",
                            loss_function="binary_crossentropy", nb_epoch=100, 
                            batch_size=20, path='./feature_extraction/DeepAE/'+aaenum+'/')
       
       ####### don't need to use the following lines if autoencoder require fine tuning
       deep_encoder = load_model('./feature_extraction/DeepAE/'+aaenum+'/deep_autoencoder'+'.h5')
       
       x_train = deep_encoder.predict(x_train)
       print('X_Train Shape after DeepAE :', x_train.shape)
       
       x_test = deep_encoder.predict(x_test)
       print('X_Test Shape after DeepAE :', x_test.shape)
       '''
       
       ##############  AAE  ##############
       '''
       aae_model('./feature_extraction/AAE/'+aaenum+'/', AdversarialOptimizerSimultaneous(),
                 xtrain=x_train,ytrain=y_train, xtest=x_test, ytest=y_train, encoded_dim=50,img_dim=x_train.shape[1], nb_epoch=100)          
       
       
       ####### don't need to use the following lines if autoencoder require fine tuning
       model = load_model('./feature_extraction/AAE/'+aaenum+'/aae_encoder'+'.h5')

       x_train = model.predict(x_train)
       print('X_Train Shape after AAE :', x_train.shape)
       
       x_test = model.predict(x_test)
       print('X_Test Shape after AAE :', x_test.shape)
       '''
                 
       ################  Variational Autoencoder  ####################
       '''
       vae_model_single('./feature_extraction/VAE/'+aaenum+'/',x_train.shape[1],
       					x_train,x_test,intermediate_dim=1000,batch_size=20,latent_dim=50,epochs=50)
       '''
       ####### don't need to use the following lines if autoencoder require fine tuning
       model = load_model('encoder.h5')
       x_train = model.predict(x_train)
       print('X_Train Shape after VAE :', x_train.shape)
       x_test = model.predict(x_test)
       print('X_Test Shape after VAE :', x_test.shape)
       
       
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
result_final.to_csv("./results/result_withoutFinetunning.tsv", sep='\t')
print (result_final)

print('###########################################')
print('Result Saved Successfully')