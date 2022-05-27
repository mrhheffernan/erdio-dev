import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: librosa dependencies apparently require specific versions of numpy, try numpy==1.21.4
import librosa
import librosa.display
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier
from joblib import dump
# from joblib import load




eq_df = pd.read_csv('./large_data/eq_harmony_combined.csv')
# display(eq_df)

# nansvec = np.isnan(eq_df['crestfactor'])
# nansvec[nansvec==True]

eq_df['power_ratio'] = np.log10(eq_df['percussive_power'].values / (eq_df['harmonic_power'].values))
eq_df['hits_ratio'] = np.log10(eq_df['percussive_hits'].values / (eq_df['harmonic_hits'].values + 1e-1) + 5e-4)

# try classifying with the log instead?
for i in range(0,len(eq_df)):
    eq_df.iloc[i,1:-9] = np.log10(eq_df.iloc[i,1:-9].values.astype(float))

def TPR(prediction, data, numclasses):
    """
    Returns True Positive Ratio given a prediction and data
    """
    confmat = confusion_matrix(prediction, data)

#     TN = confmat[0,0]
#     FP = confmat[0,1]
#     FN = confmat[1,0]
#     TP = confmat[1,1]
    TP = np.zeros(numclasses)
    FN = np.zeros(numclasses)
    R = np.zeros(numclasses)
    for i in range(numclasses):
        TP[i] = confmat[i,i]
        FN[i] = confmat[i,:].sum() - confmat[i,i]

    R = TP/(TP + FN)


    return R


def recall(prediction, data, numclasses):
    """
    Calculates recall of a prediction
    """

    confmat = confusion_matrix(prediction, data)

#     TN = confmat[0,0]
#     FP = confmat[0,1]
#     FN = confmat[1,0]
#     TP = confmat[1,1]
#     TP = confmat[6,6]
#     FN = confmat[6,:] - confmat[6,6]
    TP = np.zeros(numclasses)
    R = np.zeros(numclasses)
    FN = np.zeros(numclasses)
    for i in range(numclasses):
        TP[i] = confmat[i,i]
        FN[i] = confmat[i,:].sum() - confmat[i,i]

    R = TP/(TP + FN)

    return R

def precision(prediction, data,numclasses):
    """
    Calculates precision of a prediction
    """

    confmat = confusion_matrix(prediction, data)

#     TN = confmat[0,0]
#     FP = confmat[0,1]
#     FN = confmat[1,0]
#     TP = confmat[1,1]
#     TP = confmat[6,6]
#     FP = confmat[:,6].sum() - confmat[6,6]
    TP = np.zeros(numclasses)
    FP = np.zeros(numclasses)
    P = np.zeros(numclasses)
    for i in range(numclasses):
        TP[i] = confmat[i,i]
        FP[i] = confmat[:,i].sum() - confmat[i,i]

    P = TP/(TP + FP)

    return P

def Fmeasure(prediction, data, numclasses):
    """
    Returns Fmeasure.

    This is considered a balance of the precision and the recall.

    F = (2*P*R)/(P+R)

    where

    P = TP/(TP + FP) is the precision and
    R = TP/(TP + FN) is the recall.

    Reference:
    Müller, Meinard. Fundamentals of music processing: Audio, analysis, algorithms, applications.
    Vol. 5. Cham: Springer, 2015.
    Sec. 4.5 pp. 217
    """
    confmat = confusion_matrix(prediction, data)

#     TN = confmat[0,0]
#     FP = confmat[0,1]
#     FN = confmat[1,0]
#     TP = confmat[1,1]
#     TP = confmat[6,6]
#     FN = confmat[6,:].sum() - confmat[6,6]
#     FP = confmat[:,6].sum() - confmat[6,6]
    TP = np.zeros(numclasses)
    FN = np.zeros(numclasses)
    FP = np.zeros(numclasses)
    P = np.zeros(numclasses)
    R = np.zeros(numclasses)
    for i in range(numclasses):
        TP[i] = confmat[i,i]
        FN[i] = confmat[i,:].sum() - confmat[i,i]
        FP[i] = confmat[:,i].sum() - confmat[i,i]

    P = TP/(TP + FP)
    R = TP/(TP + FN)

    F = (2*P*R)/(P + R)

    return F


def randlayer():
    return int(round(np.random.rand()*200,0)+50)


def voter_tuple(i):
    teststr = 'mlp'+str(i)
    return (teststr, MLPClassifier(hidden_layer_sizes=(randlayer(),randlayer(),randlayer(),randlayer(),randlayer(),), max_iter=100000, early_stopping=True))


num_voters=100
accuracy_vec = np.zeros(10)
for dropfold in range(1,11):
    eq_df2 = eq_df.copy()
    eq_df2.replace({'air_conditioner':0, 'car_horn':1, 'children_playing':2, 'dog_bark':3, 'drilling':4,
                'engine_idling':5, 'gun_shot':6, 'jackhammer':7, 'siren':8, 'street_music':9},inplace=True)
#     eq_df2.replace({'air_conditioner':0, 'car_horn':1, 'children_playing':2, 'dog_bark':3, 'drilling':4,
#                 'engine_idling':0, 'gun_shot':6, 'jackhammer':4, 'siren':8, 'street_music':2},inplace=True)
#     eq_df2.replace({'air_conditioner':0, 'car_horn':0, 'children_playing':0, 'dog_bark':0, 'drilling':0,
#                     'engine_idling':0, 'gun_shot':1, 'jackhammer':0, 'siren':0, 'street_music':0},inplace=True)

    eq_df3 = eq_df2.drop(eq_df2[eq_df2['fold']==dropfold].index)
    eq_df3.drop(columns='fold',inplace=True)
    eq_df3.drop(columns='salience',inplace=True)
    X_train = eq_df3.iloc[:,1:].values
    y_train = eq_df3.iloc[:,0].values

#     X_val = eq_df2[eq_df2['fold'] == dropfold]
#     X_val = X_val.iloc[:,1:].values
#     y_val = eq_df2[eq_df2['fold'] == dropfold]
#     y_val = y_val.iloc[:,0].values

    X_val = eq_df2[eq_df2['fold'] == dropfold].copy()
    X_val.drop(columns='fold',inplace=True)
    X_val.drop(columns='salience',inplace=True)
    X_val = X_val.iloc[:,1:].values
    y_val = eq_df2[eq_df2['fold'] == dropfold]
    y_val = y_val.iloc[:,0].values

    voter_list = []

    for i in range(num_voters):
        voter_list.append(voter_tuple(i))

#     mlp.fit(X_train, y_train)
    vote_class = VotingClassifier(estimators=voter_list,
                 voting='soft', n_jobs=4)
    vote_class = vote_class.fit(X_train, y_train)
#     acc = 100*Fmeasure(y_val, mlp.predict(X_val),7)
    recall = np.round(100*TPR(y_val, vote_class.predict(X_val),10)[6],2)
    prec = np.round(100*precision(y_val, vote_class.predict(X_val),10)[6],2)
    Fmeas = np.round(100*Fmeasure(y_val, vote_class.predict(X_val),10)[6],2)
#     acc = 100*TPR(y_val, mlp.predict(X_val),7)[5]
    print("Validation TPR of", recall, ",\n \tprecision of ", prec, ",\n \tand Fmeasure of", Fmeas, "on fold", str(dropfold))
    accuracy_vec[dropfold-1] = recall
    dump(vote_class, 'hive_mind_democracy_fold'+str(dropfold)+'.joblib')

print(accuracy_vec)
print(accuracy_vec.mean())
