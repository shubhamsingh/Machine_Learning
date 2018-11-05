import pandas as pd
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')


#Solution 1
print("\n\n-------------------------Solution 1-----------------------\n\n")

#Load the Data
digits = load_digits()

#Split the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.6, random_state=True)
#K-nearest neighbors
print("\n\n----------K-nearest neighbors----------\n")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred_train=neigh.predict(X_train)
y_pred=neigh.predict(X_test)
#print("Accuracy",neigh.score(X_test, y_test, sample_weight=None))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = neigh.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
folds=5
accs = []
cms = np.zeros(10)
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    neigh.fit(X_train[train_indices], y_train[train_indices])
    y_pred=neigh.predict(X_train[test_indices])
    ac = accuracy_score(y_train[test_indices], y_pred)
    cm = confusion_matrix(y_pred,y_train[test_indices])
    accs.append(ac)
    cms = cms +  cm
print("Accuracy Using Cross Validation",np.mean(accs))
print("Confusion Matrix\n",cms)
total=0
correct=0
top_n =3
for i in X_test:
    y_pred=neigh.predict(i.reshape(1,-1))
    probs = neigh.predict_proba(i.reshape(1,-1))
    best_n = np.argsort(probs, axis=1)[:,-top_n:]
    if(y_pred in best_n):
        correct = correct+1
        total = total + 1
    else:
        total =total + 1
print("Top n_Accuracy",float(correct/total))


#Decision Tree
print("\n\n\n\n-------------Decision Tree-------------\n")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred_train=clf.predict(X_train)
y_pred=clf.predict(X_test)
#print("Accuracy",clf.score(X_test, y_test, sample_weight=None))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
folds=5
accs = []
cms = np.zeros(10)
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    clf.fit(X_train[train_indices], y_train[train_indices])
    y_pred=clf.predict(X_train[test_indices])
    ac = accuracy_score(y_train[test_indices], y_pred)
    cm = confusion_matrix(y_pred,y_train[test_indices])
    accs.append(ac)
    cms = cms +  cm
print("Accuracy Using Cross Validation",np.mean(accs))
print("Confusion Matrix\n",cms)
total=0
correct=0
top_n =3
for i in X_test:
    y_pred=clf.predict(i.reshape(1,-1))
    probs = clf.predict_proba(i.reshape(1,-1))
    best_n = np.argsort(probs, axis=1)[:,-top_n:]
    if(y_pred in best_n):
        correct = correct+1
        total = total + 1
    else:
        total =total + 1
print("Top n_Accuracy",float(correct/total))

#SVM
print("\n\n\n\n------------------SVM------------------\n")
sup = svm.SVC(probability=True)
sup.fit(X_train, y_train)
y_pred_train=sup.predict(X_train)
y_pred=sup.predict(X_test)
#print("Accuracy",sup.score(X_test, y_test, sample_weight=None))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
#y_score = sup.fit(X_train, y_train).decision_function(X_test)
#print("ROC Score",roc_auc_score(y_test, y_score))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = sup.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
folds=5
accs = []
cms = np.zeros(10)
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    sup.fit(X_train[train_indices], y_train[train_indices])
    y_pred=sup.predict(X_train[test_indices])
    ac = accuracy_score(y_train[test_indices], y_pred)
    cm = confusion_matrix(y_pred,y_train[test_indices])
    accs.append(ac)
    cms = cms +  cm
print("Accuracy Using Cross Validation",np.mean(accs))
print("Confusion Matrix\n",cms)
total=0
correct=0
top_n =3
for i in X_test:
    y_pred=sup.predict(i.reshape(1,-1))
    probs = sup.predict_proba(i.reshape(1,-1))
    best_n = np.argsort(probs, axis=1)[:,-top_n:]
    if(y_pred in best_n):
        correct = correct+1
        total = total + 1
    else:
        total =total + 1
print("Top n_Accuracy",float(correct/total))

#Logistic Regression
print("\n\n\n\n----------Logistic Regression----------\n")
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred_train=logisticRegr.predict(X_train)
y_pred=logisticRegr.predict(X_test)
#print("Accuracy",logisticRegr.score(X_test, y_test))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
#y_score = logisticRegr.fit(X_train, y_train).decision_function(X_test)
#print("ROC Score",roc_auc_score(y_test, y_score))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = logisticRegr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
folds=5
accs = []
cms = np.zeros(10)
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    logisticRegr.fit(X_train[train_indices], y_train[train_indices])
    y_pred=logisticRegr.predict(X_train[test_indices])
    ac = accuracy_score(y_train[test_indices], y_pred)
    cm = confusion_matrix(y_pred,y_train[test_indices])
    accs.append(ac)
    cms = cms +  cm
print("Accuracy Using Cross Validation",np.mean(accs))
print("Confusion Matrix\n",cms)
total=0
correct=0
top_n =3
for i in X_test:
    y_pred=logisticRegr.predict(i.reshape(1,-1))
    probs = logisticRegr.predict_proba(i.reshape(1,-1))
    best_n = np.argsort(probs, axis=1)[:,-top_n:]
    if(y_pred in best_n):
        correct = correct+1
        total = total + 1
    else:
        total =total + 1
print("Top n_Accuracy",float(correct/total))



#Solution 2
print("\n\n-------------------------Solution 2-----------------------\n\n")


#Load the Data
data1 = pd.read_csv('/Users/shubhamsingh/Downloads/cc.csv')
#print(data1.isnull().sum())
y_true=data1['default.payment.next.month']
data=data1.drop(data1.columns[len(data1.columns)-1], axis=1, inplace=False)
del data['ID']

#Split the data
X_train, X_test, y_train, y_test = train_test_split(data,y_true,test_size=0.6,random_state=True)

X = data
y = y_true

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#K-nearest neighbors
print("\n\n----------K-nearest neighbors----------\n")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred_train=neigh.predict(X_train)
y_pred=neigh.predict(X_test)
#print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("Accuracy",neigh.score(X_test, y_test, sample_weight=None))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = neigh.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=None)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
A = neigh.kneighbors_graph(X_test,n_neighbors=600,mode='connectivity')
A.toarray()

folds=3
accs = []
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    neigh.fit(X_train.values[train_indices], y_train.values[train_indices])
    y_pred=neigh.predict(X_train.values[test_indices])
    ac = accuracy_score(y_train.values[test_indices], y_pred)
    accs.append(ac)
print("Accuracy Using Cross Validation",np.mean(accs))
d ={'n_neighbors' : [5,50]}
ne = GridSearchCV(estimator=neigh, cv=3, param_grid=d,n_jobs=-1)
ne.fit(X_train,y_train)
print("Accuracy",ne.score(X_test,y_test)) 
print("Accuracy Using Cross Validation",cross_val_score(ne,X_test,y_test))
print("Score of best_estimator",ne.best_score_) 
print("Best Parameter setting",ne.best_params_)
print("Estimator which gave highest score",ne.best_estimator_)


#Decision Tree
print("\n\n\n\n-------------Decision Tree-------------\n")
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred_train=clf.predict(X_train)
y_pred=clf.predict(X_test)
#print("Accuracy",clf.score(X_test, y_test, sample_weight=None))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=None)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

folds=3
accs = []
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    clf.fit(X_train.values[train_indices], y_train.values[train_indices])
    y_pred=clf.predict(X_train.values[test_indices])
    ac = accuracy_score(y_train.values[test_indices], y_pred)
    accs.append(ac)
print("Accuracy Using Cross Validation",np.mean(accs))

dot_data = tree.export_graphviz(clf, feature_names=np.array(list(data)),class_names=['0','1'],out_file=None,filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data) 
graph.render("Credit") 
print(graph)
d ={'max_depth' : [2,3]}
ne = GridSearchCV(estimator=clf, cv=3, param_grid=d,n_jobs=-1)

ne.fit(X_train,y_train)
print("Accuracy",ne.score(X_test,y_test)) 
print("Accuracy Using Cross Validation",cross_val_score(ne,X_test,y_test))
print("Score of best_estimator",ne.best_score_) 
print("Best Parameter setting",ne.best_params_)
print("Estimator which gave highest score",ne.best_estimator_)

#SVM
print("\n\n\n\n------------------SVM------------------\n")
sup = svm.SVC(probability=True)
sup.fit(X_train, y_train)
y_pred_train=sup.predict(X_train)
y_pred=sup.predict(X_test)
#print("Accuracy",sup.score(X_test, y_test, sample_weight=None))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
#y_score = sup.fit(X_train, y_train).decision_function(X_test)
#print("ROC Score",roc_auc_score(y_test, y_score))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = sup.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=None)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
d ={'degree' : [2,3,4]}
ne = GridSearchCV(estimator=sup, cv=3, param_grid=d,n_jobs=-1)

ne.fit(X_train,y_train)
print("Accuracy",ne.score(X_test,y_test)) 
print("Accuracy Using Cross Validation",cross_val_score(ne,X_test,y_test))
print("Score of best_estimator",ne.best_score_) 
print("Best Parameter setting",ne.best_params_)
print("Estimator which gave highest score",ne.best_estimator_)

#Logistic Regression
print("\n\n\n\n----------Logistic Regression----------\n")
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred_train=logisticRegr.predict(X_train)
y_pred=logisticRegr.predict(X_test)
#print("Accuracy",logisticRegr.score(X_test, y_test))
print("Accuracy",accuracy_score(y_test, y_pred, normalize=True))
print("F1-score",f1_score(y_test, y_pred, average=None))
print("Precision and Recall Score",precision_recall_fscore_support(y_test, y_pred, average='macro'))
#y_score = logisticRegr.fit(X_train, y_train).decision_function(X_test)
#print("ROC Score",roc_auc_score(y_test, y_score))
print("Confusion Matrix for Train\n", metrics.confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Test\n", metrics.confusion_matrix(y_test, y_pred))
probs = logisticRegr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=None)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

folds=3
accs = []
kf = KFold(len(X_train), n_folds=folds, shuffle=True)
for train_indices, test_indices in kf:
    logisticRegr.fit(X_train.values[train_indices], y_train.values[train_indices])
    y_pred=logisticRegr.predict(X_train.values[test_indices])
    ac = accuracy_score(y_train.values[test_indices], y_pred)
    accs.append(ac)
print("Accuracy Using Cross Validation",np.mean(accs))
d ={'penalty' : ['l1','l2']}
ne = GridSearchCV(estimator=logisticRegr, cv=3, param_grid=d,n_jobs=-1)

ne.fit(X_train,y_train)
print("Accuracy",ne.score(X_test,y_test)) 
print("Accuracy Using Cross Validation",cross_val_score(ne,X_test,y_test))
print("Score of best_estimator",ne.best_score_) 
print("Best Parameter setting",ne.best_params_)
print("Estimator which gave highest score",ne.best_estimator_)
