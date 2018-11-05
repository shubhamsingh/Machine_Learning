#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:22:18 2018

@author: shubhamsingh
"""

# Note: X Axis is fraction of traning data set in range [0,1] denoted by l in All plots


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
import warnings
warnings.filterwarnings('ignore')

# Solution 1a

# Phase 1: Reading data File
boston = datasets.load_boston()
boston_X = boston.data
boston_Y = boston.target

# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
train_error=[]
test_error=[]
rsquare_score=[]

for i in l:
    boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(boston_X, boston_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)
# Phase 3: Model Fitting and Prediction
    boston_reg = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
    boston_reg.fit(boston_X_train, boston_Y_train)

# Phase 4: Reporting
    boston_model_intercept = boston_reg.intercept_
    boston_model_coeff = boston_reg.coef_
    boston_Y_pred = boston_reg.predict(boston_X_test)
    boston_Y_train_pred = boston_reg.predict(boston_X_train)

    boston_total_variance = np.var(boston_Y_train)
    boston_explained_variance = np.var(boston_Y_train_pred)
    boston_train_error = mean_squared_error(boston_Y_train, boston_Y_train_pred)
    boston_test_error = mean_squared_error(boston_Y_test, boston_Y_pred)
    boston_R2_score = r2_score(boston_Y_train, boston_Y_train_pred)


    train_error.append(boston_train_error)
    test_error.append(boston_test_error)
    rsquare_score.append(boston_R2_score)

# Phase 5: Plots
print("--------------------------Solution 1a------------------------------------")
fig = plt.figure(figsize=(10,8))
plt.scatter(l, train_error, marker='o', s=30, color='blue', label ='Train Error')
plt.scatter(l, test_error, marker='o', s=30, color='red', label ='Test Error')
plt.scatter(l, rsquare_score, marker='o', s=30, color='black',label='r square value')
plt.legend(loc="upper right")
plt.xlabel('Number of training examples in fraction')
plt.show()


#Solution 1b

# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
lambda_diff=[0,0.01,0.1,1]
train_error=[]
test_error=[]
rsquare_score=[]
for i in lambda_diff:
    for j in l:    
        boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(boston_X, boston_Y, test_size=1-j,train_size=j, random_state=None,shuffle=True)

# Phase 3: Model Fitting and Prediction
        boston_reg = linear_model.Ridge (alpha = i,copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
        boston_reg.fit(boston_X_train, boston_Y_train)

# Phase 4: Reporting
        boston_model_intercept = boston_reg.intercept_
        boston_model_coeff = boston_reg.coef_
        boston_Y_pred = boston_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_reg.predict(boston_X_train)

        boston_total_variance = np.var(boston_Y_train)
        boston_explained_variance = np.var(boston_Y_train_pred)
        boston_train_error = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        boston_test_error = mean_squared_error(boston_Y_test, boston_Y_pred)
        boston_R2_score = r2_score(boston_Y_train, boston_Y_train_pred)


        train_error.append(boston_train_error)
        test_error.append(boston_test_error)
        rsquare_score.append(boston_R2_score)


# Phase 5: Plots  
        
print("--------------------------Solution 1b------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(l, train_error[0:7], marker='o', s=10, color='blue', label ='Train Error')
ax1.scatter(l, test_error[0:7], marker='o', s=10, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='Number of training examples in fraction')
ax1.set_ylim(0, 35)


ax2 = fig.add_subplot(242)
ax2.scatter(l,train_error[7:14], marker='o', s=10, color='blue', label ='Train Error')
ax2.scatter(l,test_error[7:14], marker='o', s=10, color='red', label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='Number of training examples in fraction')
ax2.set_ylim(0, 35)


ax3 = fig.add_subplot(243)
ax3.scatter(l,train_error[14:21], marker='o', s=10, color='blue',label ='Train Error')
ax3.scatter(l,test_error[14:21], marker='o', s=10, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='Number of training examples in fraction')
ax3.set_ylim(0, 35)

ax4 = fig.add_subplot(244)
ax4.scatter(l,train_error[21:28], marker='o', s=10, color='blue',label ='Train Error')
ax4.scatter(l,test_error[21:28], marker='o', s=10, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='Number of training examples in fraction')
ax4.set_ylim(0, 35)

ax5 = fig.add_subplot(245)
ax5.scatter(l,rsquare_score[0:7], marker='o', s=10, color='black',label='r square value')
ax5.set_ylim(0, 1)
ax5.set(xlabel='Number of training examples in fraction')
ax5.legend(loc="upper right")

ax6 = fig.add_subplot(246)
ax6.scatter(l,rsquare_score[7:14], marker='o', s=10, color='black',label='r square value')
ax6.set_ylim(0, 1)
ax6.set(xlabel='Number of training examples in fraction')
ax6.legend(loc="upper right")

ax7 = fig.add_subplot(247)
ax7.scatter(l,rsquare_score[14:21], marker='o', s=10, color='black',label='r square value')
ax7.set_ylim(0, 1)
ax7.set(xlabel='Number of training examples in fraction')
ax7.legend(loc="upper right")

ax8 = fig.add_subplot(248)
ax8.scatter(l,rsquare_score[21:28], marker='o', s=10, color='black',label='r square value')
ax8.set_ylim(0, 1)
ax8.set(xlabel='Number of training examples in fraction')
ax8.legend(loc="upper right")

plt.show()




# Solution 2

# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]

train_error=[]
test_error=[]
rsquare_score=[]
mse_score_ridge=[]

for i in l: 
    for j in lambda_diff:
        boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(boston_X, boston_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)

# Phase 3: Model Fitting and Prediction
        boston_reg = linear_model.Ridge (alpha = j,copy_X=True, fit_intercept=True, max_iter=None,
                normalize=True, random_state=None, solver='auto', tol=0.001)
        boston_reg.fit(boston_X_train, boston_Y_train) 
        k_fold=KFold(n=len(boston_X_train),n_folds=5,shuffle=True)
        mse=0.0
        for train_indices, test_indices in k_fold:
            boston_reg.fit(boston_X_train[train_indices],boston_Y_train[train_indices])
            mse=mse+mean_squared_error(boston_Y_train[test_indices],boston_reg.predict(boston_X_train[test_indices]))
        mse_score_ridge.append((mse/5))
        
# Phase 4: Reporting
        boston_model_intercept = boston_reg.intercept_
        boston_model_coeff = boston_reg.coef_
        boston_Y_pred = boston_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_reg.predict(boston_X_train)

        boston_total_variance = np.var(boston_Y_train)
        boston_explained_variance = np.var(boston_Y_train_pred)
        boston_train_error = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        boston_test_error = mean_squared_error(boston_Y_test, boston_Y_pred)
        boston_R2_score = r2_score(boston_Y_train, boston_Y_train_pred)

        train_error.append(boston_train_error)
        test_error.append(boston_test_error)
        rsquare_score.append(boston_R2_score)

print("--------------------------Solution 2------------------------------------")
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(0, 125)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.plot(lambda_diff,mse_score_ridge[11:22],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(0, 125)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.plot(lambda_diff,mse_score_ridge[22:33],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(0, 125)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.plot(lambda_diff,mse_score_ridge[33:44],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(0, 125)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)


ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()




#Solution 3


# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]

train_error=[]
test_error=[]
rsquare_score=[]


mse_score_ridge=[]

for i in l: 
    for j in lambda_diff:
        boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(boston_X, boston_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)

# Phase 3: Model Fitting and Prediction
        boston_reg = linear_model.Lasso(alpha=j, fit_intercept=True, normalize=True, precompute=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        boston_reg.fit(boston_X_train, boston_Y_train)
        k_fold=KFold(n=len(boston_X_train),n_folds=5,shuffle=True)
        boston_reg.set_params(alpha=j)
        mse=0.0
        for train_indices, test_indices in k_fold:
            boston_reg.fit(boston_X_train[train_indices],boston_Y_train[train_indices])
            mse=mse+mean_squared_error(boston_Y_train[test_indices],boston_reg.predict(boston_X_train[test_indices]))
        mse_score_ridge.append((mse/5))

# Phase 4: Reporting
        boston_model_intercept = boston_reg.intercept_
        boston_model_coeff = boston_reg.coef_
        boston_Y_pred = boston_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_reg.predict(boston_X_train)

        boston_total_variance = np.var(boston_Y_train)
        boston_explained_variance = np.var(boston_Y_train_pred)
        boston_train_error = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        boston_test_error = mean_squared_error(boston_Y_test, boston_Y_pred)
        boston_R2_score = r2_score(boston_Y_train, boston_Y_train_pred)
        
        train_error.append(boston_train_error)
        test_error.append(boston_test_error)
        rsquare_score.append(boston_R2_score)

# Phase 5: Plots
print("--------------------------Solution 3------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(0, 150)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(0, 150)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(0, 150)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(0, 150)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)

ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()




#Solution 4a

# Phase 1: Reading data File
boston_data = datasets.load_boston()
boston_X = pd.DataFrame(boston_data.data)
boston_Y = pd.Series(boston_data.target)
boston_X=np.nan_to_num(boston_X)
boston_Y=np.nan_to_num(boston_Y)

# Normalization
boston_X=np.c_[ np.ones(506),boston_X ]
X_cap = boston_X-boston_X.min(0)/boston_X.max(0)-boston_X.min(0)
Y_cap = boston_Y;
#X_cap=(boston_X-np.mean(boston_X))/np.var(boston_X)
#Y_cap=(boston_Y-np.mean(boston_Y))/np.var(boston_Y)

# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
lambda_diff=[0,0.01,0.1,1]
train_error=[]
test_error=[]
rsquare_score=[]
for i in l:
    boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(X_cap, Y_cap, train_size=i,test_size=1-i, random_state=None,shuffle=True)
    for j in lambda_diff:
 # Phase 3: Model Fitting and Prediction
        w_cap = np.dot(np.dot(np.linalg.inv(np.dot(boston_X_train.transpose(),boston_X_train)+ np.dot(j,np.eye(14,14))),boston_X_train.transpose()),boston_Y_train)
 # Phase 4: Reporting  
        train_e= np.sum(np.square(boston_Y_train - np.dot(boston_X_train,w_cap)))
        train_error.append(train_e/int(506*i))
        
        test_e = np.sum(np.square(boston_Y_test- np.dot(boston_X_test,w_cap)))
        test_error.append(test_e/int(506*(1-i)))
        
        total_var = np.sum(np.square(boston_Y_train-np.mean(boston_Y_train)))/int(i*506)
        
        exp_var = np.sum(np.square(np.dot(boston_X_train,w_cap)-np.mean(boston_Y_train)))/int(i*506)
        
        unexp_var = np.sum(np.square(boston_Y_train-np.dot(boston_X_train,w_cap)))/int(i*506)
        
        rsquare_score.append(1-(unexp_var/total_var))
 

       
# Phase 5: Plots  
print("--------------------------Solution 4a------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(l, train_error[0:7], marker='o', s=10, color='blue', label ='Train Error')
ax1.scatter(l, test_error[0:7], marker='o', s=10, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='Number of training examples in fraction')
ax1.set_ylim(0, 35)


ax2 = fig.add_subplot(242)
ax2.scatter(l,train_error[7:14], marker='o', s=10, color='blue', label ='Train Error')
ax2.scatter(l,test_error[7:14], marker='o', s=10, color='red', label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='Number of training examples in fraction')
ax2.set_ylim(0, 35)


ax3 = fig.add_subplot(243)
ax3.scatter(l,train_error[14:21], marker='o', s=10, color='blue',label ='Train Error')
ax3.scatter(l,test_error[14:21], marker='o', s=10, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='Number of training examples in fraction')
ax3.set_ylim(0, 35)

ax4 = fig.add_subplot(244)
ax4.scatter(l,train_error[21:28], marker='o', s=10, color='blue',label ='Train Error')
ax4.scatter(l,test_error[21:28], marker='o', s=10, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='Number of training examples in fraction')
ax4.set_ylim(0, 35)

ax5 = fig.add_subplot(245)
ax5.scatter(l,rsquare_score[0:7], marker='o', s=10, color='black',label='r square value')
ax5.set_ylim(0, 1)
ax5.set(xlabel='Number of training examples in fraction')
ax5.legend(loc="upper right")

ax6 = fig.add_subplot(246)
ax6.scatter(l,rsquare_score[7:14], marker='o', s=10, color='black',label='r square value')
ax6.set_ylim(0, 1)
ax6.set(xlabel='Number of training examples in fraction')
ax6.legend(loc="upper right")

ax7 = fig.add_subplot(247)
ax7.scatter(l,rsquare_score[14:21], marker='o', s=10, color='black',label='r square value')
ax7.set_ylim(0, 1)
ax7.set(xlabel='Number of training examples in fraction')
ax7.legend(loc="upper right")

ax8 = fig.add_subplot(248)
ax8.scatter(l,rsquare_score[21:28], marker='o', s=10, color='black',label='r square value')
ax8.set_ylim(0, 1)
ax8.set(xlabel='Number of training examples in fraction')
ax8.legend(loc="upper right")

plt.show()



#Solution 4b

# Phase 1: Reading data File
boston_data = datasets.load_boston()
boston_X = pd.DataFrame(boston_data.data)
boston_Y = pd.Series(boston_data.target)
boston_X=np.nan_to_num(boston_X)
boston_Y=np.nan_to_num(boston_Y)

#Normalization
boston_X=np.c_[ np.ones(506),boston_X ]
X_cap = boston_X-boston_X.min(0)/boston_X.max(0)-boston_X.min(0)
Y_cap = boston_Y;

# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]
train_error=[]
test_error=[]
rsquare_score=[]

for i in l:
    boston_X_train, boston_X_test, boston_Y_train, boston_Y_test = train_test_split(X_cap, Y_cap, train_size=i,test_size=1-i, random_state=None,shuffle=True)   
    for j in lambda_diff:
# Phase 3: Model Fitting and Prediction
        w_cap = np.dot(np.dot(np.linalg.inv(np.dot(boston_X_train.transpose(),boston_X_train)+ np.dot(j,np.eye(14,14))),boston_X_train.transpose()),boston_Y_train)
# Phase 4: Reporting
        train_e= np.sum(np.square(boston_Y_train - np.dot(boston_X_train,w_cap)))
        train_error.append(train_e/int(506*i))
        
        test_e = np.sum(np.square(boston_Y_test- np.dot(boston_X_test,w_cap)))
        test_error.append(test_e/int(506*(1-i)))
        
        total_var = np.sum(np.square(boston_Y_train-np.mean(boston_Y_train)))/int(i*506)       
        exp_var = np.sum(np.square(np.dot(boston_X_train,w_cap)-np.mean(boston_Y_train)))/int(i*506)
        unexp_var = np.sum(np.square(boston_Y_train-np.dot(boston_X_train,w_cap)))/int(i*506)      
        rsquare_score.append(1-(unexp_var/total_var))
  


#phase5 : Plots
print("--------------------------Solution 4b------------------------------------")
fig = plt.figure(figsize=(20,8))


ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(0, 100)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(0, 100)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(0, 100)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(0, 100)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)

ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()


#Solution 5_1a

# Phase 1: Reading data File
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_Y = diabetes.target

# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
train_error=[]
test_error=[]
rsquare_score=[]
for i in l:
    diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)

# Phase 3: Model Fitting and Prediction
    diabetes_reg = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
    diabetes_reg.fit(diabetes_X_train, diabetes_Y_train)
# Phase 4: Reporting
    diabetes_model_intercept = diabetes_reg.intercept_
    diabetes_model_coeff = diabetes_reg.coef_
    diabetes_Y_pred = diabetes_reg.predict(diabetes_X_test)
    diabetes_Y_train_pred = diabetes_reg.predict(diabetes_X_train)

    diabetes_total_variance = np.var(diabetes_Y_train)
    diabetes_explained_variance = np.var(diabetes_Y_train_pred)
    diabetes_train_error = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
    diabetes_test_error = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
    diabetes_R2_score = r2_score(diabetes_Y_train, diabetes_Y_train_pred)

    train_error.append(diabetes_train_error)
    test_error.append(diabetes_test_error)
    rsquare_score.append(diabetes_R2_score)
    
# Phase 5: Plots
print("--------------------------Solution 5_1a------------------------------------")
fig = plt.figure(figsize=(10,8))
plt.scatter(l, train_error, marker='o', s=30, color='blue', label ='Train Error')
plt.scatter(l, test_error, marker='o', s=30, color='red', label ='Test Error')
plt.scatter(l, rsquare_score, marker='o', s=30, color='black',label='r square value')
plt.legend(loc="upper right")
plt.xlabel('Number of training examples in fraction')
plt.show()


#Solution 5_1b

# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
lambda_diff=[0,0.01,0.1,1]
train_error=[]
test_error=[]
rsquare_score=[]
for i in lambda_diff:
    for j in l:
        diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=1-j,train_size=j, random_state=None,shuffle=True)


# Phase 3: Model Fitting and Prediction
#diabetes_reg = linear_model.LinearRegression()
        diabetes_reg = linear_model.Ridge (alpha = i,copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
        diabetes_reg.fit(diabetes_X_train, diabetes_Y_train)

# Phase 4: Reporting
        diabetes_model_intercept = diabetes_reg.intercept_
        diabetes_model_coeff = diabetes_reg.coef_
        diabetes_Y_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_reg.predict(diabetes_X_train)

        diabetes_total_variance = np.var(diabetes_Y_train)
        diabetes_explained_variance = np.var(diabetes_Y_train_pred)
        diabetes_train_error = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        diabetes_test_error = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        diabetes_R2_score = r2_score(diabetes_Y_train, diabetes_Y_train_pred)

        train_error.append(diabetes_train_error)
        test_error.append(diabetes_test_error)
        rsquare_score.append(diabetes_R2_score)
        
# Phase 5: Plots 
print("--------------------------Solution 5_1b------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(l, train_error[0:7], marker='o', s=10, color='blue', label ='Train Error')
ax1.scatter(l, test_error[0:7], marker='o', s=10, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='Number of training examples in fraction')
ax1.set_ylim(0, 4000)


ax2 = fig.add_subplot(242)
ax2.scatter(l,train_error[7:14], marker='o', s=10, color='blue', label ='Train Error')
ax2.scatter(l,test_error[7:14], marker='o', s=10, color='red', label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='Number of training examples in fraction')
ax2.set_ylim(0, 4000)


ax3 = fig.add_subplot(243)
ax3.scatter(l,train_error[14:21], marker='o', s=10, color='blue',label ='Train Error')
ax3.scatter(l,test_error[14:21], marker='o', s=10, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='Number of training examples in fraction')
ax3.set_ylim(0, 4000)

ax4 = fig.add_subplot(244)
ax4.scatter(l,train_error[21:28], marker='o', s=10, color='blue',label ='Train Error')
ax4.scatter(l,test_error[21:28], marker='o', s=10, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='Number of training examples in fraction')
ax4.set_ylim(0, 4000)

ax5 = fig.add_subplot(245)
ax5.scatter(l,rsquare_score[0:7], marker='o', s=10, color='black',label='r square value')
ax5.set_ylim(0, 1)
ax5.set(xlabel='Number of training examples in fraction')
ax5.legend(loc="upper right")

ax6 = fig.add_subplot(246)
ax6.scatter(l,rsquare_score[7:14], marker='o', s=10, color='black',label='r square value')
ax6.set_ylim(0, 1)
ax6.set(xlabel='Number of training examples in fraction')
ax6.legend(loc="upper right")

ax7 = fig.add_subplot(247)
ax7.scatter(l,rsquare_score[14:21], marker='o', s=10, color='black',label='r square value')
ax7.set_ylim(0, 1)
ax7.set(xlabel='Number of training examples in fraction')
ax7.legend(loc="upper right")

ax8 = fig.add_subplot(248)
ax8.scatter(l,rsquare_score[21:28], marker='o', s=10, color='black',label='r square value')
ax8.set_ylim(0, 1)
ax8.set(xlabel='Number of training examples in fraction')
ax8.legend(loc="upper right")

plt.show()



#Solution 5_2


# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]

train_error=[]
test_error=[]
rsquare_score=[]
mse_score_ridge=[]

for i in l: 
    for j in lambda_diff:
        diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)       
# Phase 3: Model Fitting and Prediction
        diabetes_reg = linear_model.Ridge (alpha = j,copy_X=True, fit_intercept=True, max_iter=None,
                normalize=True, random_state=None, solver='auto', tol=0.001)
        diabetes_reg.fit(diabetes_X_train, diabetes_Y_train) 
        k_fold=KFold(n=len(diabetes_X_train),n_folds=5,shuffle=True)
        mse=0.0
        for train_indices, test_indices in k_fold:
            diabetes_reg.fit(diabetes_X_train[train_indices],diabetes_Y_train[train_indices])
            mse=mse+mean_squared_error(diabetes_Y_train[test_indices],diabetes_reg.predict(diabetes_X_train[test_indices]))
        mse_score_ridge.append((mse/5))
        
# Phase 4: Reporting
        diabetes_model_intercept = diabetes_reg.intercept_
        diabetes_model_coeff = diabetes_reg.coef_
        diabetes_Y_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_reg.predict(diabetes_X_train)

        diabetes_total_variance = np.var(diabetes_Y_train)
        diabetes_explained_variance = np.var(diabetes_Y_train_pred)
        diabetes_train_error = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        diabetes_test_error = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        diabetes_R2_score = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        train_error.append(diabetes_train_error)
        test_error.append(diabetes_test_error)
        rsquare_score.append(diabetes_R2_score)
            
# Phase 5: Plots
print("--------------------------Solution 5_2------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(2500, 5000)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.plot(lambda_diff,mse_score_ridge[11:22],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(2500, 5000)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.plot(lambda_diff,mse_score_ridge[22:33],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(2500, 5000)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.plot(lambda_diff,mse_score_ridge[33:44],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(2500, 5000)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)


ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()


#Solution 5_3

# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]

train_error=[]
test_error=[]
rsquare_score=[]


mse_score_ridge=[]

for i in l: 
    for j in lambda_diff:
        diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=1-i,train_size=i, random_state=None,shuffle=True)
# Phase 3: Model Fitting and Prediction
        diabetes_reg = linear_model.Lasso(alpha=j, fit_intercept=True, normalize=True, precompute=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        diabetes_reg.fit(diabetes_X_train, diabetes_Y_train)
        k_fold=KFold(n=len(diabetes_X_train),n_folds=5,shuffle=True)
        diabetes_reg.set_params(alpha=j)
        mse=0.0
        for train_indices, test_indices in k_fold:
            diabetes_reg.fit(diabetes_X_train[train_indices],diabetes_Y_train[train_indices])
            mse=mse+mean_squared_error(diabetes_Y_train[test_indices],diabetes_reg.predict(diabetes_X_train[test_indices]))
        mse_score_ridge.append((mse/5))

# Phase 4: Reporting
        diabetes_model_intercept = diabetes_reg.intercept_
        diabetes_model_coeff = diabetes_reg.coef_
        diabetes_Y_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_reg.predict(diabetes_X_train)

        diabetes_total_variance = np.var(diabetes_Y_train)
        diabetes_explained_variance = np.var(diabetes_Y_train_pred)
        diabetes_train_error = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        diabetes_test_error = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        diabetes_R2_score = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        train_error.append(diabetes_train_error)
        test_error.append(diabetes_test_error)
        rsquare_score.append(diabetes_R2_score)

# Phase 5: Plots
print("--------------------------Solution 5_3------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(0, 9000)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(0, 9000)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(0, 9000)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.plot(lambda_diff,mse_score_ridge[0:11],linestyle='dashed',marker='o', color='red',label ='K fold Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(0, 9000)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)

ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()



#Solution 5_41

# Phase 1: Reading data File
diabetes_data = datasets.load_diabetes()
diabetes_X = pd.DataFrame(diabetes_data.data)
diabetes_Y = pd.Series(diabetes_data.target)
diabetes_X=np.nan_to_num(diabetes_X)
diabetes_Y=np.nan_to_num(diabetes_Y)

#Normalization
diabetes_X=np.c_[ np.ones(442),diabetes_X ]
X_cap = diabetes_X-diabetes_X.min(0)/diabetes_X.max(0)-diabetes_X.min(0)
Y_cap = diabetes_Y;
#X_cap=(diabetes_X-np.mean(diabetes_X))/np.var(diabetes_X)
#Y_cap=(diabetes_Y-np.mean(diabetes_Y))/np.var(diabetes_Y)


# Phase 2: Train and Test Split
l=[.50,.60,.70,.80,.90,.95,.99]
lambda_diff=[0,0.01,0.1,1]
train_error=[]
test_error=[]
rsquare_score=[]
for i in l:
    diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(X_cap, Y_cap, train_size=i,test_size=1-i, random_state=None,shuffle=True)
    for j in lambda_diff:
# Phase 3: Model Fitting and Prediction
        w_cap = np.dot(np.dot(np.linalg.inv(np.dot(diabetes_X_train.transpose(),diabetes_X_train)+ np.dot(j,np.eye(11,11))),diabetes_X_train.transpose()),diabetes_Y_train)
# Phase 4: Reporting 
        train_e= np.sum(np.square(diabetes_Y_train - np.dot(diabetes_X_train,w_cap)))
        train_error.append(train_e/int(506*i))
        
        test_e = np.sum(np.square(diabetes_Y_test- np.dot(diabetes_X_test,w_cap)))
        test_error.append(test_e/int(506*(1-i)))
        
        total_var = np.sum(np.square(diabetes_Y_train-np.mean(diabetes_Y_train)))/int(i*506)
        
        exp_var = np.sum(np.square(np.dot(diabetes_X_train,w_cap)-np.mean(diabetes_Y_train)))/int(i*506)
        
        unexp_var = np.sum(np.square(diabetes_Y_train-np.dot(diabetes_X_train,w_cap)))/int(i*506)
        
        rsquare_score.append(1-(unexp_var/total_var))

# Phase 5: Plots   
print("--------------------------Solution 5_41------------------------------------")
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(241)
ax1.scatter(l, train_error[0:7], marker='o', s=10, color='blue', label ='Train Error')
ax1.scatter(l, test_error[0:7], marker='o', s=10, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='Number of training examples in fraction')
ax1.set_ylim(0, 5000)


ax2 = fig.add_subplot(242)
ax2.scatter(l,train_error[7:14], marker='o', s=10, color='blue', label ='Train Error')
ax2.scatter(l,test_error[7:14], marker='o', s=10, color='red', label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='Number of training examples in fraction')
ax2.set_ylim(0, 5000)


ax3 = fig.add_subplot(243)
ax3.scatter(l,train_error[14:21], marker='o', s=10, color='blue',label ='Train Error')
ax3.scatter(l,test_error[14:21], marker='o', s=10, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='Number of training examples in fraction')
ax3.set_ylim(0, 5000)

ax4 = fig.add_subplot(244)
ax4.scatter(l,train_error[21:28], marker='o', s=10, color='blue',label ='Train Error')
ax4.scatter(l,test_error[21:28], marker='o', s=10, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='Number of training examples in fraction')
ax4.set_ylim(0, 5000)

ax5 = fig.add_subplot(245)
ax5.scatter(l,rsquare_score[0:7], marker='o', s=10, color='black',label='r square value')
ax5.set_ylim(0, 1)
ax5.set(xlabel='Number of training examples in fraction')
ax5.legend(loc="upper right")

ax6 = fig.add_subplot(246)
ax6.scatter(l,rsquare_score[7:14], marker='o', s=10, color='black',label='r square value')
ax6.set_ylim(0, 1)
ax6.set(xlabel='Number of training examples in fraction')
ax6.legend(loc="upper right")

ax7 = fig.add_subplot(247)
ax7.scatter(l,rsquare_score[14:21], marker='o', s=10, color='black',label='r square value')
ax7.set_ylim(0, 1)
ax7.set(xlabel='Number of training examples in fraction')
ax7.legend(loc="upper right")

ax8 = fig.add_subplot(248)
ax8.scatter(l,rsquare_score[21:28], marker='o', s=10, color='black',label='r square value')
ax8.set_ylim(0, 1)
ax8.set(xlabel='Number of training examples in fraction')
ax8.legend(loc="upper right")

plt.show()



#Solution 5_42



# Phase 1: Reading data File
diabetes_data = datasets.load_diabetes()
diabetes_X = pd.DataFrame(diabetes_data.data)
diabetes_Y = pd.Series(diabetes_data.target)
diabetes_X=np.nan_to_num(diabetes_X)
diabetes_Y=np.nan_to_num(diabetes_Y)
diabetes_X=np.c_[ np.ones(442),diabetes_X ]

#Normalization
X_cap = diabetes_X-diabetes_X.min(0)/diabetes_X.max(0)-diabetes_X.min(0)
Y_cap = diabetes_Y;
#X_cap=(diabetes_X-np.mean(diabetes_X))/np.var(diabetes_X)
#Y_cap=(diabetes_Y-np.mean(diabetes_Y))/np.var(diabetes_Y)


# Phase 2: Train and Test Split
l=[0.99,0.90,0.80,0.70]
lambda_diff=[0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]

train_error=[]
test_error=[]
rsquare_score=[]
for i in l:
    diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(X_cap, Y_cap, train_size=i,test_size=1-i, random_state=None,shuffle=True)
    for j in lambda_diff:
# Phase 3: Model Fitting and Prediction
        w_cap = np.dot(np.dot(np.linalg.inv(np.dot(diabetes_X_train.transpose(),diabetes_X_train)+ np.dot(j,np.eye(11,11))),diabetes_X_train.transpose()),diabetes_Y_train)
# Phase 4: Reporting  
        train_e= np.sum(np.square(diabetes_Y_train - np.dot(diabetes_X_train,w_cap)))
        train_error.append(train_e/int(506*i))      
        test_e = np.sum(np.square(diabetes_Y_test- np.dot(diabetes_X_test,w_cap)))
        test_error.append(test_e/int(506*(1-i)))
        
        total_var = np.sum(np.square(diabetes_Y_train-np.mean(diabetes_Y_train)))/int(i*506)      
        exp_var = np.sum(np.square(np.dot(diabetes_X_train,w_cap)-np.mean(diabetes_Y_train)))/int(i*506)        
        unexp_var = np.sum(np.square(diabetes_Y_train-np.dot(diabetes_X_train,w_cap)))/int(i*506)       
        rsquare_score.append(1-(unexp_var/total_var))

#phase5 : Plots
print("--------------------------Solution 5_42------------------------------------")
fig = plt.figure(figsize=(20,8))


ax1 = fig.add_subplot(241)
ax1.scatter(lambda_diff, train_error[0:11], marker='o', s=30, color='blue',label ='Train Error')
ax1.scatter(lambda_diff, test_error[0:11], marker='o', s=30, color='red',label ='Test Error')
ax1.legend(loc="upper right")
ax1.set(xlabel='diffrent value of lambda')
ax1.set_ylim(0, 5500)


ax2 = fig.add_subplot(242)
ax2.scatter(lambda_diff, train_error[11:22], marker='o', s=30, color='blue',label ='Train Error')
ax2.scatter(lambda_diff, test_error[11:22], marker='o', s=30, color='red',label ='Test Error')
ax2.legend(loc="upper right")
ax2.set(xlabel='diffrent value of lambda')
ax2.set_ylim(0, 5500)


ax3 = fig.add_subplot(243)
ax3.scatter(lambda_diff, train_error[22:33], marker='o', s=30, color='blue',label ='Train Error')
ax3.scatter(lambda_diff, test_error[22:33], marker='o', s=30, color='red',label ='Test Error')
ax3.legend(loc="upper right")
ax3.set(xlabel='diffrent value of lambda')
ax3.set_ylim(0, 5500)


ax4 = fig.add_subplot(244)
ax4.scatter(lambda_diff, train_error[33:44], marker='o', s=30, color='blue',label ='Train Error')
ax4.scatter(lambda_diff, test_error[33:44], marker='o', s=30, color='red',label ='Test Error')
ax4.legend(loc="upper right")
ax4.set(xlabel='diffrent value of lambda')
ax4.set_ylim(0, 5500)


ax5 = fig.add_subplot(245)
ax5.scatter(lambda_diff, rsquare_score[0:11], marker='o', s=30, color='black',label='r square value')
ax5.legend(loc="upper right")
ax5.set(xlabel='diffrent value of lambda')
ax5.set_ylim(0, 1)

ax6 = fig.add_subplot(246)
ax6.scatter(lambda_diff, rsquare_score[11:22], marker='o', s=30, color='black',label='r square value')
ax6.legend(loc="upper right")
ax6.set(xlabel='diffrent value of lambda')
ax6.set_ylim(0, 1)

ax7 = fig.add_subplot(247)
ax7.scatter(lambda_diff, rsquare_score[22:33], marker='o', s=30, color='black',label='r square value')
ax7.legend(loc="upper right")
ax7.set(xlabel='diffrent value of lambda')
ax7.set_ylim(0, 1)

ax8 = fig.add_subplot(248)
ax8.scatter(lambda_diff, rsquare_score[33:44], marker='o', s=30, color='black',label='r square value')
ax8.legend(loc="upper right")
ax8.set(xlabel='diffrent value of lambda')
ax8.set_ylim(0, 1)

plt.show()

