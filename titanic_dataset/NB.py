# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:25:59 2019

@author: jctep
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def naive_bayes(x_train,y_train,x_test,y_test,threshold):
    
    data =  pd.concat([x_train,y_train ], axis=1)
    l = data.drop('passenger_survived_encoded', axis = 1).values.T.tolist()
    key = pd.DataFrame(list(product(*l)), 
                       columns=data.columns[0:len(data.columns)-1])
    key = key.drop_duplicates()
    key = key.sort_values(by = data.columns[0])
    origin_len = len(key.columns)
    
    for i in range(0, origin_len):
        p = pd.DataFrame(np.array(data.groupby(key.columns[i]).size()/len(data)))
        p.reset_index(level = None, inplace = True)
        p.columns = [key.columns[i], 'p_'+key.columns[i]]
        key = pd.merge(key, p, how = 'left')

        ct = pd.crosstab(data['passenger_survived_encoded'], data[key.columns[i]]).apply(lambda r: r/r.sum(), axis=1).T
        ct.reset_index(level = None, inplace = True)
        ct.columns = [key.columns[i], 'pc_'+key.columns[i]+'_0', 'pc_'+key.columns[i]+'_1']
        key = pd.merge(key, ct, how = 'left')

    key['p_survived_0'] = np.array(data.groupby('passenger_survived_encoded').size()/len(data))[0]
    key['p_survived_1'] = np.array(data.groupby('passenger_survived_encoded').size()/len(data))[1]
    
    prob_0 = 1
    prob_1 = 1
    for i in range(0, origin_len):
        prob_0 *= ((key['pc_'+str(key.columns[i])+'_0']*key['p_survived_0'])/key['p_'+str(key.columns[i])])
        prob_1 *= ((key['pc_'+str(key.columns[i])+'_1']*key['p_survived_1'])/key['p_'+str(key.columns[i])])

    key['p_0'] = prob_0/(prob_0+prob_1)
    key['p_1'] = prob_1/(prob_0+prob_1)
    
  
    pred = pd.merge(x_test, key, how = 'left')
    pred_train = pd.merge(data, key, how = 'left')
    pred['survived_code_pred'] = np.where(pred['p_1'] > threshold, 1, 0)
    pred_train['survived_code_pred'] = np.where(pred_train['p_1'] > threshold, 1, 0)
    pred = np.array(pred['survived_code_pred'])
    pred_train = np.array(pred_train['survived_code_pred'])
    accuracy = accuracy_score(pred,y_test)
    accuracy_train = accuracy_score(pred_train,y_train)
    print("Validation Accuracy: ",accuracy)
    print("Training Accuracy: ",accuracy_train)
    
    
    #y_pred = decision_tree.predict(x_test)
    #accuracy = accuracy_score(y_pred,y_test)
    error = 1 - accuracy
    recall = recall_score(pred,y_test)
    precision = precision_score(pred, y_test)
    f1_sc = f1_score(pred,y_test)

    dic = dict()
    dic['Model']='naive_bayes'
    dic['Accuracy']=accuracy
    dic['Recall']= recall
    dic['Precision']=precision
    dic['F1_Score']=f1_sc
    description = "variables:"+ str(data.columns.values)+ "_accuracy_"+str(accuracy)+"_threshold_"+str(threshold)+".pkl"
    dic['Description'] = description

   
    
    #joblib.dump(description)
    #return(dic)
    
    
    return(pred)