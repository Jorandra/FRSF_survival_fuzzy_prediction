# -*- coding: utf-8 -*-
"""
Created on Sun May 31 03:38:50 2020

@author: jandr
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index as ci_lifelines
from sklearn.model_selection import KFold

from fuzzy_mul_random_survival_forest import RandomSurvivalForest as JORGE
from random_survival_forest import RandomSurvivalForest as J_RSF
from sksurv.linear_model import CoxPHSurvivalAnalysis
from skranger.ensemble import RangerForestSurvival

#10 x10 CROSS VALIDATION
tipo='\CV_m20_s100_d50_t10_FSF_v3.csv'

trees_number=10
bucles=10
boxes=5
samples=100 
dead=50   
compartive_results_test=pd.DataFrame({}, index=range(bucles)) #bucles 

#Import data

X_train=pd.read_csv(r'C:\X_train.csv')
Y_train=pd.read_csv(r'C:\y_train.csv')
Y_train=np.array(Y_train[['time','evento']])

Xt=pd.DataFrame(X_train) 

Xt1=pd.DataFrame(X_train) #
y_t=pd.DataFrame(Y_train)


scaler=MinMaxScaler()
Xt = scaler.fit_transform(Xt)
Xt=np.around(Xt, 4)

# Variables out of the fuzziness
dicoto=[1,2,5,6,7,8,9,10]

t=1
k=-1

for k in range(bucles):    
    kf = KFold(n_splits=boxes,random_state=k,shuffle=True)
    kf.get_n_splits(Xt)
    t+=1
    #k=-1

    for train_index, test_index in kf.split(Xt):
        k+=1 #serian50 random states
        random_state = k
        print(k)


        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
            
        y_trainf=pd.DataFrame(y_train)
        y_trainf['time']=y_trainf.iloc[:,0]
        
        y_testf=pd.DataFrame(y_test)
        y_testf['time']=y_trainf.iloc[:,0]
        
        
        X_testf=pd.DataFrame(X_test)
        X_trainf=pd.DataFrame(X_train)
        
        
        y_trainf['time']=y_trainf['time'].astype('int')
        y_testf['time']=y_testf['time'].astype('int')
        
        y_trainf['cens']=y_trainf.loc[:,1]
        y_testf['cens']=y_testf.loc[:,1]
        
        X_testf=pd.DataFrame(X_test)
        X_trainf=pd.DataFrame(X_train)
        
        
        y_trainf=y_trainf[['time','cens']]
        y_testf=y_testf[['time','cens']]
        
        y_trainf.loc[y_trainf.time<0,'time']=0
        y_testf.loc[y_testf.time<0,'time']=0
        
        
        
        ###############   ####### ################  #######    ################  #######   
        #FUZZY MODEL
        ################  ####### ################  #######    ################  #######   
        
        rsf = JORGE(n_estimators=trees_number,random_state=random_state,unique_deaths=dead, min_leaf=samples,dicoto=dicoto,n_jobs=-1)
                                                        #=None #random_state
        rsf.fit(X_trainf,y_trainf)
           
        y_pred_fuzzy_test_A = rsf.predict_fuzzy(X_testf)
        
        predFU_test=pd.DataFrame({'final':y_pred_fuzzy_test_A}, index=range(len(X_testf))) 
        c_index_fuzzy_A= ci_lifelines(y_testf.loc[:,'time'], -predFU_test.loc[:,'final'], y_testf.loc[:,'cens'])
        
        rsf=0
        
        ###############   ####### ################  #######    ################  #######   
        # Random Survival Forest - Log Rank split
        ################  #######   ################  #######    ################  #######         
        
        rsforest = J_RSF(n_estimators=trees_number,unique_deaths=dead, min_leaf=samples,random_state=random_state)#, n_jobs=-1)
        
        ytrainj=y_trainf[['cens','time']]
        rsforest.fit(X_trainf,ytrainj)     
        y_pred = rsforest.predict(X_testf)
        predicted_outcome_ori_test = [x.sum() for x in y_pred]
        pred_ori_test=pd.DataFrame({'final':predicted_outcome_ori_test}, index=range(len(X_testf))) 
        
        c_index_ftr_log = ci_lifelines(y_testf['time'], -pred_ori_test['final'], y_testf['cens'])

        ###############   ####### ################  #######    ################  #######   
        # Random Survival Forest- Cind Split
        ################  #######  ################  #######    ################  #######   
        rfs = RangerForestSurvival(num_random_splits=1 ,mtry=3,split_rule="C",min_node_size=samples, n_estimators=10,seed=random_state)
        rfs.fit(X_train, y_train)

        predictions = rfs.predict(X_test) 
        chf = rfs.predict_cumulative_hazard_function(X_test)       

        pred_test=pd.DataFrame({'final':predictions}, index=range(len(X_testf))) 
        c_index_cind= ci_lifelines(y_testf.loc[:,'time'], -pred_test.loc[:,'final'], y_testf.loc[:,'cens'])

        ###############   ####### ################  #######    ################  #######   
        # Cox Proportional Hazard
        ################  #######  ################  #######    ################  #######   
        
        y_train = np.zeros(y_trainf.shape[0], dtype={'names':('Censurado', 'time'),
                         'formats':('?',  '<f8')})
        
        y_test = np.zeros(y_testf.shape[0], dtype={'names':('Censurado', 'time'),
                         'formats':('?',  '<f8')})
        
        y_train['Censurado']=y_trainf.cens
        y_train['time']=y_trainf.time
        y_test['Censurado']=y_testf.cens
        y_test['time']=y_testf.time        
        
        estimator = CoxPHSurvivalAnalysis()

        estimator.fit(X_train, y_train)
   
        cox_cind=estimator.score(X_test, y_test)
        

        ##############    ##### ################  #######    ################  #######   
        # RESULTS
        ###############    ####### ################  #######    ################  #######   

        compartive_results_test.loc[t,str(k)+'FRSF']=c_index_fuzzy_A
        compartive_results_test.loc[t,str(k)+'RSF_log']=c_index_ftr_log
        compartive_results_test.loc[t,str(k)+'RSF_cind']=c_index_cind
        compartive_results_test.loc[t,str(k)+'Cox']=cox_cind
 
        print('TEST DATA SET \n',compartive_results_test)
        
        compartive_results_test.to_csv(r'C:\FSF'+tipo)
        


print(compartive_results_test.describe())

