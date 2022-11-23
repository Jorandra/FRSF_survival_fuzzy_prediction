from fastlogranktest import logrank_test
import numpy as np
import pandas as pd
#from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index as ci_lifelines
from .fibonacci_genet import inicio_opt

def find_split(node,x,y_temp):
    """
    Find the best split for a Node.
    :param node: Node to find best split for.
    :return: score of best split, value of best split, variable to split, left indices, right indices.
    """
    score_opt = 0
    split_val_opt = None
    lhs_idxs_opt = None
    rhs_idxs_opt = None
    split_var_opt = None
    """
    for i in node.f_idxs:
        score, split_val, lhs_idxs, rhs_idxs = find_best_split_for_variable(node,x, i)
        if score > score_opt:
            score_opt = score
            split_val_opt = split_val
            lhs_idxs_opt = lhs_idxs
            rhs_idxs_opt = rhs_idxs
            split_var_opt = i

    lhs_idxs1= lhs_idxs_opt
    rhs_idxs1= rhs_idxs_opt
    split_val_opt1=split_val_opt
    """
    
    for i in node.f_idxs:           #          find_best_split_for_variable1
        score, split_val, lhs_idxs, rhs_idxs = find_best_split_for_variable1(node,node.x.loc[:,i].mul(node.bayes, axis=0).to_frame(name=i),y_temp, i)#(node,x,node.y.time,i)##1 c-index
        if score > score_opt:
            score_opt = score
            split_val_opt = split_val
            lhs_idxs_opt = lhs_idxs
            rhs_idxs_opt = rhs_idxs
            split_var_opt = i
    """         
    lhs_idxs=list(set(lhs_idxs1+ lhs_idxs_opt))
    rhs_idxs=list(set(rhs_idxs1 + rhs_idxs_opt))
    
    if split_val_opt1==None or split_val_opt== None:
        split_val_opt=None
    else:    
        split_val_opt=min(split_val_opt1,split_val_opt)
    
    """
    """
    if split_var_opt is not None:
        x_inter=node.x.loc[:,split_var_opt]
        missing_index_dup=x_inter[pd.isnull(x_inter)].index
        missing_index=list(set(missing_index_dup))
        lhs_idxs_opt=lhs_idxs_opt+missing_index
        rhs_idxs_opt=rhs_idxs_opt+missing_index
    """
    #print(split_val_opt)
    if split_val_opt is None or split_val_opt==0 or split_var_opt in node.dicoto:
        fuzzy_corte=0
    else:
        x_fuzzy=node.x.loc[:,split_var_opt].mul(node.bayes, axis=0).to_frame(name=split_var_opt)
        y_fuzzy=pd.concat([y_temp.to_frame(name='time'),node.y.cens.to_frame(name='cens')],axis=1)

        #split_val_opt#
        fuzzy_corte=inicio_opt(x_fuzzy,node.y, lhs_idxs_opt, rhs_idxs_opt, split_val_opt,node.bayes,node.random_state)
    return score_opt, split_val_opt, split_var_opt, lhs_idxs_opt, rhs_idxs_opt,fuzzy_corte

def find_best_split_for_variable(node,x,y_temp, var_idx):
    """
    Find best split for a variable of a Node. Best split for a variable is the split with the highest log rank
    statistics. The logrank_test function of the lifelines package is used here.
    :param node: Node
    :param var_idx: Index of variable
    :return: score, split value, left indices, right indices.
    """
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics(x=x, y=node.y, y_time=y_temp,
                                                              feature=var_idx,
                                                              min_leaf=node.min_leaf, time_line=node.timeline)
    return score, split_val, lhs_idxs, rhs_idxs


def logrank_statistics(x, y, y_time,feature, min_leaf,time_line):
    """
    Compute logrank_test of liflines package.
    :param x: Input samples
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """
    #print(feature, type(feature))
    #print(x)
    x_feature = x.reset_index(drop=True).loc[:, feature]#x.reset_index(drop=True).iloc[:, feature]
    score_opt = 0
    split_val_opt = None
    lhs_idxs = []
    rhs_idxs = []
    
    for split_val in x_feature.sort_values(ascending=True, kind="quicksort").unique():
        feature1 = list(x_feature[x_feature <= split_val].index)
        feature2 = list(x_feature[x_feature > split_val].index)
        if len(feature1) < min_leaf or len(feature2) < min_leaf:
            continue
        """
        durations_a = y.iloc[feature1,0]#time.iloc[feature1]#iloc[feature1, 0]
        event_observed_a = y.iloc[feature1, 1]
        durations_b = y.iloc[feature2,0]
        event_observed_b = y.iloc[feature2, 1]
        results = logrank_test(durations_A=durations_a, durations_B=durations_b,
                               event_observed_A=event_observed_a, event_observed_B=event_observed_b)
        score = results.test_statistic
        """
        
        durations_a = np.array(y_time.iloc[feature1])  #np.array(y.iloc[feature1, 0])
        event_observed_a = np.array(y.iloc[feature1, 1])
        durations_b = np.array(y_time.iloc[feature2])
        event_observed_b = np.array(y.iloc[feature2, 1])
        score = logrank_test(durations_a, durations_b, event_observed_a, event_observed_b)
        
        
        if score > score_opt:
            score_opt = round(score, 5)
            split_val_opt = round(split_val, 5)
            lhs_idxs = feature1
            rhs_idxs = feature2


    return score_opt, split_val_opt, lhs_idxs, rhs_idxs    

def find_best_split_for_variable1(node,x,y_temp, var_idx):
    """
    Find best split for a variable of a Node. Best split for a variable is the split with the highest log rank
    statistics. The logrank_test function of the lifelines package is used here.
    :param node: Node
    :param var_idx: Index of variable
    :return: score, split value, left indices, right indices.
    """
    score, split_val, lhs_idxs, rhs_idxs = logrank_statistics1(x=x, y=node.y, y_time=y_temp,
                                                              feature=var_idx,
                                                              min_leaf=node.min_leaf, time_line=node.timeline)
    return score, split_val, lhs_idxs, rhs_idxs


def logrank_statistics1(x, y, y_time,feature, min_leaf,time_line):
    """
    Compute logrank_test of liflines package.
    :param x: Input samples
    :param y: Labels
    :param feature: Feature index
    :param min_leaf: Minimum number of leafs for each split.
    :return: best score, best split value, left indices, right indices
    """    
    x_feature = x.reset_index(drop=True).loc[:, feature]#x.reset_index(drop=True).iloc[:, feature]
    score_opt = 0
    split_val_opt = None
    lhs_idxs = []
    rhs_idxs = []
    x_inter1=x.reset_index(drop=True)
    missing=x_inter1[pd.isnull(x_inter1).any(axis=1)].index.unique().to_list()
    y['chf']=None
    chf = NelsonAalenFitter()
    for split_val in x_feature.sort_values(ascending=True, kind="quicksort").unique():

        feature1 = list(x_feature[x_feature <= split_val].index)+missing
        feature2 = list(x_feature[x_feature > split_val].index)+missing
        if len(feature1) < min_leaf or len(feature2) < min_leaf:
            continue
        if set(feature1) == set(missing)  or set(feature2)== set(missing):
            continue        
        durations_a = y_time.iloc[feature1]#iloc[feature1, 0]
        event_observed_a = y.iloc[feature1, 1]
        durations_b = y_time.iloc[feature2]
        event_observed_b = y.iloc[feature2, 1]
        #print('feature',x_feature)
        #print('miss',missing)
        Fin_chf=chf.fit(durations_a, event_observed=event_observed_a, timeline=time_line)
        chf_acum1=Fin_chf.cumulative_hazard_.sum()[0]

        Fin_chf=chf.fit(durations_b, event_observed=event_observed_b, timeline=time_line)
        chf_acum2=Fin_chf.cumulative_hazard_.sum()[0]

        y.iloc[feature1,2]=chf_acum1
        y.iloc[feature2,2]=chf_acum2
        todos_idxs= feature1+feature2            
        #c_index= ci_lifelines(y.loc[:,'time'], -y.loc[:,'chf'], y.loc[:,'cens'])
        c_index= ci_lifelines(y.iloc[todos_idxs,0], -y.iloc[todos_idxs,2], y.iloc[todos_idxs,1])

        score = c_index

        if score > score_opt:
            score_opt = round(score, 3)
            split_val_opt = round(split_val, 3)
            lhs_idxs = feature1
            rhs_idxs = feature2
         

    return score_opt, split_val_opt, lhs_idxs, rhs_idxs






