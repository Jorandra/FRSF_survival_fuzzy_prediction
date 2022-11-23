import multiprocessing
from sklearn.utils.fixes import _joblib_parallel_args
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines.utils import concordance_index as ci_lifelines


from .survivaltree import SurvivalTree
from .scoring import concordance_index


class RandomSurvivalForest:

    def __init__(self, n_estimators=100, min_leaf=20, unique_deaths=1,
                 n_jobs=None, parallelization_backend="multiprocessing", random_state=None, dicoto=[]):
        """
        A Random Survival Forest is a prediction model especially designed for survival analysis.
        :param n_estimators: The numbers of trees in the forest.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param random_state: The random state to create reproducible results.
        :param n_jobs: The number of jobs to run in parallel for fit. None means 1.
        """
        self.n_estimators = n_estimators
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.n_jobs = n_jobs
        self.parallelization_backend = parallelization_backend
        self.random_state = random_state
        self.bootstrap_idxs = None
        self.bootstraps = []
        self.oob_idxs = None
        self.oob_score = None
        self.trees = []
        self.random_states = []
        self.timeline = None
        self.dicoto=dicoto        


    def fit(self, x, y):
        """
        Build a forest of trees from the training set (X, y).
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event
        in the second with the shape [n_samples, 2]
        :return: self: object
        """
        self.timeline = range(y.iloc[:, 0].min(), y.iloc[:, 0].max(), 1)
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs is None:
            self.n_jobs = 1
        self.random_states = np.random.RandomState(seed=self.random_state).randint(0, 2 ** 31 - 1, self.n_estimators)
        self.bootstrap_idxs = self.draw_bootstrap_samples(x)
        
        trees = Parallel(n_jobs=self.n_jobs,**_joblib_parallel_args(prefer='threads'))(delayed(self.create_tree)(x, y, i)
                                                                                   for i in range(self.n_estimators))
        
        #trees = Parallel(n_jobs=self.n_jobs, backend=self.parallelization_backend)(delayed(self.create_tree)(x, y, i)
        #                                                                           for i in range(self.n_estimators))



        for i in range(len(trees)):
            if trees[i].prediction_possible:
                self.trees.append(trees[i])
                self.bootstraps.append(self.bootstrap_idxs[i])
                
        self.oob_score = self.compute_oob_score(x, y)
        self.fuzzy_score=self.fuzzy_ensemble_chf(x, y)
        
        return self

    def create_tree(self, x, y, i):
        """
        Grows a survival tree for the bootstrap samples.
        :param y: label data frame y with survival time as the first column and event as second
        :param x: feature data frame x
        :param i: Indices
        :return: SurvivalTree
        """
        n_features =  int(round(np.sqrt(x.shape[1]), 0))#no subset split e.variables  int(round((x.shape[1]), 0)) #int(round(np.sqrt(x.shape[1]), 0))
        if self.random_state is None:
            f_idxs = np.random.permutation(x.shape[1])[:n_features]
        else:
            f_idxs = np.random.RandomState(seed=self.random_states[i]).permutation(x.shape[1])[:n_features]

        tree = SurvivalTree(x=x.iloc[self.bootstrap_idxs[i], :], y=y.iloc[self.bootstrap_idxs[i], :],
                            f_idxs=f_idxs, n_features=n_features, dicoto=self.dicoto,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            random_state=self.random_states[i], timeline=self.timeline)

        return tree

    def compute_oob_ensembles(self, xs):
        """
        Compute OOB ensembles.
        :return: List of oob ensemble for each sample.
        """
        results = [compute_oob_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees,
                                            bootstraps=self.bootstraps) for sample_idx in range(xs.shape[0])]
        #oob_ensemble_chfs = [i for i in results if not i.empty]
        return results#oob_ensemble_chfs

    def compute_oob_score(self, x, y):
        """
        Compute the oob score (concordance-index).
        :return: c-index of oob samples
        """
        oob_ensembles = self.compute_oob_ensembles(x)
        #c = concordance_index(y_time=y.iloc[:, 0], y_pred=oob_ensembles, y_event=y.iloc[:, 1])
        c=1
        return c

    def predict(self, xs):
        """
        Predict survival for xs.
        :param xs: The input samples
        :return: List of the predicted cumulative hazard functions.
        """
        ensemble_chfs = [compute_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees)
                         for sample_idx in range(xs.shape[0])]
        return ensemble_chfs

    def draw_bootstrap_samples(self, data):
        """
        Draw bootstrap samples
        :param data: Data to draw bootstrap samples of.
        :return: Bootstrap indices for each of the trees
        """
        boot=True # poner a true si se quiere hacer el OOB en el muestreo.
        bootstrap_idxs = []
        for i in range(self.n_estimators):
            if boot==False:
               bootstrap_idx= np.array(list(range(0,len(data))))
            else:
                no_samples = len(data)
                data_rows = range(no_samples)
                if self.random_state is None:
                    bootstrap_idx = np.random.choice(data_rows, no_samples)
                else:
                    np.random.seed(self.random_states[i])
                    bootstrap_idx = np.random.choice(data_rows, no_samples)
            bootstrap_idxs.append(bootstrap_idx)

        return bootstrap_idxs
    
    def fuzzy_ensemble_chf(self,x,y):
        CHF_aggr_trees=pd.DataFrame({}, index=x.index) 
        CHF_aggr2_trees=pd.DataFrame({}, index=x.index) 
        for h in range(self.n_estimators):
            CHF_aggr_trees['final_aggr_trees'+str(h)]=fuzzy_ensemble_interm(self.trees[h].U_s, self.trees[h].fuzzy_pos)
            CHF_aggr2_trees['final_count_trees'+str(h)]=pd.DataFrame(CHF_aggr_trees.apply(lambda x: 1 if x['final_aggr_trees'+str(h)] > 0 else 0 , axis=1))
            print('cont trees ',h)
            print(CHF_aggr_trees)
            #hago el promedio
        CHF_aggr_trees['final_aggr_trees']=CHF_aggr_trees.sum(axis=1)
        CHF_aggr2_trees['final_count_trees']=CHF_aggr2_trees.sum(axis=1)
        CHF_aggr_trees['final_aggr_trees']=CHF_aggr_trees['final_aggr_trees']/CHF_aggr2_trees['final_count_trees']
        CHF_aggr_trees=CHF_aggr_trees.fillna(0)
            #cindex
        CHF_valboin=CHF_aggr_trees[(CHF_aggr_trees['final_aggr_trees']!=0)]
        indexn1=CHF_valboin.index
        c_index_fuzzy_boin = ci_lifelines(y.loc[indexn1,'time'], -CHF_aggr_trees.loc[indexn1,'final_aggr_trees'], y.loc[indexn1,'cens'])
        
        return CHF_valboin #c_index_fuzzy_boin 
    
    def predict_fuzzy(self, xs):
        """
        Predict survival for xs.
        :param xs: The input samples
        :return: List of the predicted cumulative hazard functions.
        """
        ensemble_chfs = [compute_ensemble_chf_fuzzy(sample_idx=sample_idx, xs=xs, trees=self.trees)
                         for sample_idx in range(xs.shape[0])]
        return ensemble_chfs
   

def compute_ensemble_chf(sample_idx, xs, trees):
    denominator = 0
    numerator = 0
    for b in range(len(trees)):
        sample = xs.iloc[sample_idx].to_list()
        chf = trees[b].predict(sample)
        denominator = denominator + 1
        numerator = numerator + 1 * chf
        print(numerator,'denominador', denominator)
    ensemble_chf = numerator / denominator
    return ensemble_chf


def compute_oob_ensemble_chf(sample_idx, xs, trees, bootstraps):
    denominator = 0
    numerator = 0
    results=[]
    bayes_acum=1
    
    for b in range(len(trees)):
        #print(bootstraps[b])
        print('TREE NUMBER',b)
        if sample_idx not in bootstraps[b]:
            print('TREE NUMBER',b)
            sample = xs.iloc[sample_idx].to_list()
            chf = trees[b].predict_fuzzy(sample,1,results,bayes_acum)
            chf =np.array(chf).mean()
            #chf = trees[b].predict(sample)
            
            denominator = denominator + 1
            numerator = numerator + 1 * chf
            results=[]
    if denominator != 0:
        oob_ensemble_chf = numerator / denominator
    else:
        oob_ensemble_chf = 0#pd.Series()
    return oob_ensemble_chf

#################################################################################
#
#################################################################################

def fuzzy_ensemble_interm(U_s, Fuzzy_pos):
    longit=min(len(U_s),len(Fuzzy_pos))
    CHF_aggr=pd.DataFrame({'final_aggr':[0]}, index=range(longit)) 
    CHF_aggr2=pd.DataFrame({'final_count':[0]}, index=range(longit)) 

    dtre=U_s
    col_name=list(dtre.columns)
    
    num_max_old=0
    for f in range(len(col_name)):
        string=col_name[f]
        if int(string.find('U_s'))>=0:
            continue
       
        elif int(string.find('node')) >=0:
            num_max=int(string[:string.find('node')])    
            num_max=max(num_max, num_max_old)
            num_max_old=num_max
    col=num_max+1
    
    lis_col2=[]
    for i in range(col):
        name=str(i)+'node_chf'+str(1)
        name2=str(i)+'node_chf'+str(2)
        name3=str(i)+'node_chfA'+str(1)
        name4=str(i)+'node_chfA'+str(2)
        lis_col2.append(name)
        lis_col2.append(name2)
        lis_col2.append(name3)
        lis_col2.append(name4)
        
    New_cols=[x for x in lis_col2 if x not in col_name]
    Old_cols=[x for x in lis_col2 if x in col_name]
    New_cols_val=pd.DataFrame(columns=New_cols, index= range(longit))
    New_Data=pd.concat([dtre[Old_cols],New_cols_val],axis=1)
    New_Data=New_Data.fillna(0)
    
    
    CHF_aggr['final_aggr']=New_Data.sum(axis=1)
    for col in (New_Data.columns):
        CHF_aggr2['final_count'+str(col)]=pd.DataFrame(New_Data.apply(lambda x: 1 if x.loc[col] !=0 else 0 , axis=1))
    
    CHF_aggr2['final_count']=CHF_aggr2.sum(axis=1)    
    CHF_aggr['final_aggr']=CHF_aggr['final_aggr']/CHF_aggr2['final_count']   
    CHF_aggr=CHF_aggr.fillna(0)
    return CHF_aggr['final_aggr']

def compute_ensemble_chf_fuzzy(sample_idx, xs, trees):
    
    bayes_acum=1
    denominator = 0
    numerator = 0
    results=[]
    inter=np.array([])
    for b in range(len(trees)):
        
        sample = xs.iloc[sample_idx].to_list()        
        chf = trees[b].predict_fuzzy(sample,1,results,bayes_acum)
        
        chf =np.array(chf)
        inter=np.concatenate((inter,chf))
        
        #AGREGACIÓN OPCIÓN 1 MEDIA DE LAS MEDIAS DE LOS TREES
        chf =np.array(chf).mean()
        print('TREES NUMBER',b,chf)
        print('Resultados',results)
        denominator = denominator + 1
        numerator = numerator + 1 * chf
        results=[]
        #print(numerator)
        #print(chf,numerator, denominator)
    ensemble_chf = numerator / denominator #sacar del for
        
    #OPCIÓN 2: MEDIA DE TODOS LOS RESULTADOS DEL TREE  
    #ensemble_chf=inter.mean()
    return ensemble_chf





