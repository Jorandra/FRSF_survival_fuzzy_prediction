# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:34:22 2022

@author: jandr
"""
import pandas as pd
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from lifelines import NelsonAalenFitter

def inicio_opt(x,y,lhs_idxs_opt,rhs_idxs_opt,alfa,bayes,seedf):
    
    lhs_idxs_opt_fuzzy = lhs_idxs_opt
    rhs_idxs_opt_fuzzy = rhs_idxs_opt
   
    Temporary_fuzzy1_pos=pd.DataFrame({'inicio':[0]}, index=x.index)
    Temporary_fuzzy1_pos['inicio']=x
    Temporary_fuzzy1_pos['mem_L']=funcion_pertenencia(Temporary_fuzzy1_pos['inicio'], alfa, alfa)
    Temporary_fuzzy1_pos['mem_R']=1-Temporary_fuzzy1_pos['mem_L']              
    
    Label_right= compute_node_label( y.iloc[rhs_idxs_opt_fuzzy, :],bayes)
    Label_left= compute_node_label( y.iloc[lhs_idxs_opt_fuzzy, :],bayes)    
    U_c_inic= pd.DataFrame(membresia_clase(Temporary_fuzzy1_pos, Label_left, Label_right))

    lhs_idxs_opt_f=x.iloc[lhs_idxs_opt_fuzzy, :].index.tolist()
    rhs_idxs_opt_f=x.iloc[rhs_idxs_opt_fuzzy, :].index.tolist()

    bayesfib=bayes    
    bayes_inic=bayes
    
    lims=max(0,min((round(x[x.columns[0]].max(),3)-alfa),(alfa-round(x[x.columns[0]].min(),3))))   #max(0,min((round(x[x.columns[0]].max(),3)),(round(x[x.columns[0]].min(),3))))   #
    #print('límites',lims)
    
    #INTERVALO DE LOS GENES DEL AG
    varbound=np.array([[0,lims]])
    #,lhs_idxs_opt_fuzzy1,rhs_idxs_opt_fuzzy1,lhs_idxs_opt_f1,rhs_idxs_opt_f1,bayesfib
    
    algorithm_param = {'max_num_iteration':10,\
                       'population_size':10,\
                       'mutation_probability':0.01,\
                       'elit_ratio': 0.20,\
                       #'crossover_probability': 0.7,\
                       'parents_portion': 0.3,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None} #None
        
    #iter 10,# size10,
    #FUNCIÓN FITNESS C-INDEX   




    def optimization_fibonacci(beta):
        """
        función a optimizar
        :return: self
        """
        #print('betas_optimizacion',beta, 'alfa ',alfa)
        
        Temporary_fuzzy1_pos['mem_L']= funcion_pertenencia(Temporary_fuzzy1_pos['inicio'],alfa, abs(alfa-beta))
        Temporary_fuzzy1_pos['mem_R']=1-Temporary_fuzzy1_pos['mem_L']             
    
        Filter_node=Temporary_fuzzy1_pos[(Temporary_fuzzy1_pos['mem_L']<1)&(Temporary_fuzzy1_pos['mem_L']>0)]
        Filter_node1=Filter_node.index
    
        New_indices= list(set(Filter_node1.to_list())) #indices no repetidos
        #pasar los índices a posiciones de x.
        idxs_new=list(x.index.get_indexer_for(New_indices))
        #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.
        New_pos_l=[x for x in idxs_new if x not in lhs_idxs_opt_fuzzy]
        New_pos_r=[x for x in idxs_new if x not in rhs_idxs_opt_fuzzy]
        #duplicamos los que tienen probabilidad de pertenencia a ambos nodos          
        lhs_idxs_opt_fuzzy1=lhs_idxs_opt_fuzzy+New_pos_l
        rhs_idxs_opt_fuzzy1=rhs_idxs_opt_fuzzy+New_pos_r
    
        Filter_node2=x.loc[New_indices, :].index.to_list()#.drop_duplicates().to_list()
        New_indices2= list((Filter_node2)) #indices no repetidos
        
        #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.

        
        New_pos_l2=[x for x in New_indices2 if x not in lhs_idxs_opt_f]
        New_pos_r2=[x for x in New_indices2 if x not in rhs_idxs_opt_f]
      
        lhs_idxs_opt_f1=lhs_idxs_opt_f+New_pos_l2
        rhs_idxs_opt_f1=rhs_idxs_opt_f+New_pos_r2        
        
        interm_3=Temporary_fuzzy1_pos.loc[lhs_idxs_opt_f1,'mem_L'].index.drop_duplicates()
        interm_4=Temporary_fuzzy1_pos.loc[rhs_idxs_opt_f1,'mem_R'].index.drop_duplicates()
    
        MintermL=Temporary_fuzzy1_pos.loc[interm_3,'mem_L'].reset_index(drop=False)
        MintermL.drop_duplicates('index',inplace = True)
        MintermL.index=MintermL['index']
        
        MintermR=Temporary_fuzzy1_pos.loc[interm_4,'mem_R'].reset_index(drop=False)
        MintermR.drop_duplicates('index',inplace = True)
        MintermR.index=MintermR['index']
    

        bayesfib=bayes.to_frame(name='A')
        interm8 = bayesfib.index.drop_duplicates()
        bayesfib=bayesfib.loc[interm8].reset_index(drop=False)
        bayesfib.drop_duplicates('index',inplace = True)
        bayesfib.index=bayesfib['index']
    
        interm6=bayesfib.loc[lhs_idxs_opt_f1].index.drop_duplicates()
        interm7=bayesfib.loc[rhs_idxs_opt_f1].index.drop_duplicates()  
        
    
        bayesfib.loc[lhs_idxs_opt_f1,'mem_L']=MintermL.loc[:,'mem_L']*bayesfib.loc[interm6,'A']
        bayesfib.loc[rhs_idxs_opt_f1,'mem_R']=MintermR.loc[:,'mem_R']*bayesfib.loc[interm7,'A']
        
        
        Label_right_pos= compute_node_label(y.iloc[rhs_idxs_opt_fuzzy1, :],bayesfib['mem_R'])
        Label_left_pos= compute_node_label( y.iloc[lhs_idxs_opt_fuzzy1, :],bayesfib['mem_L'])
            
        U_c_pos=pd.DataFrame(membresia_clase(Temporary_fuzzy1_pos, Label_left_pos, Label_right_pos))
        #, U_s.iloc[self.x.index,:]
        Minimizar=bayes_inic*(U_c_inic['Comb']-U_c_pos['Comb'])**2 #
       # print('minimo_',Minimizar.sum())
        Minim=Minimizar.sum()
        
        return Minim


    model=ga(function=optimization_fibonacci,dimension=1,variable_type='real',variable_boundaries=varbound,algorithm_parameters=algorithm_param,function_timeout = 80)
    result=model.run(seed=seedf, progress_bar_stream=None,no_plot=True)#,no_plot=True)
           
        
    #print(model.param)
    #convergence=model.report
    return result.variable[0] # model.best_variable[0]

    
    
def funcion_pertenencia(x, alfa,beta):
 
    beta =float(beta)
    x_index=x.index#.get_level_values(0)
   
    #beta_int=abs(alfa-beta)  ######!!!!!!OJO!!!!! PARA OBTENER EL DELTA
    trans_ini=round(alfa-(beta),5) #round(alfa-(beta_int),5)
    trans_fin=round(alfa+(beta),5)#round(alfa+(beta_int),5)
    rangem=round(float(trans_fin-trans_ini),5)
    memb= []
    for f in range(x.shape[0]):
        x_i=x.iloc[f]
        control=0
        if x_i > trans_fin:
            membership= 0.00
        elif np.isnan(x_i):
            membership=0.5
        elif x_i<=trans_ini:
            membership= 1.00
        else:
            
            if x_i<=alfa:
                control=0.5
            membership=control + abs(x_i-trans_ini)/rangem
               
            #membership=(trans_fin-x_i)/rangem
        memb.append(membership)
    membership_Sl=memb
    return membership_Sl

def membresia_clase(U_c, Label_left, Label_right):
         
    
    U_c['weight_mem_R']=U_c['mem_R']*Label_right
    U_c['weight_mem_L']=U_c['mem_L']*Label_left
    U_c['Comb']=(U_c['weight_mem_R']+U_c['weight_mem_L'])
    U_C= U_c['Comb']
    return U_C


def compute_node_label(y,bayes):
    """
    Compute the CHF.
    :return: self
    """
    chf = NelsonAalenFitter()
    t = y.time*bayes.loc[y.index.drop_duplicates()]
    e = y.iloc[:, 1]
    Fin_chf=chf.fit(t, event_observed=e)#, timeline=self.timeline)
    Node_Label=Fin_chf.cumulative_hazard_.sum()[0]

    return Node_Label