from .node import Node
from .splitting import find_split
from .tree_helper import select_new_feature_indices
import pandas as pd
import numpy as np


class SurvivalTree:

    def __init__(self, x, y, f_idxs, n_features, dicoto,unique_deaths=1, min_leaf=20, random_state=None, timeline=None):
        """
        A Survival Tree to predict survival.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.random_state = random_state
        self.score = 0
        self.index = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.prediction_possible = None
        self.timeline = timeline
        self.U_s=pd.DataFrame({'U_s_left':[1],'U_s_right':[1]},index=range(0,x.shape[0])) #JLA
        self.fuzzy_pos=pd.DataFrame({'left_node':[0],'right_node':[0],'node_number':[0],'node_chf':[0],'0U_s':[0],'0U_s_right':[0]},index=range(0,x.shape[0])) #JLA
        self.nodes_count=0
        self.node_parent= 0
        self.beta= 0
        self.bayes=pd.Series(1,index=x.index)#range(0,x.shape[0]))#{'inicio':[0]},,'mem_L':[1],'mem_R':[1]
        self.direcc=1
        self.y_temp=0
        self.dicoto=dicoto
        self.grow_tree()

    def grow_tree(self):
        """
        Grow the survival tree recursively as nodes.
        :return: self
        """
        #print('DICO',self.dicoto)
        unique_deaths = self.y.iloc[:, 1].reset_index().drop_duplicates().sum()[1]
        
        y_temp=self.y.time#self.bayes*self.y.time
        y_temp=y_temp.sort_index()
        
        self.bayes=self.bayes.sort_index()
        self.x=self.x.sort_index() # el resultado de la multiplicación siempre devuelve el DFrame ordenado por índice.
        self.y=self.y.sort_index()

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt, fuzzy_corte= find_split(self,self.x,y_temp)

        self.beta=round(abs(self.split_val-fuzzy_corte),4)#fuzzy_corte

        
        lhs_idxs_opt_f=self.x.iloc[lhs_idxs_opt, :].index.tolist()
        rhs_idxs_opt_f=self.x.iloc[rhs_idxs_opt, :].index.tolist()        
       
        pos_parent_node=lhs_idxs_opt_f +rhs_idxs_opt_f
        
        self.nodes_count=self.nodes_count
        name_node=self.nodes_count

        self.node_parent=0
        
        self.fuzzy_pos.loc[pos_parent_node,str(name_node)]=self.split_val
        self.fuzzy_pos.loc[pos_parent_node,str(name_node)+'_var']=self.split_var
        
        self.fuzzy_pos.loc[lhs_idxs_opt_f,'left_node']=self.fuzzy_pos.loc[lhs_idxs_opt_f,'left_node']+1 #JLA
        self.fuzzy_pos.loc[rhs_idxs_opt_f,'right_node']=self.fuzzy_pos.loc[rhs_idxs_opt_f,'right_node']+1 #JLA
        ##
        self.U_s.loc[:,str(name_node)+'U_s_left']=1
        self.U_s.loc[:,str(name_node)+'U_s_right']=1
        Temporary_Fuzzy_pos=pd.DataFrame({'inicio':[0]}, index=self.x.index) 


        if self.split_var is not None and unique_deaths > self.unique_deaths:
            self.prediction_possible = True



        #############################################################################################################################    
            if self.split_var not in self.dicoto and self.split_val>0:
                #para split fuzzy
                x_fuzzy=self.x.loc[:,self.split_var].mul(self.bayes, axis=0).to_frame(name=self.split_var)
                Temporary_Fuzzy_pos['inicio']= x_fuzzy#.loc[:,self.split_var]
                Temporary_Fuzzy_pos['mem_L']=self.funcion_pertenencia(Temporary_Fuzzy_pos['inicio'], self.split_val, round(abs(self.split_val-fuzzy_corte),3))
                Temporary_Fuzzy_pos['mem_R']=1-Temporary_Fuzzy_pos['mem_L']              
                lhs_idxs_opt_fuzzy=lhs_idxs_opt
                rhs_idxs_opt_fuzzy=rhs_idxs_opt
                
                Filter_node=Temporary_Fuzzy_pos[(Temporary_Fuzzy_pos['mem_L']<1)&(Temporary_Fuzzy_pos['mem_L']>0)]
                Filter_node1=Filter_node.index
                New_indices= list(set(Filter_node1.to_list())) #posiciones no repetidos
                idxs_new=list(self.x.index.get_indexer_for(New_indices)) #busca la posición en las x_i NO el índice     
                New_pos_l=[x for x in idxs_new if x not in lhs_idxs_opt_fuzzy]
                New_pos_r=[x for x in idxs_new if x not in rhs_idxs_opt_fuzzy]       
                lhs_idxs_opt_fuzzy=lhs_idxs_opt_fuzzy+New_pos_l
                rhs_idxs_opt_fuzzy=rhs_idxs_opt_fuzzy+New_pos_r                
                #OBTENER ÍNDICES NO REPETIDOS
                Filter_node2=self.x.loc[New_indices, :].index.to_list()#.drop_duplicates().to_list()
                New_indices2= list((Filter_node2)) #indices no repetidos   
                New_pos_l2=[x for x in New_indices2 if x not in lhs_idxs_opt_f]
                New_pos_r2=[x for x in New_indices2 if x not in rhs_idxs_opt_f]             
                lhs_idxs_opt_f=lhs_idxs_opt_f+New_pos_l2
                rhs_idxs_opt_f=rhs_idxs_opt_f+New_pos_r2
                
                lhs_idxs_opt=lhs_idxs_opt_fuzzy
                rhs_idxs_opt=rhs_idxs_opt_fuzzy
            
            else:
                
                Temporary_Fuzzy_pos['mem_L']= (self.funcion_pertenencia(self.x.iloc[:,self.split_var], self.split_val, 0)) #2º alfa too because delta==0
                Temporary_Fuzzy_pos['mem_R']=1-Temporary_Fuzzy_pos['mem_L']                            



            
            self.bayes=pd.DataFrame(self.bayes,columns=['A'])
            interm_3=Temporary_Fuzzy_pos.loc[lhs_idxs_opt_f,'mem_L'].index.drop_duplicates()
            interm_4=Temporary_Fuzzy_pos.loc[rhs_idxs_opt_f,'mem_R'].index.drop_duplicates()
    
            MintermL=Temporary_Fuzzy_pos.loc[interm_3,'mem_L'].reset_index(drop=False)
            MintermL.drop_duplicates('index',inplace = True)
            MintermL.index=MintermL['index']
            
            MintermR=Temporary_Fuzzy_pos.loc[interm_4,'mem_R'].reset_index(drop=False)
            MintermR.drop_duplicates('index',inplace = True)
            MintermR.index=MintermR['index']
            #self.bayes.loc[lhs_idxs_opt_f,'mem_L']=Temporary_Fuzzy_pos['mem_L']
            #self.bayes.loc[rhs_idxs_opt_f,'mem_R']=Temporary_Fuzzy_pos['mem_R']
        ####################################################################################################################################
            
            
            
            
            
            
            lf_idxs, rf_idxs = select_new_feature_indices(self.random_state, self.x, self.n_features)

            self.lhs = Node(x=self.x.iloc[lhs_idxs_opt, :], y=self.y.iloc[lhs_idxs_opt, :],
                            tree=self, f_idxs=lf_idxs, n_features=self.n_features, dicoto=self.dicoto,fuzzy_pos=self.fuzzy_pos,nodes_count=self.nodes_count, U_s=self.U_s, direcc=self.direcc,  node_parent=self. node_parent, beta=self.beta, bayes=MintermL.loc[:,'mem_L'],
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf, random_state=self.random_state,
                            timeline=self.timeline)

            self.rhs = Node(x=self.x.iloc[rhs_idxs_opt, :], y=self.y.iloc[rhs_idxs_opt, :],
                            tree=self, f_idxs=rf_idxs, n_features=self.n_features, dicoto=self.dicoto,fuzzy_pos=self.fuzzy_pos,nodes_count=self.nodes_count, U_s=self.U_s, direcc=self.direcc,  node_parent=self. node_parent, beta=self.beta, bayes= MintermR.loc[:,'mem_R'],
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf, random_state=self.random_state,
                            timeline=self.timeline)
            

            return self
        else:
            self.prediction_possible = False
            return self


#########################################################################################        
    #para que tenga en cuenta desde el primer split    
    def funcion_pertenencia(self,x, alfa,beta):
        """
        Compute the CHF.
        :return: self
        """
        #alfa viene del algoritmo
        #beta viene de la optimización es el delta que se añade al alfa

        if self.direcc==1:
            control_memb_inf=1
            control_memb_sup=0            
        else:
            control_memb_inf=0
            control_memb_sup=1
            
        beta =float(beta)
        x_index=x.index

        #beta_int=abs(alfa-beta)
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
            #memb.sort()
        #print(memb)
        #print('x',x)
        membership_Sl=memb#pd.DataFrame(memb,index=x_index)
 
        return membership_Sl
####################################################################################






    def predict(self, x):
        """
        Predict survival for x.
        :param x: The input sample.
        :return: The predicted cumulative hazard function.
        """
        if x[self.split_var] <= self.split_val:
            self.lhs.predict(x)
        else:
            self.rhs.predict(x)
        return self.chf

    def predict_fuzzy(self, x, direcc,results,bayes_acum):
        """
        Predict survival for x.
        :param x: The input sample.
        :return: The predicted cumulative hazard function.
        """
        import pandas as pd
        #print(x[self.split_var])
        #print(x)
        #print(self.lhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta))
        bayes_acumd=bayes_acum
        xd=x
        
        if x[self.split_var] <= (self.split_val-self.beta):
            
            inter=self.lhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
            bayes_acum=bayes_acum*inter[0]
            if self.split_var in self.dicoto: bayes_acum=1
            print(bayes_acum)
            x=[x*bayes_acum if x not in self.dicoto else x for x in x]
            print(x)
            
            self.lhs.predict_fuzzy(x,1,results,bayes_acum)
            
        elif x[self.split_var] > (self.split_val+self.beta):
            
            inter=self.rhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
            bayes_acum=bayes_acum*(1-inter[0])
            if self.split_var in self.dicoto: bayes_acum=1
            print(bayes_acum)
            x=[x*bayes_acum  if x not in self.dicoto else x for x in x]
            print(x)

            self.rhs.predict_fuzzy(x,2,results,bayes_acum)
        else:
            
            inter=self.lhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
            bayes_acum=bayes_acum*inter[0]
            if self.split_var in self.dicoto: bayes_acum=1
            print(bayes_acum)
            x=[x*bayes_acum if x not in self.dicoto else x for x in x]
            print(x)
            
            self.lhs.predict_fuzzy(x,1,results,bayes_acum)
            
            inter=self.rhs.funcion_pertenencia(pd.DataFrame(xd).loc[self.split_var],self.split_val,self.beta)
            bayes_acumd=bayes_acumd*(1-inter[0])
            if self.split_var in self.dicoto: bayes_acumd=1
            print(bayes_acumd)
            xd=[xd*bayes_acumd if xd not in self.dicoto  else xd for xd in xd]
            print(xd) 
            
            self.rhs.predict_fuzzy(xd,2,results,bayes_acum)
        
        return results#,self.chf

