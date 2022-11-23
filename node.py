from lifelines import NelsonAalenFitter
from .splitting import find_split
#from .Parent_node import parent_node
#from .fibonaccisearch import FibonacciSearch
from .tree_helper import select_new_feature_indices
import pandas as pd #JLA
from scipy import optimize
import numpy as np
import random 
import pandas as pd

class Node:

    def __init__(self, x, y, tree, f_idxs, n_features, fuzzy_pos, nodes_count, U_s, direcc, node_parent,beta, bayes, dicoto, unique_deaths=1, min_leaf=20, random_state=None, timeline=None):
        """
        A Node of the Survival Tree.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param tree: The corresponding Survival Tree
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.tree = tree
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.fuzzy_pos= fuzzy_pos
        self.nodes_count=nodes_count
        self.node_parent=node_parent
        self.U_s=U_s
        self.direcc= direcc
        self.unique_deaths = unique_deaths
        self.random_state = random_state
        self.min_leaf = min_leaf
        self.score = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.node_num=0
        self.node_num=self.node_num+1  #JA
        self.chf_terminal = None
        self.terminal_left = False
        self.terminal_right = False
        self.timeline = timeline
        self.beta=beta
        self.bayes=bayes
        self.bayes.columns=[0]
        self.bayes_node=pd.DataFrame({'a':[0]})
        self.y_temp=0
        self.dicoto=dicoto
        #☻self.fuzzy_pos= pd.DataFrame({'left_node':[0],'right_node':[0]}, index=range(0,514)) #JLA
        #pd.DataFrame({'left_node':[0],'right_node':[0]}, index=range(0,514)) #JLA
        self.grow_tree()

    def grow_tree(self):
        """
        Grow tree by calculating the Nodes recursively.
        :return: self
        """
        #dicoto=(2,3,7)
        #dicoto=list(dicoto)
 
        
        self.node_parent=self.nodes_count
        #######**********************************************
        U_s1=pd.DataFrame({'mem_parent':[0]} , index=self.x.index)
       # U_s1['mem_parent']=pd.DataFrame(self.fuzzy_pos.loc[:,str(self.node_parent)+'U_s'])  #str(name_node-1)
        """
        if self.direcc==1:
            #utilizar el memb_L
            x_temp=self.x.iloc[:,:].multiply(self.bayes.loc[self.x.index,'mem_L'],axis=0)
            self.y_temp=self.bayes.loc[self.x.index,'mem_L']*self.y.cens
            U_s1['mem_parent']=self.bayes.loc[self.x.index,'mem_L']

        else:
            x_temp=self.x.iloc[:,:].multiply(self.bayes.loc[self.x.index,'mem_R'],axis=0)
            self.y_temp=self.bayes.loc[self.x.index,'mem_R']*self.y.cens
            U_s1['mem_parent']=self.bayes.loc[self.x.index,'mem_R']
        """
        self.x=self.x.sort_index() # el resultado de la multiplicación siempre devuelve el DFrame ordenado por índice.
        self.y=self.y.sort_index()
        self.bayes=self.bayes.sort_index()
        
        y_temp=self.y.time  #self.bayes*self.y.time         
        
        #y_temp1=y_temp.loc[self.y.index.drop_duplicates()]
        #self.bayes_node=self.bayes
            #utilizar el memb_R
                               
        #######**********************************************        
        
        unique_deaths = self.y.iloc[:, 1].reset_index().drop_duplicates().sum()[1]
        min_samples=self.y.iloc[:, 1].count()
        if unique_deaths <= self.unique_deaths: #if unique_deaths <= self.unique_deaths:
            if self.x.empty==True:   #corregir un error cuando llegaba vacío
               return self
            else:
                #print('unique_deaths')
                self.compute_terminal_node()
            #self.fuzzy_pos.loc[pos_parent_node,str(name_node)]=self.split_val
                return self

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt,fuzzy_corte = find_split(self,self.x,y_temp)
        

        

        if self.split_var is None:
            #print('sin variable')
            self.compute_terminal_node() 
 
            return self

        lhs_idxs_opt_f=self.x.iloc[lhs_idxs_opt, :].index.tolist()
        rhs_idxs_opt_f=self.x.iloc[rhs_idxs_opt, :].index.tolist()
        
        self.fuzzy_pos.loc[lhs_idxs_opt_f,'left_node']=self.fuzzy_pos.loc[lhs_idxs_opt_f,'left_node'] +1#JLA
        self.fuzzy_pos.loc[rhs_idxs_opt_f,'right_node']=self.fuzzy_pos.loc[rhs_idxs_opt_f,'right_node'] +1#JLA
        
        pos_parent_node=lhs_idxs_opt_f +rhs_idxs_opt_f
     
        
        
        self.nodes_count=self.nodes_count+1
        name_node=self.nodes_count

###############################################################################################################################################
###############################################################################################################################################

        #○fixed variables
        alfa=float(self.split_val)
        alfa1=float(self.split_val)
        Variable_split= self.split_var
        #print('split',alfa,'variable',Variable_split)
        U_s_parent=self.U_s
        lhs_idxs_opt_fuzzy=lhs_idxs_opt
        rhs_idxs_opt_fuzzy=rhs_idxs_opt
        num_simul=0
        Temporary_Fuzzy=pd.DataFrame({'inicio':[0]}, index=self.x.index)
        Temporary_Fuzzy_pos=pd.DataFrame({'inicio':[0]}, index=self.x.index)
        
        Temporary_Fuzzy['inicio']=(self.x.iloc[:,Variable_split])#Para las variables dicotomicas no debe de haber ponderacion
        #Membresía
        
        Temporary_Fuzzy['mem_L']= (self.funcion_pertenencia(Temporary_Fuzzy['inicio'],alfa, 0)) #alfa 2º alfa too because delta==0
        Temporary_Fuzzy['mem_R']=1-Temporary_Fuzzy['mem_L']  
        
 
        if self.split_var not in self.dicoto and self.split_val>0: 
        # Compute Node Label
            x_fuzzy=self.x.loc[:,self.split_var].mul(self.bayes, axis=0).to_frame(name=self.split_var)
            
            num_simul=1
            Label_right= self.compute_node_label( self.y.iloc[rhs_idxs_opt, :],self.bayes)
            Label_left= self.compute_node_label( self.y.iloc[lhs_idxs_opt, :],self.bayes)
            
        # U_c
       
  


            #*****METEMOS LA PONDERACIÓN PARA LA MINIMIZACIÓN

            Temporary_Fuzzy['inicio']= x_fuzzy#.iloc[:,:] #para las cuantitativas
        #Membresía
            U_c_inic= pd.DataFrame(self.membresia_clase(Temporary_Fuzzy, Label_left, Label_right))
            Temporary_Fuzzy['mem_L']= (self.funcion_pertenencia(Temporary_Fuzzy['inicio'],alfa, 0)) #2º alfa too because delta==0
            Temporary_Fuzzy['mem_R']=1-Temporary_Fuzzy['mem_L']  
            
            Temporary_Fuzzy_pos['inicio']=Temporary_Fuzzy['inicio']
                #, U_s.iloc[self.x.index,:]
        #Optimization
            bracket_1=self.x.iloc[:,Variable_split].min()#x_temp.iloc[:,Variable_split].min()#
            bracket_2= self.x.iloc[:,Variable_split].max()#x_temp.iloc[:,Variable_split].max()#
            standar_desv=self.x.iloc[:,Variable_split].std() #x_temp.iloc[:,Variable_split].std()#
            iteraciones=self.x.iloc[:,Variable_split].count()
            min_x=bracket_1#max(round(alfa*0.80,4),max(bracket_1,0.001))
            max_x=bracket_2#min(round(alfa*1.20,4),bracket_2)
            intervalo= random.uniform(max(round(alfa*0.95,4),max(bracket_1,0.001)),min(round(alfa*1.05,4),bracket_2))#alfa*1.25#+(bracket_2-bracket_1)/10
            
            intervalo= max(0,random.normalvariate(alfa,standar_desv))
            
                        
            #minimum = optimize.minimize(self.optimization_fibonacci,args=(U_c_inic,U_s1,Temporary_Fuzzy,lhs_idxs_opt_fuzzy,rhs_idxs_opt_fuzzy,alfa),x0=intervalo, method = "L-BFGS-B", options={'maxiter':int(iteraciones*0.20)}, bounds=[(min_x, max_x)])#bounds=[(min(bracket_1,0.001), bracket_2)])      
            #minimum_x= minimum.x
            
            alfa=self.split_val#round(float(minimum.x),4) #
            #self.split_val=round(float(minimum.x),4)
            
            """
            ######*************REASIGNACIÓN SEGÚN SPLITTING******
            x_feature1=(self.x.iloc[:,Variable_split])#x_temp.iloc[:,Variable_split]#
            x_feature1 = x_feature1.reset_index(drop=True)#.iloc[:, Variable_split]
            #x_feature1.sort_values(ascending=True, kind="quicksort").unique()
            feature1l = list(x_feature1[x_feature1 <= self.split_val].index)
            feature2r = list(x_feature1[x_feature1 > self.split_val].index)
            lhs_idxs_opt_fuzzy=feature1l
            rhs_idxs_opt_fuzzy=feature2r
            """
            
 
            #minimum_x=FibonacciSearch(np.array(self.x.iloc[:,Variable_split]), alfa,int(0.08*self.x.iloc[:,Variable_split].count()))
            #minimum_x=alfa
            minimum_x=fuzzy_corte  #SPLIT GA
            betar=round(float(minimum_x),4)
            #print('Número del nodo',name_node)
            self.beta=round(abs(self.split_val-fuzzy_corte),4)#fuzzy_corte#round(abs(betar-alfa),4) #SPLIT GA 
            #print('x_0_intervalo',intervalo)
            #print( 'alfa ',alfa,'betas_fibonacci',betar)
            #print('inferior', bracket_1, 'superior', bracket_2) 
            #print('split_overlap',alfa -betar,'after',alfa+betar, 'ocupacion',(2*betar)/(bracket_2-bracket_1))    
            #if self.direcc==1:
            Temporary_Fuzzy_pos['inicio']=Temporary_Fuzzy['inicio']
            Temporary_Fuzzy_pos['mem_L']=self.funcion_pertenencia(Temporary_Fuzzy['inicio'], alfa,round(abs(alfa-betar),3))
            Temporary_Fuzzy_pos['mem_R']=1-Temporary_Fuzzy_pos['mem_L']              
            #else:
             #   Temporary_Fuzzy_pos['mem_R']=self.funcion_pertenencia(Temporary_Fuzzy['inicio'], alfa, betar)
             # Temporary_Fuzzy_pos['mem_L']=1-Temporary_Fuzzy_pos['mem_R'] 
                    
            #OBTENER POSICIONES NO REPETIDAS
            Filter_node=Temporary_Fuzzy_pos[(Temporary_Fuzzy_pos['mem_L']<1)&(Temporary_Fuzzy_pos['mem_L']>0)]
            Filter_node1=Filter_node.index
            New_indices= list(set(Filter_node1.to_list())) #posiciones no repetidos
            #pasar los índices a posiciones de x.
            idxs_new=list(self.x.index.get_indexer_for(New_indices)) #busca la posición en las x_i NO el índice
            #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.
            New_pos_l=[x for x in idxs_new if x not in lhs_idxs_opt_fuzzy]
            New_pos_r=[x for x in idxs_new if x not in rhs_idxs_opt_fuzzy]
            #duplicamos los que tienen probabilidad de pertenencia a ambos nodos          
            lhs_idxs_opt_fuzzy=lhs_idxs_opt_fuzzy+New_pos_l
            rhs_idxs_opt_fuzzy=rhs_idxs_opt_fuzzy+New_pos_r
            
            #OBTENER ÍNDICES NO REPETIDOS
            Filter_node2=self.x.loc[New_indices, :].index.to_list()#.drop_duplicates().to_list()
            New_indices2= list((Filter_node2)) #indices no repetidos
            #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.
            New_pos_l2=[x for x in New_indices2 if x not in lhs_idxs_opt_f]
            New_pos_r2=[x for x in New_indices2 if x not in rhs_idxs_opt_f]
          
            lhs_idxs_opt_f=lhs_idxs_opt_f+New_pos_l2
            rhs_idxs_opt_f=rhs_idxs_opt_f+New_pos_r2
            
        
        else:
           # if self.direcc==1:
            Temporary_Fuzzy_pos['mem_L']= (self.funcion_pertenencia(Temporary_Fuzzy['inicio'],alfa, 0)) #2º alfa too because delta==0
            Temporary_Fuzzy_pos['mem_R']=1-Temporary_Fuzzy_pos['mem_L']
            #lhs_idxs_opt_f=lhs_idxs_opt
            #rhs_idxs_opt_f=rhs_idxs_opt
            #print('Número del nodo',name_node)
            #print('betas_optimizacion', 'alfa ',alfa)
        

        #Las que tienen membresía izquierda o derecha<1
        #Para continuar el tree growing
        lhs_idxs_opt=lhs_idxs_opt_fuzzy
        rhs_idxs_opt=rhs_idxs_opt_fuzzy
        
        #Para AÑADIR LOS ÍNDICES
        """
        interm=self.fuzzy_pos.loc[lhs_idxs_opt_f,str(self.node_parent)+'U_s'].index.drop_duplicates()
        interm_2=self.fuzzy_pos.loc[rhs_idxs_opt_f,str(self.node_parent)+'U_s_right'].index.drop_duplicates()
        """
        interm_3=Temporary_Fuzzy_pos.loc[lhs_idxs_opt_f,'mem_L'].index.drop_duplicates()
        interm_4=Temporary_Fuzzy_pos.loc[rhs_idxs_opt_f,'mem_R'].index.drop_duplicates()

        MintermL=Temporary_Fuzzy_pos.loc[interm_3,'mem_L'].reset_index(drop=False)
        MintermL.drop_duplicates('index',inplace = True)
        MintermL.index=MintermL['index']
        
        MintermR=Temporary_Fuzzy_pos.loc[interm_4,'mem_R'].reset_index(drop=False)
        MintermR.drop_duplicates('index',inplace = True)
        MintermR.index=MintermR['index']
        #Asignación del U_s['mem']
        """
        self.fuzzy_pos.loc[lhs_idxs_opt_f,str(name_node)+'U_s']=self.fuzzy_pos.loc[interm,str(self.node_parent)+'U_s']*MintermL['mem_L'] #JLA
        self.fuzzy_pos.loc[rhs_idxs_opt_f,str(name_node)+'U_s_right']=self.fuzzy_pos.loc[interm_2,str(self.node_parent)+'U_s_right']*MintermR['mem_R'] #JLA
        
        if self.direcc==1:
    
            U_s_parent=pd.DataFrame(self.U_s.loc[:,str(self.node_parent)+'U_s_left'])  #(name_node-1)
            self.U_s.loc[interm,'U_s_left']=U_s_parent.loc[interm,str(self.node_parent)+'U_s_left'].mul(MintermL['mem_L'],1)#JLA
            self.U_s.loc[interm_2,'U_s_right']=U_s_parent.loc[interm_2,str(self.node_parent)+'U_s_left'].mul(MintermR['mem_R'],1) #JLA
            self.U_s.loc[interm,str(name_node)+'U_s_left']=self.U_s.loc[interm,'U_s_left']
            self.U_s.loc[interm_2,str(name_node)+'U_s_right']=self.U_s.loc[interm_2,'U_s_right']
 

        else:
            #print('pasa direccion derecha')

            U_s_parent=pd.DataFrame(self.U_s.loc[:,str(self.node_parent)+'U_s_right'])
            self.U_s.loc[interm,'U_s_left']=U_s_parent.loc[interm,str(self.node_parent)+'U_s_right'].mul(MintermL['mem_L'],1) #JLA
            self.U_s.loc[interm_2,'U_s_right']=U_s_parent.loc[interm_2,str(self.node_parent)+'U_s_right'].mul(MintermR['mem_R'],1) #JLA   
            self.U_s.loc[interm,str(name_node)+'U_s_left']=self.U_s.loc[interm,'U_s_left']
            self.U_s.loc[interm_2,str(name_node)+'U_s_right']=self.U_s.loc[interm_2,'U_s_right']
 
        #print('random state',self.random_state)
#################################################################################################################################################
        """
        ########***** Los índices de la Temporary están corregidos en MEMB
          # los índices de Bayes están corregidos en interm 6 e interm 7
        self.bayes=self.bayes.to_frame(name='A')
        interm8 = self.bayes.index.drop_duplicates()
        self.bayes=self.bayes.loc[interm8].reset_index(drop=False)
        self.bayes.drop_duplicates('index',inplace = True)
        self.bayes.index=self.bayes['index']

        interm6=self.bayes.loc[lhs_idxs_opt_f].index.drop_duplicates()
        interm7=self.bayes.loc[rhs_idxs_opt_f].index.drop_duplicates()  
        

        self.bayes.loc[lhs_idxs_opt_f,'mem_L']=MintermL.loc[:,'mem_L']*self.bayes.loc[interm6,'A']
        self.bayes.loc[rhs_idxs_opt_f,'mem_R']=MintermR.loc[:,'mem_R']*self.bayes.loc[interm7,'A']
        

            #########*********            

#################################################################################################################################################
    
        self.fuzzy_pos.loc[pos_parent_node,str(name_node)]=self.split_val
        self.fuzzy_pos.loc[pos_parent_node,str(name_node)+'_var']=self.split_var
        #self.fuzzy_pos.loc[pos_parent_node,str(name_node)]=self.node_parent


        lf_idxs, rf_idxs = select_new_feature_indices(self.random_state, self.x, self.n_features)
        
    
        #if self.direcc>=1:
        self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree, lf_idxs,
                            self.n_features,self.fuzzy_pos, self.nodes_count, self.U_s,1 ,self.node_parent,self.beta,self.bayes.loc[lhs_idxs_opt_f,'mem_L'],self.dicoto, min_leaf=self.min_leaf, random_state=self.random_state, timeline=self.timeline)
            
            #self.direcc=2
                    
        self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree, rf_idxs,
                            self.n_features,self.fuzzy_pos, self.nodes_count,self.U_s, 2,self.node_parent,self.beta,self.bayes.loc[rhs_idxs_opt_f,'mem_R'],self.dicoto, min_leaf=self.min_leaf, random_state=self.random_state, timeline=self.timeline)
        """    
        else:
            self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree, rf_idxs,
                            self.n_features,self.fuzzy_pos, self.nodes_count,self.U_s, 2,self. node_parent, min_leaf=self.min_leaf, random_state=self.random_state, timeline=self.timeline)
            
            self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree, lf_idxs,
                            self.n_features,self.fuzzy_pos, self.nodes_count, self.U_s, 1,self. node_parent, min_leaf=self.min_leaf, random_state=self.random_state, timeline=self.timeline)
        """
        
        
        return self

    def compute_terminal_node(self):
        """
        Compute the terminal node if condition has reached.
        :return: self
        """
        #if self.direcc==1:
        self.terminal_left = True
        #else:
        self.terminal_right = True
        
        self.chf = NelsonAalenFitter()
        t = self.y.iloc[:, 0] #*self.bayes
        e = self.y.iloc[:, 1] #
        #print(self.y)
        self.chf.fit(t, event_observed=e, timeline=self.timeline)
        Fin_chf=self.chf.fit(t, event_observed=e, timeline=self.timeline)
        chf_acum=Fin_chf.cumulative_hazard_.sum()[0]
        chf_acum2=Fin_chf.cumulative_hazard_.max()[0]
        print('CHF', max(chf_acum,100))
        #JA
        #self.node_num=self.node_num+1
        self.fuzzy_pos.loc[self.x.index.tolist(),'node_number']=round(chf_acum2,2)#t.count()#self.node_num #+'LEFT'
        self.fuzzy_pos.loc[self.x.index.tolist(),'node_chf']= round(chf_acum,2)
      
        columnas=list(self.U_s.columns)
        if str(self.nodes_count)+'node_chf'+str(self.direcc) not in columnas:

            self.U_s.loc[self.x.index.tolist(),str(self.nodes_count)+'node_chf'+str(self.direcc)]= round(chf_acum,2)
        
        elif self.U_s.loc[self.x.index.tolist(),str(self.nodes_count)+'node_chf'+str(self.direcc)].sum()==0:
            
            self.U_s.loc[self.x.index.tolist(),str(self.nodes_count)+'node_chf'+str(self.direcc)]= round(chf_acum,2)

        else:
            self.U_s.loc[self.x.index.tolist(),str(self.nodes_count)+'node_chfA'+str(self.direcc)]= round(chf_acum,2)
      
        
        return self


    def predict(self, x):
        """
        Predict the cumulative hazard function if its a terminal node. If not walk through the tree.
        :param x: The input sample.
        :return: Predicted cumulative hazard function if terminal node
        """
        if self.terminal_left or self.terminal_right:
            self.tree.chf = self.chf.cumulative_hazard_
            # print('chf_left',self.tree.chf.sum()[0])
            # print(self.nodes_count)
            # print(self.terminal_left,self.terminal_right)
            self.tree.chf = self.tree.chf.iloc[:, 0]
            return self.tree.chf.dropna()

        else:
            if x[self.split_var] <= self.split_val:
                self.lhs.predict(x)
                # print(self.nodes_count)
                # print('izq_over',self.split_val)
                # print('var',x[self.split_var])
                # print(self.terminal_left,self.terminal_right)
            else:
                self.rhs.predict(x)
                # print(self.nodes_count)
                # print('der_over',self.split_val)
                # print('var',x[self.split_var])
                # print(self.terminal_left,self.terminal_right)
                
    def compute_node_label(self,y,bayes):
        """
        Compute the CHF.
        :return: self
        """
        chf = NelsonAalenFitter()
        t = y.time*bayes.loc[y.index.drop_duplicates()]
        e = y.iloc[:, 1]
        Fin_chf=chf.fit(t, event_observed=e, timeline=self.timeline)
        Node_Label=Fin_chf.cumulative_hazard_.sum()[0]

        return Node_Label

    def optimization_fibonacci(self,beta,U_c_inic,U_s,Temporary_Fuzzy_pos,lhs_idxs_opt_fuzzy,rhs_idxs_opt_fuzzy,alfa):
        """
        función a optimizar
        :return: self
        """
        print('betas_optimizacion',beta, 'alfa ',alfa)
        #print(Temporary_Fuzzy_pos)
        Temporary_Fuzzy_pos['mem_L']= self.funcion_pertenencia(Temporary_Fuzzy_pos['inicio'],alfa, beta)
        Temporary_Fuzzy_pos['mem_R']=1-Temporary_Fuzzy_pos['mem_L']             

        Filter_node=Temporary_Fuzzy_pos[(Temporary_Fuzzy_pos['mem_L']<1)&(Temporary_Fuzzy_pos['mem_L']>0)]
        Filter_node1=Filter_node.index

        New_indices= list(set(Filter_node1.to_list())) #indices no repetidos
        #pasar los índices a posiciones de x.
        idxs_new=list(self.x.index.get_indexer_for(New_indices))
        #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.
        New_pos_l=[x for x in idxs_new if x not in lhs_idxs_opt_fuzzy]
        New_pos_r=[x for x in idxs_new if x not in rhs_idxs_opt_fuzzy]
        #duplicamos los que tienen probabilidad de pertenencia a ambos nodos          
        lhs_idxs_opt_fuzzy=lhs_idxs_opt_fuzzy+New_pos_l
        rhs_idxs_opt_fuzzy=rhs_idxs_opt_fuzzy+New_pos_r

        Filter_node2=self.x.loc[New_indices, :].index.to_list()#.drop_duplicates().to_list()
        New_indices2= list((Filter_node2)) #indices no repetidos
        
        #comparar opt_f si están en la lista, sino están hay que añadirles a la derecha o izq.
        lhs_idxs_opt_f=self.x.iloc[lhs_idxs_opt_fuzzy, :].index.tolist()
        rhs_idxs_opt_f=self.x.iloc[rhs_idxs_opt_fuzzy, :].index.tolist()
        
        New_pos_l2=[x for x in New_indices2 if x not in lhs_idxs_opt_f]
        New_pos_r2=[x for x in New_indices2 if x not in rhs_idxs_opt_f]
      
        lhs_idxs_opt_f=lhs_idxs_opt_f+New_pos_l2
        rhs_idxs_opt_f=rhs_idxs_opt_f+New_pos_r2        
        
        interm_3=Temporary_Fuzzy_pos.loc[lhs_idxs_opt_f,'mem_L'].index.drop_duplicates()
        interm_4=Temporary_Fuzzy_pos.loc[rhs_idxs_opt_f,'mem_R'].index.drop_duplicates()

        MintermL=Temporary_Fuzzy_pos.loc[interm_3,'mem_L'].reset_index(drop=False)
        MintermL.drop_duplicates('index',inplace = True)
        MintermL.index=MintermL['index']
        
        MintermR=Temporary_Fuzzy_pos.loc[interm_4,'mem_R'].reset_index(drop=False)
        MintermR.drop_duplicates('index',inplace = True)
        MintermR.index=MintermR['index']

        bayesfib=self.bayes
        bayesfib=bayesfib.to_frame(name='A')
        interm8 = bayesfib.index.drop_duplicates()
        bayesfib=bayesfib.loc[interm8].reset_index(drop=False)
        bayesfib.drop_duplicates('index',inplace = True)
        bayesfib.index=bayesfib['index']

        interm6=bayesfib.loc[lhs_idxs_opt_f].index.drop_duplicates()
        interm7=bayesfib.loc[rhs_idxs_opt_f].index.drop_duplicates()  
        

        bayesfib.loc[lhs_idxs_opt_f,'mem_L']=MintermL.loc[:,'mem_L']*bayesfib.loc[interm6,'A']
        bayesfib.loc[rhs_idxs_opt_f,'mem_R']=MintermR.loc[:,'mem_R']*bayesfib.loc[interm7,'A']
        
        
        Label_right_pos= self.compute_node_label(self.y.iloc[rhs_idxs_opt_fuzzy, :],bayesfib['mem_R'])
        Label_left_pos= self.compute_node_label( self.y.iloc[lhs_idxs_opt_fuzzy, :],bayesfib['mem_L'])
            
        U_c_pos=pd.DataFrame(self.membresia_clase(Temporary_Fuzzy_pos, Label_left_pos, Label_right_pos))
        #, U_s.iloc[self.x.index,:]
        Minimizar=self.bayes*(U_c_inic['Comb']-U_c_pos['Comb'])**2 #
       # print('minimo_',Minimizar.sum())
        Minim=Minimizar.sum()
        return Minim

    
    def membresia_clase(self, U_c, Label_left, Label_right):
             
        
        U_c['weight_mem_R']=U_c['mem_R']*Label_right
        U_c['weight_mem_L']=U_c['mem_L']*Label_left
        U_c['Comb']=(U_c['weight_mem_R']+U_c['weight_mem_L'])
        U_C= U_c['Comb']
        return U_C



    def funcion_pertenencia(self,x, alfa,beta):
        """
        Compute the CHF.
        :return: self
        """
        #alfa viene del algoritmo
        #beta viene de la optimización es el delta que se añade al alfa
        #x=x.sort_values()
        #print(x)
        if self.direcc==1:
            control_memb_inf=1
            control_memb_sup=0            
        else:
            control_memb_inf=0
            control_memb_sup=1
            
        beta =float(beta)
        x_index=x.index#.get_level_values(0)
       # print(x_index)
        #beta_int=abs(alfa-beta)  ######!!!!!!OJO!!!!! PARA OBTENER EL DELTA
        trans_ini=round(alfa-(beta),5) #round(alfa-(beta_int),5)
        trans_fin=round(alfa+(beta),5)#round(alfa+(beta_int),5)
        rangem=round(float(trans_fin-trans_ini),5)
       # print(rangem)
        memb= []
        for f in range(x.shape[0]):
            #print(x)
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
    
    def predict_fuzzy(self, x,direcc,results,bayes_acum):
        """
        Predict the cumulative hazard function if its a terminal node. If not walk through the tree.
        :param x: The input sample.
        :return: Predicted cumulative hazard function if terminal node
        """
        import pandas as pd
        #print(pd.DataFrame(x))
        #print(x)
        xd=x
        bayes_acumd=bayes_acum
        
        if  self.terminal_left or self.terminal_right:
            #print(self.nodes_count)
            #print(self.terminal_left)
            self.tree.chf = self.chf.cumulative_hazard_
            #print('chf_left',self.tree.chf.sum()[0])
            res=self.tree.chf.sum()[0]
            self.tree.chf = self.tree.chf.iloc[:, 0]
            res= self.tree.chf.dropna()
            res=res.sum()
           # self.terminal_left==False
            results.append(res)
            #print(results)
                        
            return results, self.tree.chf.dropna()
        
        else:

        #if self.beta>0:
            if round(x[self.split_var],5) <= round((self.split_val-self.beta),5):
                 
                 inter=self.lhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
                 bayes_acum=bayes_acum*inter[0]
                 #print('node byes izq',bayes_acum, 'inter',inter[0])
                 if self.split_var in self.dicoto: bayes_acum=1
                 x=[x*bayes_acum  if x not in self.dicoto else x for x in x]
                 #print(x)                
                
                 self.lhs.predict_fuzzy(x,1,results,bayes_acum)
                
                
            elif round(x[self.split_var],5) > round((self.split_val+self.beta),5):
                 
                 inter=self.rhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
                 bayes_acum=bayes_acum*(1-inter[0])
                 #print('node byes der',bayes_acum, 'inter',(1-inter[0]))
                 if self.split_var in self.dicoto: bayes_acum=1
                 x=[x*bayes_acum  if x not in self.dicoto else x for x in x]
                 #print(x)                               
                
                 self.rhs.predict_fuzzy(x,2,results,bayes_acum)
            
            else:
                 inter=self.lhs.funcion_pertenencia(pd.DataFrame(x).loc[self.split_var],self.split_val,self.beta)
                 
                 bayes_acum=bayes_acum*inter[0]
                 #print('node byes both',bayes_acum, 'inter',inter[0])
                 if self.split_var in self.dicoto: bayes_acum=1
                 x=[x*bayes_acum if x not in self.dicoto else x for x in x]
                 #print(x)                
                 
                 self.lhs.predict_fuzzy(x,1,results,bayes_acum)
                 
                 interd=self.rhs.funcion_pertenencia(pd.DataFrame(xd).loc[self.split_var],self.split_val,self.beta)
                 
                 bayes_acumd=bayes_acumd*(1-interd[0])
                 #print('node byes both',bayes_acumd, 'inter',(1-interd[0]))
                 if self.split_var in self.dicoto: bayes_acum=1
                 xd=[xd*bayes_acumd  if x not in self.dicoto else xd for xd in xd]
                 #print(xd)                          
                 
                 
                 self.rhs.predict_fuzzy(xd,2,results,bayes_acum)
            
        """     
        else:
            if x[self.split_var] <= self.split_val:
                print('doble',x[self.split_var])
                self.lhs.predict_fuzzy(x)
            else:
                print('doble',x[self.split_var])
                self.rhs.predict_fuzzy(x) 
        """
        
        """
        else  self.terminal_right and direcc==2:
            print(self.nodes_count)
            print(self.terminal_right)
            self.tree.chf = self.chf.cumulative_hazard_
            print('chf_right',self.tree.chf.sum()[0])
            res=self.tree.chf.sum()[0]
            self.tree.chf = self.tree.chf.iloc[:, 0]
            self.terminal_right==False
            results.append(res)
            print(results)
            return results,self.tree.chf.dropna(),results
        """ 
        