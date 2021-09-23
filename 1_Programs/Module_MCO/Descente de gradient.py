# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 01:41:34 2020

@author: Yanis
"""

import numpy as np
import pandas as pd



def Descente_de_Gradiant_MCO(X,Y,L=0.01,ite=10000):
    
 
##### Séparation de la DataFrame en listes de colonnes

    Nobs=len(X)

    list_var=[]
    
    for var in range(0,np.shape(X)[1]):
        
       list_var.append(X[:,var].reshape(Nobs,1))
       
       
##### Génération des coeffs aléatoire pour initialiser la descente

    list_coeff=[]
       
       
    for coeff in range(0,np.shape(X)[1]):
        
        list_coeff.append(np.random.randint(10))
        
        
        
##### Le taux d'apprentisage    
    L = L
    
##### Le nombre d'itération maximum
    ite = ite
    
#### Initialisation de la différence abs((MSE+1)-MSE) pour
    # arréter l'algo
    E=1
    
##### Le nombre a partir du quel on stop l'algo si 
    # abs((MSE+1)-MSE) est plus petit
    Precision =  0.0000000000001 
    
##### Compteur du nombre d'itération nécéssaire a la convergence
    count=0



##### Exécution de l'algorithme de déscente de gradiant
    
    for i in range(ite): 
        
        while E > Precision :
            
            
            list_equa=[]
               
               
            for var_un in range (0,len(list_coeff)):
                
                list_equa.append(list_coeff[var_un]*list_var[var_un])
                
                
            
            Y_pred=np.sum(list_equa,axis=0)
            
            MSE=np.sum( (Y-Y_pred)**2, axis=0 )/Nobs
            
              
            
            gradient=[]
            
            for grad in range (0,len(list_coeff)):
                
                gradient.append((1/Nobs) * sum(-(list_var[grad]) * (Y - Y_pred)))
            
            
            for varia in range (0,len(list_coeff)):
            
                list_coeff[varia]=list_coeff[varia] - (L* gradient[varia])
                
                
                
                
            list_equa=[]
               
               
            for var_un in range (0,len(list_coeff)):
                
                list_equa.append(list_coeff[var_un]*list_var[var_un])
                
                
                    
            Y_pred=np.sum(list_equa,axis=0)    
                
            newMSE=np.sum( (Y-Y_pred)**2, axis=0 )/Nobs
            
            E=abs(newMSE-MSE)
            
            count+=1
            
            
##### Résultats

    coefficients=np.array(list_coeff).reshape(len(list_coeff),1)
            
    Var_Resultat_algo=[coefficients,count,newMSE]
    Index_Resultat_algo=["Coefficients","nb d'itération","MSE"]
    Resultat_algo=pd.Series(data=Var_Resultat_algo, index=Index_Resultat_algo)
                
    print("Coefficients estimés :")
    print(coefficients)
    print("nombre d'itération :"+str(count))
    print("MSE :"+str(newMSE))
    
    return Resultat_algo


    


def normalisation (X,Y,const):
    
    Y=(Y-Y.mean())/Y.std()
    
    if const==True:
        X[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
    else:
        X=(X-X.mean())/X.std()
        
    return X,Y
        
    






def denormalisation_coeffs(X,Y,coeffs):
        
    Nobs=len(X)
    
    list_var=[]
    
    for var in range(1,np.shape(X)[1]):
        
       list_var.append(X[:,var].reshape(Nobs,1))
    
       
    list_correction=[]
    
    for lol in range(1,len(list_var)+1):
        
        a=np.std(Y)/np.std(X[:,lol])
        
        list_correction.append(a)
        
    
    correction=np.array(list_correction).reshape(len(list_correction),1)
    
    coeffs_dénormalisé=coeffs[1:,:]*correction
    
    
    
    
    
    
    list_correc_const=[]
    
    for x in range(1,len(list_var)+1):
        
        a=np.mean(X[:,x])/np.std(X[:,x])
        list_correc_const.append(a)
        
    vec_correc_const=np.array(list_correc_const).reshape(len(list_correc_const),1)
    vec_correc=vec_correc_const*coeffs[1:,:]
    
    theta0=np.std(Y)*(coeffs[0]-np.sum(vec_correc))+np.mean(Y)
    
    coeffs_dénormalisé=np.insert(coeffs_dénormalisé,0,theta0)
    
    return coeffs_dénormalisé

X=(X-X.mean())/X.std()
Y=(Y-Y.mean())/Y.std()


Descente_de_Gradiant_MCO(X,Y,L=0.01,ite=10000)
    

sigY=Y.std()
sigX=X.std()
muY=Y.mean()
muX=Y.mean()
coeffnorm=0.81642273

(sigY/sigX)-(muX/sigX)*coeffnorm+muY
    


    list_var=[]
                
                
                for var in range(0,np.shape(X)[1]):
                    
                   list_var.append(X[:,var].reshape(Nobs,1))
                   
                   
                for c in range(0,len(coefficients)):
                    
                    X[1,0]*(((coefficients[c])*(np.std(Y)/np.std(X)))-(np.mean(X)*np.std(Y)
                      
                     