# -*- coding: utf-8 -*-

"""
Created on Tue Jul 21 15:36:52 2020

@author: Yanis
"""


import os

os.chdir(r"C:\Users\Yanis\01 Projets\01 Python Projects\Projet MCO\Code")

from Module_MCO.MCO import MCO

import pandas as pd
import numpy as np
import statsmodels.api as sm


Anscombe1=pd.read_csv(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\anscombe1.csv")
Anscombe2=pd.read_csv(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\anscombe2.csv")
Anscombe3=pd.read_csv(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\anscombe3.csv")
Anscombe4=pd.read_csv(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\anscombe4.csv")
Pokdex=pd.read_pickle(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\Scrapédex_final.pkl")







Results1=MCO(Data=Anscombe1,varX=["x"],varY=["y"],Const=True,Estimation="AL")


Results2=MCO(Data=Anscombe2,varX=["x"],varY=["y"],Const=True,Estimation="GD",Graphs=True)

Results3=MCO(Data=Anscombe3,varX=["x"],varY=["y"],Const=True,White=True,Graphs=True)

Results4=MCO(Data=Anscombe4,varX=["x"],varY=["y"],Const=True,White=True,Graphs=True,Output=True)

Results5=MCO(Data=Anscombe1,varX=["x","id"],varY=["y"],Const=True,Method="MCO",Estimation="AL")



maskleg=Pokdex["Légendaire"]==0
sp2=Pokdex[maskleg]



resultspoke=MCO(Data=sp2,varX=["Poids","Taille","Taux de capture","A 2 types ?"],varY=["Moyenne des statistiques de base"],Const=True,Estimation="AL")


algo_gd=resultspoke["Résultat_algo_GD"]
a=algo_gd[3]

import matplotlib.pyplot as plt


plt.plot(a)


############################



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

    Var_Resultat=[list_coeff,count,newMSE]
    Index_Resultat=["Coefficients","nb d'itération","MSE"]
    Resultat_algo=pd.Series(data=Var_Resultat, index=Index_Resultat)
        
    print("Coefficients estimés :")
    print(list_coeff)
    print("nombre d'itération :"+str(count))
    print("MSE :"+str(newMSE))
    
    return Resultat_algo
    

Ynorm=resultspoke["Y"]
Xnorm=resultspoke["X"]

Y=resultspoke["Y"]
X=resultspoke["X"]

(np.std(Y)/np.std(X[:,1]))*0.1561


#X[:,1]=(X[:,1]-X[:,1].mean())/np.sqrt(X[:,1].var())
#X[:,2]=(X[:,2]-X[:,2].mean())/np.sqrt(X[:,2].var())
#X[:,3]=(X[:,3]-X[:,3].mean())/np.sqrt(X[:,3].var())

    
    
    
    
testpokeGD=Descente_de_Gradiant_MCO(X,Y)



   
    
    



 
            






















import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
poketest=Pokdex.copy()
poketest["const"]=1


maskleg=poketest["Légendaire"]==0
sp2=poketest[maskleg]

y=sp2["Moyenne des statistiques de base"]
X=sp2[["const","Poids","Taille","Taux de capture","A 2 types ?"]]
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())




Xp1=pd.DataFrame([poketest["const"],poketest["Poids"],poketest["PV_base"]]).to_numpy()
X1=Xp1.transpose()
Y1=poketest["Taille"].to_numpy()
len(X1)
np.size(X1,1)
model=sm.OLS(Y1,X1)
results = model.fit(cov_type='HC1')
print(results.summary())

ypred=results.predict(X1)
rmse=rmse(Y1, ypred)

print("RMSE ="+str(rmse))



import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
Anscombe1test=Anscombe1.copy()
Anscombe1test["const"]=1


Xp1=pd.DataFrame([Anscombe1test["const"],Anscombe1test["x"]]).to_numpy()
X1=Xp1.transpose()
Y1=Anscombe1test["y"].to_numpy()
len(X1)
np.size(X1,1)
model=sm.OLS(Y1,X1)
results = model.fit()
print(results.summary())

ypred=results.predict(X1)
rmse=rmse(Y1, ypred)

print("RMSE ="+str(rmse))



a=results.resid
b=np.diag(a**2)



c=Results1["Residus"].reshape((11,))
d=np.diag(c**2)













Y=resultspoke["Y"]
X=resultspoke["X"]
Nobs=len(X)

list_var=[]

for var in range(0,np.shape(X)[1]):
    
   list_var.append(X[:,var].reshape(Nobs,1))
   

coeffs=(np.array([0.1561,0.2093,-0.5598,0.0494])).reshape(len([0.1561,0.2093,-0.5598,0.09494]),1)

list_correction=[]


for lol in range(1,len(list_var)):
    
    a=np.std(Y)/np.std(X[:,lol])
    
    list_correction.append(a)

correction=np.array(list_correction).reshape(len(list_correction),1)

coeffs_dénormalisé=coeffs*correction






list_correc_const=[]

for x in range(1,len(list_var)):
    
    a=np.mean(X[:,x])/np.std(X[:,x])
    list_correc_const.append(a)
    
vec_correc_const=np.array(list_correc_const).reshape(len(list_correc_const),1)
vec_correc=vec_correc_const*coeffs

theta0=np.std(Y)*(0-np.sum(vec_correc))+np.mean(Y)













Y=resultspoke["Y"]
X=resultspoke["X"]
coeffs=resultspoke["Coefficients"]

test=denormalisation_coeffs(X,Y,coeffs)


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
    
    

Data=Anscombe1
varX=["x"]
varY=["y"]
const=True
Method="MCO"
Estimation="GD"




Pokdex=pd.read_pickle(r"C:\\Users\\Yanis\\01 Projets\\01 Python Projects\\Projet MCO\\Data\\Scrapédex_final.pkl")



maskleg=Pokdex["Légendaire"]==0
sp2=Pokdex[maskleg]



Data=sp2
varX=["Poids","Taille","Taux de capture","A 2 types ?"]
varY=["Moyenne des statistiques de base"]
const=True
Method="MCO"
Estimation="GD" 
White=True 
Graphs=True

























































A=Results2["Résultat_algo_GD"]["evolution_coeffs"]

for i in range(0,len(A)):
    A[]




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri
import seaborn as sns
from scipy import stats
from datetime import date
from datetime import datetime




            
coeffsgraph=pd.DataFrame(A)
coeffsgraph.columns=["Coeffs"]
sns.scatterplot(x=coeffsgraph.index, y="Coeffs" ,data=coeffsgraph,color="g")
plt.title("Evolution des coefficients")
plt.xlabel("Nombre d'itération")
plt.ylabel('MSE')  










