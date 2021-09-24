# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:37:04 2020

@author: Yanis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri
import seaborn as sns
from scipy import stats
from datetime import date
from datetime import datetime


def MCO (Data,varX,varY,Const=True,Estimation="AL",White=False,Graphs=False,Output=False,Nb_ite=100000,Learning_rate=0.001,Precision=0.0000000000001):
    
########################## Date et heure ##########################

    today = date.today()
    d1=today.strftime("%d/%m/%Y")
    
    now = datetime.now()
    t1 = now.strftime("%H:%M:%S")
    
########################## Intitialisation constante ##########################
    
    if Const==True:
        try:
            Data.insert(0, "const",1)
        except:
            pass
        varX.insert(0,"const")
        X=Data[varX]
    else:
        X=Data[varX]

########################## Base calcul ##########################
    
    Nobs=len(X)

    X=X.to_numpy()
    
    tX=X.transpose()
    
    Y=Data[varY].to_numpy()
    
    Px=X.dot((np.linalg.inv(tX.dot(X)))).dot(tX)
    
    Mx=np.identity(Nobs)-Px
    
    
    if Const==True:
        df_model=np.size(X,1)-1
    else:
        df_model= np.size(X,1)
    
    df_resid=Nobs-np.size(X,1)
    
########################## Estimation ##########################

############### Moindre carré ordinaire


##### MCO Analitique

    if Estimation=="AL":
        
        MCO=(np.linalg.inv(tX.dot(X))).dot(tX).dot(Y)
        
        Yhat=X.dot(MCO)
        Resid=Y-Yhat
        
        varModel=((Resid.transpose()).dot(Resid))/(df_resid)
        std_err_Model=np.sqrt(varModel)

        varMCO=varModel*(np.linalg.inv(tX.dot(X)))
        std_err_MCO=varMCO.copy()
        for s in range(0,len(MCO)):
            std_err_MCO[s,s]=np.sqrt(std_err_MCO[s,s])

##### MCO Gradient descendant   
        
    elif Estimation=="GD":
        

        
        #### Normalisation des données
        Ynorm=(Y-Y.mean())/Y.std()
        
        Xnorm=X.copy()
        
        if Const==True:
            Xnorm[:,1:]=(Xnorm[:,1:]-Xnorm[:,1:].mean())/Xnorm[:,1:].std()
        else:
            Xnorm=(Xnorm-Xnorm.mean())/Xnorm.std()
            

        
        ##### Séparation de la DataFrame en listes de colonnes
    
        list_var=[]
        
        for var in range(0,np.shape(X)[1]):
            
           list_var.append(Xnorm[:,var].reshape(Nobs,1))
           
           
    ##### Génération des coeffs aléatoire pour initialiser la descente
    
        list_coeff=[]
           
           
        for coeff in range(0,np.shape(X)[1]):
            
            list_coeff.append(0.5)
            
        coeffs=np.asarray(list_coeff).reshape(len(list_coeff),1)
            
            
            
    ##### Le taux d'apprentisage    

        learning_rate = Learning_rate
        L=[]

        for l in range(0,len(coeffs)):
            L.append(learning_rate)

        L=np.asarray(L).reshape(len(L),1)
        
    ##### Le nombre d'itération maximum
        ite = Nb_ite
        
    #### Initialisation de la différence abs((MSE+1)-MSE) pour
        # arréter l'algo
        E=1
        
    ##### Le nombre a partir du quel on stop l'algo si 
        # abs((MSE+1)-MSE) est plus petit
        precision = Precision
        
    ##### Compteur du nombre d'itération nécéssaire a la convergence
        count=0
        list_MSE=[]
        list_Coeffs=[]
    
    
    
        ##### Exécution de l'algorithme de déscente de gradiant
        
        while (count < ite and E > precision)==True:
    
            list_equa=[]
                
            for o in range (0,len(coeffs)):
                
                list_equa.append(coeffs[o]*list_var[o])
                
                
            
            Y_pred=np.sum(list_equa,axis=0)
            
            MSE=np.sum( (Ynorm-Y_pred)**2, axis=0 )/Nobs
            
            list_MSE.append(MSE)
            
            
            if count > 0 :
                E=abs(list_MSE[count-1]-list_MSE[count])
            else:
                pass
                
            

            gradient=[]
            
            for grad in range (0,len(coeffs)):
                
                gradient.append((1/Nobs) * sum(-(list_var[grad]) * (Ynorm - Y_pred)))
                
            gradient=np.asarray(gradient).reshape(len(gradient),1)
            
            
            for varia in range (0,len(coeffs)):
            
                coeffs[varia]=coeffs[varia] - (L[varia,0]* gradient[varia])
                
            
            list_Coeffs.append(coeffs)
            count+=1
            
  

  

                                        
        

    ##### Résultats
    
        Var_Resultat_algo=[coeffs,count,MSE,list_MSE,list_Coeffs]
        Index_Resultat_algo=["Coeffs","itérations","MSE","evolution_MSE","evolution_coeffs"]
        Resultat_algo=pd.Series(data=Var_Resultat_algo, index=Index_Resultat_algo)
        
    
    
    

    
    ##### Dénormalisation des coefficients    
    
        if Const==True:
   
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
            
            MCO=np.insert(coeffs_dénormalisé,0,theta0).reshape(len(coeffs_dénormalisé)+1,1)
            
            
        else:
            pass
        
        
        #### Calcul de la regression avec les coefficients estimés
            
        Yhat=X.dot(MCO)
        Resid=Y-Yhat
    
        varModel=((Resid.transpose()).dot(Resid))/(df_resid)
        std_err_Model=np.sqrt(varModel)

        varMCO=varModel*(np.linalg.inv(tX.dot(X)))
        std_err_MCO=varMCO.copy()
        for s in range(0,len(MCO)):
            std_err_MCO[s,s]=np.sqrt(std_err_MCO[s,s])
                            


########################## Test de student  ##########################
    
    tstats_list=[]
    
    for i in range(0,len(MCO)):
        tstats=(MCO[i])/(std_err_MCO[i,i])
        tstats_list.append(tstats[0])
     
        
    Pvalue_list=[]
    
    for p in range(0,len(tstats_list)):
        Pvalue=2*(stats.norm(0, 1).sf(abs(tstats_list[p])))
        Pvalue_list.append(Pvalue)
        
        
########################## Intervalles de confiance ##########################
    
        

    int_conf_inf_list=[]
    
    for c in range(0,len(MCO)):
        int_conf_inf=(MCO[c]-(1.96*std_err_MCO[c,c])).astype(float)
        int_conf_inf_list.append(int_conf_inf[0])
        
    int_conf_sup_list=[]
    
    for u in range(0,len(MCO)):
        int_conf_sup=(MCO[u]+(1.96*std_err_MCO[u,u]))
        int_conf_sup_list.append(int_conf_sup[0])
        
        
        
######################## White 80's Matrice ##########################
        

    if White==True:
        
        Vnorm=np.linalg.inv(tX.dot(X))
        Whiteplus=tX.dot(np.diag((Resid.reshape((Nobs,))**2))).dot(X)
        VarWhite=(Vnorm.dot(Whiteplus)).dot(Vnorm)

        std_err_MCO_White=np.sqrt(VarWhite)
        std_err_MCO_White=VarWhite.copy()
        for s in range(0,len(MCO)):
            std_err_MCO_White[s,s]=np.sqrt(std_err_MCO_White[s,s])        
    
        White_tstats_list=[]
    
        for i in range(0,len(MCO)):
            White_tstats=(MCO[i])/(std_err_MCO_White[i,i])
            White_tstats_list.append(White_tstats[0])
         
            
        White_Pvalue_list=[]
        
        for p in range(0,len(White_tstats_list)):
            White_Pvalue=2*(stats.norm(0, 1).sf(abs(White_tstats_list[p])))
            White_Pvalue_list.append(White_Pvalue)
            
        
    
        White_int_conf_inf_list=[]
        
        for c in range(0,len(MCO)):
            White_int_conf_inf=(MCO[c]-(1.96*std_err_MCO_White[c,c])).astype(float)
            White_int_conf_inf_list.append(White_int_conf_inf[0])
            
        White_int_conf_sup_list=[]
        
        for u in range(0,len(MCO)):
            White_int_conf_sup=(MCO[u]+(1.96*std_err_MCO_White[u,u]))
            White_int_conf_sup_list.append(White_int_conf_sup[0])
            
            
            
        else:
            pass
    
        
######################## Criteres d'information ##########################
        

    SCE=np.sum( (Yhat-np.mean(Yhat))**2)
    
    SCR=np.sum( (Resid)**2)
    
    SCT=SCE+SCR
    
    R2=(np.var(Yhat))/(np.var(Y))
    
    if Const==True:
        ajustR2= 1 - (Nobs-1)/(df_resid) *(1-R2)
    else:
        ajustR2= 1 - (Nobs/(df_resid)) *(1-R2)
        
        
    Fstat=(R2/(df_model))/((1-R2)/(df_resid))
    
    F_value=stats.f.sf(Fstat,df_model,df_resid)


    MAE=np.sum( abs(Resid), axis=0 )/Nobs  
    MSE=np.sum( (Y-Yhat)**2, axis=0 )/Nobs
    RMSE=np.sqrt(MSE)
    
    
########################## Vraissemblance ##########################

    log_vraissemblance=-0.5*(np.sum(np.log(varModel)+np.log(2*np.pi)+(((Y-X.dot(MCO))**2)/varModel)))
    
    k=np.size(X,1)
    
    AIC=2*k-2*log_vraissemblance

    BIC=-2*log_vraissemblance+np.log(Nobs)*k
    
    
########################## statistique de Durbin et Watson ##########################

    Residmin1=np.empty([len(Resid),1], dtype=float, order='F')
    
    for r in range(0,len(Residmin1)):
        Residmin1[r]=Resid[r-1]
        
    Residudw = np.delete(Resid, 0)
    Residmin1 = np.delete(Residmin1, 0)
    
      
    DW=np.sum((Residudw-Residmin1)**2,axis=0)/(np.sum(Residudw**2,axis=0))
    

########################## Test de Jarque Bera ##########################


    b1_1=(np.sum((Resid-Resid.mean())**3))/Nobs
    b1_2=((np.sum((Resid-Resid.mean())**2))/Nobs)**(3/2)
    b1=b1_1/b1_2
    
    
    b2_1=(np.sum((Resid-Resid.mean())**4))/Nobs
    b2_2=((np.sum((Resid-Resid.mean())**2))/Nobs)**(2)
    b2=b2_1/b2_2
    
    Tstat_JB=Nobs*(((b1**2)/6)+(((b2-3)**2)/24))
    
    Pvalue_JB=stats.chi2.sf(Tstat_JB,2)
          
    
########################## Série résulat final ##########################

  
    Var_Resultat=[X,tX,Px,Mx,Y,MCO,Yhat,Resid,varModel,std_err_Model,varMCO,std_err_MCO,R2,ajustR2,Fstat,F_value,tstats_list,Pvalue_list,int_conf_inf_list,int_conf_sup_list,SCE,SCR,SCT,MAE,MSE,RMSE,log_vraissemblance,AIC,BIC,DW,Tstat_JB,Pvalue_JB]
    Index_Resultat=["X","X'","Px","Mx","Y","Coefficients","Y estimé","Résidus","Variance modele","Ecart-type modele","Variance coeffs","Ecart-type coeffs","R2","R2 ajusté","Fstat","Pvalue Fstat","Tstats","Pvalue Tstats","Borne inf","borne sup","SCE","SCR","SCT","MAE","MSE","RMSE","log-vraissemblance","AIC","BIC","Durbin-Watson","Jarque-Bera","Pvalue-JB"]
    
    if White==True:
        Var_Resultat.extend([VarWhite,std_err_MCO_White,White_tstats_list,White_Pvalue_list,White_int_conf_inf_list,White_int_conf_sup_list])
        Index_Resultat.extend(["Variance robuste","Ecart-type robuste","T-stats robustes","Pvalue robustes","Borne inf robuste","Borne sup robuste"])
        
    else:
        pass
    
    if Estimation=="GD":
        Var_Resultat.append(Resultat_algo)
        Index_Resultat.append("Résultat_algo_GD")
    else:
        pass
        

    Resultat=pd.Series(data=Var_Resultat, index=Index_Resultat)
    
   

    
########################## initialisation du print et du write ##########################

    def print_side_by_side(a, b, size=30, space=4):
        while a or b:
            print(a[:size].ljust(size) + " " * space + b[:size])
            a = a[size:]
            b = b[size:]
            
    def print_side_by_side_by7(a, b, c, d ,e, f, g,  size=30, space=4):
        while a or b:
            print(a[:size].ljust(size) + " " * space + b[:size].ljust(size) + " " * space + c[:size].ljust(size) + " " * space + d[:size].ljust(size)+ " " * space + e[:size].ljust(size)+ " " * space + f[:size].ljust(size)+ " " * space + g[:size].ljust(size))
            a = a[size:]
            b = b[size:]
            c = c[size:]
            d = d[size:]
            e = e[size:]
            f = f[size:]
            g = g[size:]


    def write_side_by_side(file,a, b, size=30, space=4):
        while a or b:
            file.write(a[:size].ljust(size) + " " * space + b[:size])
            a = a[size:]
            b = b[size:]
            
    def write_side_by_side_by7(file,a, b, c, d ,e, f, g,  size=30, space=4):
        while a or b:
            file.write(a[:size].ljust(size) + " " * space + b[:size].ljust(size) + " " * space + c[:size].ljust(size) + " " * space + d[:size].ljust(size)+ " " * space + e[:size].ljust(size)+ " " * space + f[:size].ljust(size)+ " " * space + g[:size].ljust(size))
            a = a[size:]
            b = b[size:]
            c = c[size:]
            d = d[size:]
            e = e[size:]
            f = f[size:]
            g = g[size:]

        
########################## Impression des résultats ##########################

    
    print(" \n \n Régression linéaire ")
    print(" =====================================================================================")
    print_side_by_side(" Var régressé: "+str(varY),"R2: "+str(round(R2,4)))
    print_side_by_side(" Méthode: MCO","R2 ajusté: "+str(round(ajustR2,4)))
    print_side_by_side(" Estimation: "+Estimation,"F-stat: "+str(round(Fstat,4)))
    print_side_by_side(" Date: "+str(d1),"Pvalue-Fstat: "+str(round(F_value,4)))
    print_side_by_side(" Heure: "+str(t1),"log vraissemblance:"+str(round(log_vraissemblance,4)))
    print_side_by_side(" Nobs: "+str(Nobs),"AIC:"+str(round(AIC,4)))
    print_side_by_side(" Dl_Modele: "+str(df_model),"BIC:"+str(round(BIC,5)))
    print_side_by_side(" Dl_Résidu: "+str(df_resid)," ")
    print(" =======================================================================================")
    print(" |   Var   |   Coeff   |   std err   |   Tstat   |   P-val   |   [0.025   |   0.975 ]  |")
    print(" ---------------------------------------------------------------------------------------")
    
    for l in range(0,np.size(X,1)):
        print_side_by_side_by7("   "+varX[l],str(round(MCO[l,0],4)),str(round(std_err_MCO[l,l],4)),str(round(tstats_list[l],4)),str(round(Pvalue_list[l],4)),str(round(int_conf_inf_list[l],4)),str(round(int_conf_sup_list[l],4)),size=9)
        
        #print("    "+varX[l]+"       "+str(round(MCO[l,0],4))+"      "+str(round(std_err_MCO[l,l],4))+"        "+str(round(tstats_list[l],4))+"       "+str(round(Pvalue_list[l],4))+"       ["+str(round(int_conf_inf_list[l],4))+"      "+str(round(int_conf_sup_list[l],4))+"]   ")
    print(" =====================================================================================")
    print_side_by_side("SCE: "+str(round(SCE,4)),"MAE: "+str(round(MAE[0],4)))
    print_side_by_side("SCR: "+str(round(SCR,4)),"MSE: "+str(round(MSE[0],4))) 
    print_side_by_side("SCT: "+str(round(SCT,4)),"RMSE: "+str(round(RMSE[0],4))) 
    print("Durbin et watson: "+str(round(DW,4)))
    print("Jarque-Bera:"+str(round(Tstat_JB,4)))
    print("Pvalue JB:"+str(round(Pvalue_JB,4)))
    if Estimation=="GD":
        print("Nb d'itération: "+str(count))
        

    if White==True:
        print(" \n \n =====================================================================================")
        print("           Estimation avec la matrice de variance-covariance de White (80)")
        print(" =====================================================================================")
        print(" |   Var   | MCO_Coeff |  W std err  |  W Tstat  |  W P-val  |  W [0.025  | W 0.975 ] |")
        print(" ---------------------------------------------------------------------------------------")
    
        for l in range(0,np.size(X,1)):
            print_side_by_side_by7("   "+varX[l],str(round(MCO[l,0],4)),str(round(std_err_MCO_White[l,l],4)),str(round(White_tstats_list[l],4)),str(round(White_Pvalue_list[l],4)),str(round(White_int_conf_inf_list[l],4)),str(round(White_int_conf_sup_list[l],4)),size=9)
            
    else:
        pass
    
    
    
    
########################## Ecriture de l'Output ##########################
    
    
    if Output==True:
        
        Error=None
        count=0
        while Error is None:
            try:
                Path='Résultats_MCO'
                os.mkdir(Path)
                Error="Good"
            except:
                try:
                    Path="Résultat_MCO("+str(count+1)+")"
                    os.mkdir(Path)
                    Error="Good"
                except:
                    count+=1
        
        f=open(Path+"//result_output.txt","w+")
        f.write(" Régression linéaire ")
        f.write(" \n ====================================================================================="+"\n")
        write_side_by_side(f," Var régressé: "+str(varY),"R2: "+str(round(R2,4))+"\n")
        write_side_by_side(f," Méthode: MCO","R2 ajusté: "+str(round(ajustR2,4))+"\n")
        write_side_by_side(f," Estimation: "+Estimation,"F-stat: "+str(round(Fstat,4))+"\n")
        write_side_by_side(f," Date: "+str(d1),"Pvalue-Fstat: "+str(round(F_value,4))+"\n")
        write_side_by_side(f," Heure: "+str(t1),"log vraissemblance:"+str(round(log_vraissemblance,4))+"\n")
        write_side_by_side(f," Nobs: "+str(Nobs),"AIC:"+str(round(AIC,4))+"\n")
        write_side_by_side(f," Dl_Modele: "+str(df_model),"BIC:"+str(round(BIC,5))+"\n")
        write_side_by_side(f," Dl_Résidu: "+str(df_resid)," "+"\n")
        f.write(" \n =======================================================================================")
        f.write(" \n |   Var   |   Coeff   |   std err   |   Tstat   |   P-val   |   [0.025   |   0.975 ]  |")
        f.write(" \n ---------------------------------------------------------------------------------------")
        
        for l in range(0,np.size(X,1)):
            write_side_by_side_by7(f,"\n   "+varX[l],str(round(MCO[l,0],4)),str(round(std_err_MCO[l,l],4)),str(round(tstats_list[l],4)),str(round(Pvalue_list[l],4)),str(round(int_conf_inf_list[l],4)),str(round(int_conf_sup_list[l],4)),size=9)
            
            #print("    "+varX[l]+"       "+str(round(MCO[l,0],4))+"      "+str(round(std_err_MCO[l,l],4))+"        "+str(round(tstats_list[l],4))+"       "+str(round(Pvalue_list[l],4))+"       ["+str(round(int_conf_inf_list[l],4))+"      "+str(round(int_conf_sup_list[l],4))+"]   ")
        f.write(" \n ====================================================================================="+"\n")
        write_side_by_side(f,"SCE: "+str(round(SCE,4)),"MAE: "+str(round(MAE[0],4))+"\n")
        write_side_by_side(f,"SCR: "+str(round(SCR,4)),"MSE: "+str(round(MSE[0],4))+"\n") 
        write_side_by_side(f,"SCT: "+str(round(SCT,4)),"RMSE: "+str(round(RMSE[0],4))+"\n") 
        f.write("Durbin et watson: "+str(round(DW,4))+"\n")
        f.write("Jarque-Bera:"+str(round(Tstat_JB,4))+"\n")
        f.write("Pvalue JB:"+str(round(Pvalue_JB,4))+"\n")

        if White==True:
            f.write(" \n \n =====================================================================================")
            f.write(" \n          Estimation avec la matrice de variance-covariance de White (80)")
            f.write(" \n =====================================================================================")
            f.write(" \n |   Var   | MCO_Coeff |  W std err  |  W Tstat  |  W P-val  |  W [0.025  | W 0.975 ] |")
            f.write(" \n ---------------------------------------------------------------------------------------")
        
            for l in range(0,np.size(X,1)):
                write_side_by_side_by7(f,"\n   "+varX[l],str(round(MCO[l,0],4)),str(round(std_err_MCO_White[l,l],4)),str(round(White_tstats_list[l],4)),str(round(White_Pvalue_list[l],4)),str(round(White_int_conf_inf_list[l],4)),str(round(White_int_conf_sup_list[l],4)),size=9)
                
        else:
            pass
        
        f.close() 
        
    else:
        pass
        
########################## Impression des graphiques ##########################


    if Graphs==True :
        
        sns.set_style("darkgrid")
        
        Ygraph=pd.DataFrame(Y)
        Ygraph.columns=["Y"]
        sns.scatterplot(x=Ygraph.index, y="Y" ,data=Ygraph,color="g")
        plt.title("Nuage de point de la variable dependante")
        plt.xlabel('Index')
        plt.ylabel('Variable dependante')  
        if Output==True:
            plt.savefig(Path+"//Nuage de point de la variable dependante.png")
        else:
            pass
        plt.show()
        plt.close()
        
    
        Yhatgraph=pd.DataFrame(Yhat)
        Residgraph=pd.DataFrame(Resid)
        graphbase=pd.concat([Yhatgraph, Residgraph], axis=1)
        graphbase.columns = ['Y estimé', 'Résidus']
        sns.scatterplot(x='Y estimé',y='Résidus', data=graphbase, color="g")
        plt.title("Nuage de point entre Résidus et Y estimé")
        if Output==True:
            plt.savefig(Path+"//Nuage de point entre Résidus et Y estimé.png")
        else:
            pass
        plt.show()
        plt.close()        
        
        sns.distplot(Resid,color="g")
        plt.title("Distribution des Résidus")
        if Output==True:
            plt.savefig(Path+"//Distribution des Résidus.png")
        else:
            pass
        plt.show()
        plt.close()        
                
        residu=pd.DataFrame(Resid)
        residu.columns=["Residu"]
        residu.head()
        residu=residu.sort_values(by=["Residu"],ascending=True).reset_index()
        residu["count"]=residu.index+1
        residu.head()
        n_rows=residu.shape[0]
        residu['percentile_area']=(residu['count'])/n_rows
        residu["z_theorical"]=ndtri(residu['percentile_area'])
        residu['z_actual']=(residu["Residu"]-residu["Residu"].mean())/residu["Residu"].std(ddof=0)
        
        plt.scatter(residu.z_actual, residu.z_theorical)
        plt.plot([-2,-1,0,1,2],[-2,-1,0,1,2])
        plt.title("QQ plot : residus vs N(0,1)")
        plt.xlabel('Quantiles théoriques')
        plt.ylabel('valeurs ordonées')
        if Output==True:
            plt.savefig(Path+"//QQ plot.png")
        else:
            pass
        plt.show()
        plt.close()    
        
        
        if Estimation=="GD":
            
            
            MSEgraph=pd.DataFrame(Resultat_algo["evolution_MSE"])
            MSEgraph.columns=["MSE"]
            sns.scatterplot(x=MSEgraph.index, y="MSE" ,data=MSEgraph,color="g")
            plt.title("Evolution MSE")
            plt.xlabel("Nombre d'itération")
            plt.ylabel('MSE (normalisée)')  
            if Output==True:
                plt.savefig(Path+"//Evolution MSE.png")
            else:
                pass
            
            plt.show()
            plt.close()

    
        
    else:
        pass
    
########################## Fin ##########################

    return Resultat
