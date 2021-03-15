#Jérémie JULIEN

import numpy as np
import time as t
import matplotlib.pyplot as plt
#I
#Q1
print('Q1')
'''
A = np.array([[1,1,1,1],[2,4,-3,2],[-1,-1,0,-3],[1,-1,4,9]])
B = np.array([[1],[1],[2],[-8]])
a = np.array([[1,1,1,1,1],[2,4,-3,2,1],[-1,-1,0,-3,2],[1,-1,4,9,-8]])
'''
#====================================================================================================
#==========================================FONCTION==================================================
#====================================================================================================

#============================================GAUSS===================================================
def ReductionGauss(Au):
    n, m = np.shape(Au)
    for i in range(0, n-1):
        if Au[i,i] == 0 :
            Au[i, :] = Au[i+1]
        else :
            for j in range(i+1,n):
                g = Au[j,i] / Au[i,i]
                Au[j, :] = Au[j, :] - g * Au[i, :]
    return Au
#Q2
print('Q2')
def ResolutionSystTriSup(Tu):
    n, m = np.shape(Tu)
    x = np.zeros(n)
    x[n-1] = Tu[n-1,m-1] / Tu[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = Tu[i,m-1]
        for j in range(i+1, n):
            x[i]= x[i] - Tu[i,j] * x[j]
        x[i] =  x[i] / Tu[i,i]   
    return x
#Q3
print('Q3')
def Gauss(A, B):
    n, m = np.shape(A)
    B = B.reshape(n,1)
    R = np.column_stack((A,B))
    return ResolutionSystTriSup(ReductionGauss(R))
'''Gauss(A,B)'''

#II
#=============================================LU=====================================================

def DecompositionLU(A):
    n, n = np.shape(A)
    L = np.eye(n)
    U = np.copy(A)
    for i in range(0, n-1):
        for j in range(i+1,n):
            g = U[j,i] / U[i,i]
            L[j, i] = g
            U[j, :] = U[j, :] - g * U[i, :]
    return L,U

def ResolutionLU(L,U,B):
    n, n = np.shape(L)
    x = np.zeros(n)
    y = []
    #Ly=b
    for i in range(0,n):
        y.append(B[i])
        for k in range(0,i):
            y[i] = y[i] - L[i,k] * y[k]
        y[i] = y[i] / L[i,i]
    #Ux=y
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.dot(U[i,i+1:],x[i+1:])) / U[i,i]
    return x

def LU(A, B):
    n, n = np.shape(A)
    B = B.reshape(n,1)
    L,U = DecompositionLU(A)
    return ResolutionLU(L,U,B)

#===========================================PivotPartiel=============================================

def GaussChoixPivotPartiel(A,B):
    n,n = np.shape(A)
    B = B.reshape(n,1)
    
    x = np.zeros(n)
    temp = [0,0,0,0]
    jmax = 0
    for i in range(0, n-1):
        jmax = i 
        pivotmax= abs(A[jmax,i])
        for j in range(i,n):
            if abs(A[j,i]) > pivotmax : 
                jmax = j
                pivotmax = abs(A[jmax,i])
            
        if pivotmax <= 1e-16:
            print ("La matrice est singulière") 
            return B
        
        temp[:] = A[i,:]
        A[i,:] = A[jmax,:]
        A[jmax,:] = temp[:]

        for k in range(i+1,n):
            g = A[k,i] / A[i,i]
            A[k, :] = A[k, :] - g * A[i, :]

    #Ax=B
    T = np.column_stack((A,B))
    x[n-1] = T[n-1,n] / T[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = T[i,n]
        for j in range(i+1, n):
            x[i]= x[i] - T[i,j] * x[j]
        x[i] =  x[i] / T[i,i]
    return x

#============================================PivotTotal==============================================

def GaussChoixPivotTotal(A,B):
    n,m = np.shape(A)
    B = B.reshape(n,1)
    
    x = np.zeros(n)
    templ = [0,0]
    tempc = [0,0]
    jmax = 0
    kmax = 0
    for i in range(0, n-1):
        jmax = i
        kmax = i
        pivotmax= abs(A[jmax,kmax])

        if i >= 1:
            for k in range(i,n-1):
                for j in range(i,n-1):
                    if abs(A[j,k]) > pivotmax : 
                        jmax = j
                        kmax = k
                        pivotmax = abs(A[jmax,kmax])

            tempc[:] = A[:,i]
            A[:,i] = A[:,kmax]
            A[:,kmax] = tempc[:]
            templ[:] = A[i,:]
            A[i,:] = A[jmax,:]
            A[jmax,:] = templ[:]

        for k in range(i+1,n):
            g = A[k,i] / A[i,i]
            A[k, :] = A[k, :] - g * A[i, :]
    #Ax=B
    T = np.column_stack((A,B))
    x[n-1] = T[n-1,n] / T[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = T[i,n]
        for j in range(i+1, n):
            x[i]= x[i] - T[i,j] * x[j]
        x[i] =  x[i] / T[i,i]
    return x

#Q4
print("Q4")
#====================================================================================================
#===========================================GRAPHS===================================================
#====================================================================================================



#=============================================RANGE==================================================
pas = 100
maxi = 1002
mini = 1


def graphiques():
    #DONNEES
    #temps
    taille=[]
    Y_gauss=[]
    Y_lu=[]
    Y_pp=[]
    Y_pt=[]
    #erreur
    taille_=[]
    e_gauss=[]
    e_lu=[]
    e_pp=[]
    e_pt=[]

    #CALCUL
    for i in range(mini, maxi, pas):

        print(i,"\n")
        A = np.random.rand(i, i)
        B = np.random.rand(1, i)
        C = np.copy(A)
        D = np.copy(A)
        E = np.copy(A)
        

        #Gauss
        t1_gauss = t.time()
        x1 = Gauss(A,B)
        print('Solutions :', x1)
        t2_gauss = t.time()
        y1 = np.linalg.norm(np.dot(A,x1)-np.ravel(B))
        print(y1)
        

        #LU
        t1_lu = t.time()
        x2 = LU(C,B)
        print('Solutions :', x2)
        t2_lu = t.time()
        y2 = np.linalg.norm(np.dot(C,x2)-np.ravel(B))
        print(y2)
        
        
        #PPartiel
        t1_pp = t.time()
        x3 = GaussChoixPivotPartiel(D,B)
        print('Solutions :', x3)
        t2_pp = t.time()
        y3 = np.linalg.norm(np.dot(D,x3)-np.ravel(B))
        print(y3)

        
        #PTotal
        t1_pt = t.time()
        x4 = GaussChoixPivotTotal(E,B)
        print('Solutions :', x4)
        t2_pt = t.time()
        y4 = np.linalg.norm(np.dot(E,x4)-np.ravel(B))
        print(y4)
        
        #enregistrement des résultats
        #temps
        taille.append(i)
        Y_gauss.append(t2_gauss-t1_gauss)
        Y_lu.append(t2_lu-t1_lu)
        Y_pp.append(t2_pp-t1_pp)
        Y_pt.append(t2_pt-t1_pt)
    
        #erreur
        taille_.append(i)
        e_gauss.append(y1)
        e_lu.append(y2)
        e_pp.append(y3)
        e_pt.append(y4)

    #GRAPHS
    #temps
    plt.plot(taille,Y_gauss,'-b',label = 'Gauss')   #Gauss
    plt.plot(taille,Y_lu,'-g',label = 'LU')   #LU
    plt.plot(taille,Y_pp,'-r',label = 'PivotPartiel')   #PPartiel
    plt.plot(taille,Y_pt,'-c',label = 'PivotTotal')   #PTotal
    '''
    plt.semilogy(taille,Y_gauss,'-b',label = 'Gauss')   #Gauss
    plt.semilogy(taille,Y_lu,'-g',label = 'LU')   #LU
    plt.semilogy(taille,Y_pp,'-r',label = 'PivotPartiel')   #PPartiel
    plt.semilogy(taille,Y_pt,'-c',label = 'PivotTotal')   #PTotal
    '''
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Temps d'éxecution - T(s)")
    plt.title("Temps en fonction de la taille n")
    plt.legend()
    plt.show()

    #erreur
    plt.plot(taille_,e_gauss,'-b',label = 'Gauss')   #Gauss
    plt.plot(taille_,e_lu,'-g',label = 'LU')   #LU
    plt.plot(taille_,e_pp,'-r',label = 'PivotPartiel')   #PPartiel
    plt.plot(taille_,e_pt,'-c',label = 'PivotTotal')   #PTotal
    '''
    plt.semilogy(taille_,e_gauss,'-b',label = 'Gauss')   #Gauss
    plt.semilogy(taille_,e_lu,'-g',label = 'LU')   #LU
    plt.semilogy(taille_,e_pp,'-r',label = 'PivotPartiel')   #PPartiel
    plt.semilogy(taille_,e_pt,'-c',label = 'PivotTotal')   #PTotal
    '''
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Erreur - ||AX = B||")
    plt.title("Erreur en fonction de la taille n")
    plt.legend()
    plt.show()

graphiques()


