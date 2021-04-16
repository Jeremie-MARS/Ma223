#Jérémie JULIEN

from math import *
import numpy as np
import time as t
import matplotlib.pyplot as plt
'''
A = np.array([[4,-2,-4],[-2,10,5],[-4,5,6]])
B = np.array([[(6)],[-9],[-7]])
'''
#====================================================================================================
#==========================================FONCTION==================================================
#====================================================================================================

#==========================================CHOLESKY==================================================

def Cholesky(A):
    n,n = np.shape(A)
    L = np.zeros((n,n))
    for i in range(0,n):
        S=0
        for j in range(0,i):
            S += L[i,j]**2
        L[i,i] = sqrt(A[i,i]-S)

        for k in range(i+1,n):
            S=0
            for j in range(0,i):
                S += L[k,j]*L[i,j]
            L[k,i] = (A[k,i] - S) / L[i,i]
        
    return L
'''print(Cholesky(A))'''

def ResolCholesky(A,B):
    n, n = np.shape(A)
    B = B.reshape(n,1)
    L = Cholesky(A)
    L_T = np.transpose(L)
    x = np.zeros(n)
    y = []
    #Ly=b
    for i in range(0,n):
        y.append(B[i])
        for k in range(0,i):
            y[i] = y[i] - L[i,k] * y[k]
        y[i] = y[i] / L[i,i]
    #LTx=y
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.dot(L_T[i,i+1:],x[i+1:])) / L_T[i,i]
    return x
'''print(ResolCholesky(A,B))'''

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

#====================================================================================================
#===========================================GRAPHS===================================================
#====================================================================================================


#=============================================RANGE==================================================
pas = 100
maxi = 1002
mini = 1

def MatricePositive(n):
    for i in range(n):
        A = np.random.rand(n,n)
        Atrans = np.transpose(A)
        M = np.dot(A,Atrans)
        print(M)
        return M


def graphiques():
    #DONNEES
    #temps
    taille=[]
    Y_cho=[]
    Y_lu=[]
    Y_li=[]
    
    #erreur
    taille_=[]
    e_cho=[]
    e_lu=[]
    e_li=[]

    #CALCUL
    for i in range(mini, maxi, pas):

        print(i,"\n")
        A = MatricePositive(i)
        B = np.random.rand(1,i)
        B_ = B.reshape(i,1)
        C = np.copy(A)
        D = np.copy(A)

        #Cholesky
        t1_cho = t.time()
        x1 = ResolCholesky(A,B)
        print('Solutions :', x1)
        t2_cho = t.time()
        y1 = np.linalg.norm(np.dot(A,x1)-np.ravel(B))
        print(y1)
        

        #LU
        t1_lu = t.time()
        x2 = LU(C,B)
        print('Solutions :', x2)
        t2_lu = t.time()
        y2 = np.linalg.norm(np.dot(C,x2)-np.ravel(B))
        print(y2)


        #Linalg.solve
        t1_li = t.time()
        x3 = np.linalg.solve(D,B_)
        x3 = x3.reshape(1,i)
        print('Solutions :', x3)
        t2_li = t.time()
        y3 = np.linalg.norm(np.dot(D,np.ravel(x3))-np.ravel(B))
        print(y3)
        
        
        #enregistrement des résultats
        #temps
        taille.append(i)
        Y_cho.append(t2_cho-t1_cho)
        Y_lu.append(t2_lu-t1_lu)
        Y_li.append(t2_li-t1_li)

        #erreur
        taille_.append(i)
        e_cho.append(y1)
        e_lu.append(y2)
        e_li.append(y3)

    #GRAPHS
    #temps
    plt.plot(taille,Y_cho,'-b',label = 'Cholesky')   #Cholesky
    plt.plot(taille,Y_lu,'-g',label = 'LU')   #LU
    plt.plot(taille,Y_li,'-r',label = 'Lg.solve')   #Linalg.solve
    '''
    plt.semilogy(taille,Y_cho,'-b',label = 'Cholesky')   #Cholesky
    plt.semilogy(taille,Y_lu,'-g',label = 'LU')   #LU
    plt.semilogy(taille,Y_li,'-r',label = 'Lg.solve')   #Linalg.solve
    '''
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Temps d'éxecution - T(s)")
    plt.title("Temps en fonction de la taille n")
    plt.legend()
    plt.show()

    #erreur
    plt.plot(taille_,e_cho,'-b',label = 'Cholesky')   #Cholesky
    plt.plot(taille_,e_lu,'-g',label = 'LU')   #LU
    plt.plot(taille,e_li,'-r',label = 'Lg.solve')   #Linalg.solve
    '''
    plt.semilogy(taille_,e_cho,'-b',label = 'Cholesky')   #Cholesky
    plt.semilogy(taille_,e_lu,'-g',label = 'LU')   #LU
    plt.semilogy(taille,e_li,'-r',label = 'Lg.solve')   #Linalg.solve
    '''
    plt.xlabel("Taille de la matrice - n")
    plt.ylabel("Erreur - ||AX = B||")
    plt.title("Erreur en fonction de la taille n")
    plt.legend()
    plt.show()

graphiques()
