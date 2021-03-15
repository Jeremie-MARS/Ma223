#Jeremie

import numpy as np
import time as t
import matplotlib.pyplot as plt

A = np.array([[2,5,6],[4,11,9],[-2,-8,7]])
B = np.array([[7],[12],[3]])


def GaussChoixPivotTotal(A,B):
    n,m = np.shape(A)
    B = B.reshape(n,1)
    
    x = np.zeros(n)
    templ = [0,0]
    tempc = [0,0]
    jmax = 0
    for i in range(0, n-1):
        print('A',A)
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
            
            print('Abase',A)
            tempc[:] = A[:,i]
            A[:,i] = A[:,kmax]
            A[:,kmax] = tempc[:]
            templ[:] = A[i,:]
            A[i,:] = A[jmax,:]
            A[jmax,:] = templ[:]
            
            print('Amodif',A)
        for k in range(i+1,n):
            g = A[k,i] / A[i,i]
            A[k, :] = A[k, :] - g * A[i, :]
        print('Acalcul',A)

    #Ax=B
    print('Afinal',A)
    T = np.column_stack((A,B))
    x[n-1] = T[n-1,n] / T[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = T[i,n]
        for j in range(i+1, n):
            x[i]= x[i] - T[i,j] * x[j]
        x[i] =  x[i] / T[i,i]
    return x
print(GaussChoixPivotTotal(A,B))