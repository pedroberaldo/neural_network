#!/usr/bin/env python
# encoding: utf-8

############################################################################
#                                                                          #
# Author: Pedro Beraldo                                                    #   
#                                                                          #
# About: Auxiliary methods for optmization of functions                    #
# -> bissection() : Find the best learning rate use the bissection method  #
# -> grad() : return function value in X and the gradient in x1 and x2     #
# -> hessian(x) : return the hessian in the point x                        #
############################################################################
import numpy as np

def bissection(X,d):
    alfa_l = 0
    alfa_u = np.random.rand()
    xnew = X + alfa_u*d
    
    
    f,g = grad(xnew)
    h = g.T.dot(d)
    while h < 0:
        alfa_u = 2*alfa_u
        Xnew = X + alfa_u*d
        f,g = grad(Xnew)
        h = g.T.dot(d)

    alfa_m = (alfa_l+alfa_u) / 2
    k = np.ceil(np.log((alfa_u-alfa_l) / 1.0e-5))
    nit = 0
    while nit < k and np.abs(h) > 1.0e-5:
        if alfa_m == 0: break
        Xnew = X + alfa_m*d
        f,g = grad(Xnew)
        h=g.T.dot(d)
        if h > 0:
            alfa_u = alfa_m
        else:
            alfa_l = alfa_m
        alfa_m = (alfa_l+alfa_u)/2
        nit +=1
    
    alfa = alfa_m;

    return alfa


def bissection_mlp(X,d,A,B,dJdA,dJdB,N):
    direction = -concatenate(dJdA, dJdB)

    alfa_l = 0
    alfa_u = np.random.rand(1,1)
    

    Aaux = A - alfa_u*dJdA
    Baux = B - alfa_u*dJdB
    dJdAaux,dJdBaux = grad(X,d,Aaux,Baux,N)

    g = concatenate(dJdA, dJdB)
    
    hl = np.dot(g.T, direction)
    
    while hl < 0:
        #print ("bla")
        alfa_u = 2 * alfa_u
        Aaux = A - alfa_u*dJdA
        Baux = B - alfa_u*dJdB
        dJdAaux,dJdBaux = grad(X,d,Aaux,Baux,N)
        
        g = concatenate(dJdAaux, dJdBaux)
        hl = np.dot(g.T, direction)
        #print (hl)

    alfa_m = (alfa_l+alfa_u)/2;
    Aaux = A - alfa_m*dJdA;
    Baux = B - alfa_m*dJdB;
    dJdAaux,dJdBaux = grad(X,d,Aaux,Baux,N)
    #g = np.array([[dJdAaux.flatten()],[dJdBaux.flatten()]])
    g = concatenate(dJdAaux, dJdBaux)
    hl = np.dot(g.T, direction)

    nit = 0;
    nitmax = np.ceil(np.log((alfa_u-alfa_l)/1.0e-5))

    while nit < nitmax and np.abs(hl)>1.0e-5:
        nit = nit+1;
        if hl > 0:
            alfa_u = alfa_m
        else:
            alfa_l = alfa_m

        alfa_m = (alfa_l+alfa_u)/2
        Aaux = A - alfa_m*dJdA
        Baux = B - alfa_m*dJdB
        dJdAaux,dJdBaux = grad(X,d,Aaux,Baux,N)
        
        g = concatenate(dJdAaux, dJdBaux)
        hl = np.dot(g.T, direction)
    alfa = alfa_m;

    return alfa

def bfgs(X,d,A,B,N):
    
    epsi = 10e-5
    
    n = X.shape[0]
    #M = np.eye(N) # M = inv(second grad)
    i = 0
    
    dJdA,dJdB = grad(X,d,A,B,N) # gerar uma matriz unica é um único vetor gradiente
    dJdX = concatenate(dJdA, dJdB)
    M = np.eye(dJdX.shape[0]) # Is this correct? ***********
    

    while np.linalg.norm(dJdX) > epsi:
        direction = -np.dot(M,dJdX)
        
        if i % n == 0:
            direction = -dJdX
            M = np.eye(n)

        #alpha = bissection_mlp(X,direction) # unico alfa
        alpha = bissection_mlp(X,d,A,B,dJdA,dJdB,N)
        #alpha = LineSearch(dJdX,X,direction)
        
        X = X - alpha * direction

        oldGrad = dJdX
        #f, g = op.grad(X)
        dJdA,dJdB = grad(X,d,A,B,N)
        dJdX = concatenate(dJdA, dJdB)

        M += aprox_hessian_bfgs(dJdX, oldGrad, alpha, M)

        i += 1

    return (dir_a, dir_b)
    
def aprox_hessian_bfgs(grad, oldGrad, alpha,M):
    p = alpha * (-grad)
    q = (grad - oldGrad)
    pTdotq = np.dot(p.T,q)
    MdotQ = np.dot(M,q)

    M = np.outer(p,p.T)*(np.dot(p.T,q) + np.dot(q.T, MdotQ)) / pTdotq**2 - (np.outer(MdotQ,p.T)+ np.outer(p, MdotQ)) / pTdotq

    return M

def grad(X,d,A,B,N):
    X_bias = add_bias(X)
    Zin = np.dot(X_bias,A.T)
    Z = 1 / (1 + np.exp(-Zin))
    Z_bias = add_bias(Z)
    Yin = np.dot(Z_bias,B.T)
    Y = Yin # assuming a linear output
    err = Y - d
    
    dJdB = 1 / N * np.dot(err.T,Z_bias)
    dJdZ = np.dot(err,B)
    dJdZ = dJdZ[:,1:] # Removing bias
    dJdA = 1 / N * np.dot((dJdZ * ((1-Z) * Z)).T,X_bias)
    
    return dJdA, dJdB

def concatenate(matA, matB): # N is the size of the column in each matrix
    matA_concat_matB = np.concatenate((matA.flatten(),matB.flatten()))
    return np.reshape(matA_concat_matB, (matA_concat_matB.shape[0],))

def add_bias(X):
    return np.insert(X,0,1,axis=1)


'''
def grad(X):
    f = (X[0]-2)**2 + (X[1]-3)**2
    g = np.array([2*(X[0]-2), 2*(X[1]-3)])
    #f = (1-X[0])**2 + 100 * (X[1] - X[0] ** 2) ** 2
    #g = np.array([2 * (-1 + X[0] + 200 * X[0] ** 3 - 200 * X[0]*X[1]), 200 *(-X[0]**2 + X[1])]) # rosenbrock
    #h=[ 2 0;0 2];
    #f = 0.5*X[0]**2+2.5*X[1]**2
    #g = np.array([X[0], 5 * X[1]])
    #h=[ 1 0;0 5];

    return f,g '''