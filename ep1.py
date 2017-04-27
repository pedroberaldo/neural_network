from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import optimizer_aux as op

def MSE_calc(err,N):
    return 1 / N * sum(sum(err**2))

def MLP(X,d,h):
    N, ni = X.shape
    no = d.shape[1]
    
    # Initialize weights
    A = np.random.rand(h,ni+1) / 5
    B = np.random.rand(no,h+1) / 5
    
    # feedforward
    Y = feed_forward(X,A,B)
    # Error
    err = Y - d # result - expetcted
    MSE = MSE_calc(err,N)
    nepmax = 10000
    nepochs = 0
    alpha = 1
    vet_mse = np.array(MSE)
    #vet_mse = np.append(vet_mse,MSE)
    
    while nepochs < nepmax and (MSE > 1.0e-5):
        nepochs += 1
        dJdA, dJdB = op.grad(X,d,A,B,N)
        #print (nepochs)
        #search_dir_A,search_dir_B = op.bfgs(X,d,A,B,N) # using BFGS
        
        alpha = op.bissection_mlp(X,d,A,B,dJdA,dJdB,N)
        #alpha = 0.9
        #print (op.bfgs(X,d,A,B,N))
        #A = A - alpha * search_dir_A
        A = A - alpha * dJdA
        #B = B - alpha * search_dir_B
        B = B - alpha * dJdB
        Y = feed_forward(X,A,B)
        err = Y - d
        MSE = MSE_calc(err,N)
        vet_mse = np.append(vet_mse,MSE)
    #print (vet_mse)
    #plt.plot(range(nepochs+1),vet_mse)
    Y = feed_forward(X,A,B)
    print (Y)
    print ("Epochs:", nepochs)

def feed_forward(X,A,B):
    X_bias = op.add_bias(X)
    Zin = np.dot(X_bias,A.T)
    Z = 1 / (1 + np.exp(-Zin))
    Z_bias = op.add_bias(Z)
    Yin = np.dot(Z_bias,B.T)
    #Y = 1 / (1 + np.exp(-Yin))
    Y = Yin
    return Y

X = np.array([[0,0],[0,1],[1,0],[1,1]])

#d = np.array([[0],[0],[0],[1]])
d = np.array([[0],[1],[1],[0]])

print (MLP(X,d,3))