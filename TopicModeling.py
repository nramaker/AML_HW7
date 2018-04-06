import pandas as pd
import numpy as np
from random import *
from scipy.stats import multinomial

k = 30

def load_vocab():
    file = "./NIPS/vocab.nips.txt"
    print("   Loading vocab from file: {}".format(file))
    data = pd.read_csv(file)
    return data

def load_data():
    file = "./NIPS/docword.nips.txt"
    print("   Loading data from file: {}".format(file))
    head = pd.read_csv(file, header=None, nrows=3).as_matrix()
    d = head[0][0]
    n = head[1][0]
    t = head[2][0]
    data = pd.read_csv(file, header=None, skiprows=3, sep=' ').as_matrix()
    return data, d, n, t

def build_word_counts(data, d, n, t):
    print("   Building word_count vectors.")
    word_counts = np.zeros((d, n))

    for i in range (0, t):
        row = data[i]
        docID = row[0]
        wordId = row[1]
        count = row[2]
        word_counts[docID-1][wordId-1] = word_counts[docID-1][wordId-1] + count
    return word_counts

# def expectation(points, cluster_centers, cluster_weights):
#     #TODO: implement estimation
#     return []

# def maximization(points, w_ij, t):
#     #TODO implement maximization
#     return np.asarray([]), np.asarray([])

def show_histogram(weights):
    #TODO display histogram
    pass

def show_table(word_probs):
    #TODO display table
    pass

# def measure_change(old_mus, new_mus):
#     dist = np.linalg.norm(new_mus-old_mus)
#     #should we take the average change?
#     return dist

if __name__ == "__main__":
    print("##### HW7 Topic Modeling #####")

    #load data from files
    vocab = load_vocab()
    data, d, n, t = load_data()

    #organize the data into word counts per document
    word_counts = build_word_counts(data, d, n, t)

    rows = word_counts.shape[0]  # word indexes per document
    cols = word_counts.shape[1]  # documents indexes
    # print("   rows {} : columns {}".format(rows, cols))

    #initialize variables
    #TODO: better initialization
    w_ij  = np.zeros((k,rows))
    
    #these are the random initial cluster centers
    np.random.seed(1)
    mus = np.asarray([word_counts[np.random.randint(0, cols-1)]]*k) #randomly choose k documents as centroids

    #these are the inital cluster weights
    pis = [[1/k]] * k  #equally weighted clusters to start


    #perform clustering
    max_iters = 1000
    min_tau = 0.001

    iteration = 0
    converged = False
    while( iteration < max_iters and converged==False):
        print("### Performing E/M algorithm: iteration {}".format(iteration))

        #this is the old code, delete?
        # #e step
        # w_ij = expectation(word_counts, mus, pis)

        # #m step
        # old_mus = mus
        # mus, pis = maximization(word_counts, w_ij, n)

        # E - step
        print("   E-step")
        priors_probs = np.zeros((k,rows))

        print("     Calculating Priors.")
        for i in range(0,k-1):
            for j in range(0,rows-1):
                priors_probs[i,j] = multinomial.pmf(word_counts[j,], sum(word_counts[j,]), mus[i,])
    
        w_ij_new = np.zeros((k,rows))
        sumcp = np.dot(np.asarray(pis).T, priors_probs)

        sumcp = sumcp + 0.0001 #avoid divide by zero ?
        
        #new assignments
        print("     Calculating New Assignments.")
        for i in range(0,k-1):
            w_ij_new[i,] = (np.asarray(pis[i]).T * priors_probs[i,])/sumcp
        
        # M - step
        print("   M-Step")
        old_mus = mus
        for i in range(0, k-1):
            pis[i] = sum(w_ij[i,])/rows
            mus[i,] = (np.dot(w_ij[i,], word_counts))/sum(word_counts[i,])
            mus[i,] = mus[i,]/n #normalize 

        print("  Measuring Tau")

        # sum1 = sum(w_ij[0])
        # print("sum1 {}".format(sum1))
        # sum2 = sum(w_ij_new[0])
        # print("sum2 {}".format(sum2))

        #TODO: there is a defect here, tau evaluates to NaN
        tau = np.linalg.norm(w_ij-w_ij_new)
        print("   Detected probabilities change of {}".format(tau))
        if(tau < min_tau):
            converged=True
            print("   EM algorithm has converged, stopping.")
        w_ij=w_ij_new
        iteration+=1
            
    
        #c is the cluster weights :  pis
        #t is the cluster centers :  mus
        #a is the cluster assignements: w_ij
        #p is prior cluster probs : priors_probs 
        #H is our data : word_counts

        # E - step
#   while(1){
#     p = matrix(0,k,rows)
#     for(i in 1:k){
#       for(j in 1:rows){
#         p[i,j] = dmultinom(H[j,],sum(H[j,]),t[i,])
#       }
#     }
#     anew = matrix(0,k,rows)
#     sumcp = c %*% p
#     #print(dim(anew))
#     #print(dim(sumcp))
#     #print(dim(p))
#     sumcp = sumcp + 0.0001
#     for(i in 1:k){
#       anew[i,] = (c[i] * p[i,])/sumcp
#     }
#     #check anew and old !!
#     print(iter)
#     #print(anew)
#     print(norm((a-anew),"O"))
#     if(norm((a-anew),"O") < tau)
#       break
#     a = anew
    
#     # M - step
#     for(i in 1:k){
#       c[i] = sum(a[i,])/rows
#       t[i,] = (a[i,] %*% as.matrix(H))/sum(a[i,])
#       t[i,] = t[i,]/12419
#     }
    
    show_histogram(pis)
    show_table(w_ij)

        