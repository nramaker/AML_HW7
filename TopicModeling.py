import pandas as pd
import numpy as np
from random import *
from scipy.stats import multinomial

#TODO: set this back to 30. Setting k=2 so that it takes less time while we iterate
#k = 30
k=2

def load_vocab():
    file = "./NIPS/vocab.nips.txt"
    print("   Loading vocab from file: {}".format(file))
    data = pd.read_csv(file)
    return data

def load_data():
    file = "./NIPS/docword.nips.txt"
    data_file = "./NIPS/nips.txt"
    print("   Loading data from file: {}".format(file))
    head = pd.read_csv(file, header=None, nrows=3).as_matrix()
    d = head[0][0]
    n = head[1][0]
    t = head[2][0]
    nips = pd.read_csv(data_file,delimiter=' ')
    nips = nips.pivot(columns='wordid', index='docid', values='count')
    # print(nips)
    #nips.fillna(0.001,inplace=True)
    nips.fillna(.001, inplace=True)
    # print(nips)
    return nips.as_matrix(), d, n, t

def show_histogram(weights):
    #TODO display histogram
    pass

def show_table(word_probs):
    #TODO display table
    pass

def doc_word_freq(docid,wordid,data):
    return(data[wordid][docid])

def topic_word_freq(topicid,wordid,topics):
    return(topics[topicid,wordid]) 

def doc_likelihood(docid, topic, data):
    # P(d | weights)
    words = list(range(0,data.shape[0]))
    adw = [np.round(doc_word_freq(docid,wordid,data)) for wordid in words]
    likelihood = 1
    for word in range(data.shape[0]):
        likelihood *= (topic)**(adw[word-1])
    return(likelihood)

if __name__ == "__main__":
    print("##### HW7 Topic Modeling #####")

    #load data from files
    vocab = load_vocab()
    word_counts, d, n, t = load_data()

    rows = word_counts.shape[0]  # word indexes per document
    cols = word_counts.shape[1]  # documents indexes

    #initialize variables
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

        #pis are the cluster weights 
        #mus are the cluster centers
        #wi_j is the cluster assignements
        #prior_probs is prior cluster probabilities 
        # word_counts is our data

        # E - step
        print("   E-step")
        priors_probs = np.zeros((k,rows))

        print("     Calculating Priors.")
        for i in range(0,k):  # i is the topic
            print("     Priors for cluster={}".format(i))
            for j in range(0,rows):  #j is the document
                priors_probs[i,j] = doc_likelihood(j, np.asarray(pis)[i], word_counts)
    
        w_ij_new = np.zeros((k,rows))
        sumcp = np.dot(np.asarray(pis).T, priors_probs)

        sumcp = sumcp + 0.0001 #avoid divide by zero ?
        
        #new assignments
        print("     Calculating New Assignments.")
        for i in range(0,k):
            w_ij_new[i,] = (np.asarray(pis[i]).T * priors_probs[i,])/sumcp
        
        # M - step
        print("   M-Step")
        old_mus = mus
        for i in range(0, k):
            pis[i] = sum(w_ij_new[i,])/rows
            mus[i,] = (np.dot(w_ij_new[i,], word_counts))/sum(word_counts[i,])
            mus[i,] = mus[i,]/n #normalize 

        print("  Measuring Tau")

        tau = np.linalg.norm(w_ij-w_ij_new)
        print("   Detected probabilities change of {}".format(tau))
        if(tau < min_tau):
            converged=True
            print("   EM algorithm has converged, stopping.")
        w_ij=w_ij_new
        iteration+=1
    
    show_histogram(pis)
    show_table(w_ij)

        