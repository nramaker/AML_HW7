import pandas as pd
import numpy as np
from random import *
#from scipy.stats import multinomial
import matplotlib.pyplot as plt

#k = 30
k=30

def load_vocab():
    file = "./NIPS/vocab.nips.txt"
    print("   Loading vocab from file: {}".format(file))
    data = pd.read_csv(file)
    return data

def load_data():
    nips = pd.read_csv('./NIPS/nips.txt',delimiter=' ')
    nips = nips.pivot(columns='wordid', index='docid', values='count')
    nips = nips.fillna(value=0)
    nips = np.add(nips,0.1)

    badcols = list(np.flatnonzero(np.sum(nips.values, 1)<1000))
    allcols = list(range(1,1491))
    cols = [x for x in allcols if x not in badcols]
    nips = nips.iloc[:,cols]
    nips.columns = list(range(0,nips.shape[1]))
    #nips = nips.add(1)
    nips_data = nips.values

    # print(str(nips.shape[0]) + " unique documents")
    # print(str(nips.shape[1]) + " unique words")
    nips.head()
    return nips

def show_histogram(weights):
    plt.figure(1)
    
    indexes = list(range(0,weights.shape[1]))
    print("indexes : {}".format(indexes))
    print("weights : {}".format(weights))
    plt.bar(indexes, weights[0], label=indexes)
     
    plt.xscale('linear')
    plt.yscale('linear')
        
    plt.title('Likelihood of Each Topic Being Chosen')
    # plt.legend()
    plt.grid(True)
    plt.show()
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

def e_step(data,mu,k,pi):
    
    # data is a dxn numpy array of word counts
    # mu is a kxd numpy array of cluster centroids
    # k is the cluster number
    # pi is a kx1 array of cluster membership probabilities
    
    n = data.shape[0]
    d = data.shape[1]

    h1 = np.multiply(data,data)
    h1 = np.matmul(h1,np.ones((d,k)))
    h2 = np.matmul(np.ones((n,d)), np.multiply(mu,mu).T)
    h3 = np.matmul(data,mu.T)
    H = -0.5*(np.subtract(np.add(h1,h2),2*h3))

    P = np.matmul(np.ones((n,1)),pi.T)
    E = np.multiply(np.exp(H),P)
    F = np.matmul(E,np.ones((k,k)))
    F = np.add(F,0.01)
    W = np.divide(E,F)

    return(W)

def m_step_log(W,data,pis,alpha):
    n = data.shape[0]
    d = data.shape[1]
    pi_new = np.log(np.matmul(W.T,np.ones((n,1))))
    pi_new = np.subtract(pi_new, np.log(n))
    W = np.multiply(alpha,W)
    mu_new = np.matmul(W.T,data)
    mu_new = np.divide(mu_new,np.matmul(W.T,np.ones((n,d))))
    pi_new = np.matmul(W.T,np.ones((n,1)))
    pi_new = np.divide(pi_new,n)
    return(mu_new,pi_new)
    
def evalulate(mu_old,mu_new,threshold):
    diff = np.subtract(mu_new,mu_old)
    diff = np.linalg.norm(diff)
    print("   Detected change of {}".format(diff))
    if diff <= threshold:
        return(1)
    else:
        return(0)

if __name__ == "__main__":
    print("##### HW7 Topic Modeling #####")

    #load data from files
    vocab = load_vocab()
    nips_data = load_data() 

    #initialize variables
    d = nips_data.shape[1]
    n = nips_data.shape[0]
    topic_centers = np.random.random(k)
    mu = topic_centers.reshape((k,1))
    mus = np.random.random(k*d)
    mus = mus.reshape((k,d))
    pi = np.ones((k,1))*1/k


    #perform clustering
    max_iters = 50
    threshold = 0.001

    iteration = 0
    converged = False
    while( iteration < max_iters and converged==False):
        print("### Performing E/M algorithm: iteration {}".format(iteration))

        # E - step
        print("   E-step")
        W = e_step(data=nips_data,k=k,mu=mus,pi=pi)
        
        # M - step
        print("   M-Step")
        mu_old = mus
        mus, pi = m_step_log(data=nips_data,W=W, pis=pi, alpha=10)

        #Measure
        converged = evalulate(mu_new=mus,mu_old=mu_old, threshold=threshold)
        
        iteration+=1
    
    show_histogram(np.asarray(pi).T)
    show_table(W)

        