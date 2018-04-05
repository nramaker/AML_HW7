import pandas as pd
import numpy as np

k = 30

def load_vocab():
    file = "./NIPS/vocab.nips.txt"
    print("   Loading vocab from file: {}".format(file))
    data = pd.read_csv(file)
    return data

def load_data():
    file = "./NIPS/docword.nips.txt"
    print("   Loading vocab from file: {}".format(file))
    head = pd.read_csv(file, header=None, nrows=3).as_matrix()
    d = head[0][0]
    n = head[1][0]
    t = head[2][0]
    data = pd.read_csv(file, header=None, skiprows=3, sep=' ').as_matrix()
    return data, d, n, t

def build_word_counts(data, d, t):
    print("   Building word_count vectors.")
    word_counts = np.zeros((d, t))

    for i in range (0, t):
        row = data[i]
        docID = row[0]
        wordId = row[1]
        count = row[2]
        word_counts[docID-1][wordId-1] = word_counts[docID-1][wordId-1] + count
    return word_counts

def expectation(points, cluster_centers, cluster_weights):
    #TODO: implement estimation
    return []

def maximization(points, w_ij, t):
    #TODO implement maximization
    return np.asarray([]), np.asarray([])

def show_histogram(weights):
    #TODO display histogram
    pass

def show_table(word_probs):
    #TODO display table
    pass

def measure_change(old_mus, new_mus):
    dist = np.linalg.norm(new_mus-old_mus)
    #should we take the average change?
    return dist

if __name__ == "__main__":
    print("##### HW7 Topic Modeling #####")

    #load data from files
    vocab = load_vocab()
    data, d, n, t = load_data()

    #organize the data into word counts per document
    word_counts = build_word_counts(data, d, t)

    #initialize variables
    #TODO: better initialization
    w_ij  = []
    mus = []
    pis = [[1/k]] * k

    #perform clustering
    max_iters = 1000
    min_change = 0.001

    iteration = 0
    converged = False
    while( iteration < max_iters and converged==False):
        print("### Performing E/M algorithm: iteration {}".format(iteration))

        #e step
        w_ij = expectation(word_counts, mus, pis)

        #m step
        old_mus = mus
        mus, pis = maximization(word_counts, w_ij, n)

        change = measure_change(old_mus, mus)
        print("   Detected cluster center change of {}".format(change))
        if(change < min_change):
            converged=True
            print("   EM algorithm has converged, stopping.")
        iteration+=1
    
    show_histogram(pis)
    show_table(w_ij)

        