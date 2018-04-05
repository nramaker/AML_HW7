k = 30

def load_vocab():
    #TODO: implement load_vocab
    return []

def load_data():
    #TODO: implement load_data
    return [], 0,0,0

def expectation(points, cluster_centers, cluster_weights):
    #TODO: implement estimation
    return []

def maximization(points, w_ij, n):
    #TODO implement maximization
    return [], []

def show_histogram(weights):
    #TODO display histogram
    pass

def show_table(word_probs):
    #TODO display table
    pass

def measure_change(old_mus, new_mus):
    #TODO measure change function
    return 0.0

if __name__ == "__main__":
    print("##### HW7 Topic Modeling #####")

    #load data from files
    vocab = load_vocab()
    word_counts, d, n, t = load_data()

    #initialize variables
    #TODO: better initialization
    w_ij  = []
    mus = []
    pis = []

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
        if(change < min_change):
            converged=True
            print("   EM algorithm has converged, stopping.")
        iteration+=1
    
    show_histogram(pis)
    show_table(w_ij)

        