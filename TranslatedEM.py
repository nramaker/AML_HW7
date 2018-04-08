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
def m_step(W,data):
    # W is a nxk array of document likelihoods (given cluster)
    # data is a dxn array of word counts
    D = np.matmul(W.T,data)
    G = np.matmul(W.T,np.ones((n,d)))
    mu_new = np.divide(D,G)
    return(mu_new)
    
def evalulate(mu_old,mu_new,threshold):
    diff = np.subtract(mu_new,mu_old)
    if diff <= threshold:
        return(1)
    else:
        return(0)
    
###########
## Initialize
k = 3
d = nips_data.shape[1]
n = nips_data.shape[0]
topic_centers = np.random.random(topic_count)
mu = topic_centers.reshape((topic_count,1))
mus = np.random.random(k*d)
mus = mus.reshape((k,d))
pi = np.ones((k,1))*1/k
W = e_step(data=nips_data,k=3,mu=mus,pi=pi)
mu_new = m_step(data=nips_data,W=W)
evalulate(mu_new=mu_new,mu_old=mus, threshold=0.001)