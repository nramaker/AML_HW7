from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
import operator


def reduce_image(image, segments):
    print("### Reducing {} into {} color segments.".format(image, segments))

    original = Image.open(image)

    # vectorize the pixels into RGB values
    pixels, width, height = get_pixels_from_image(original)
    
    # cluster the vectors
    cluster_assignments, cluster_centers=cluster_pixels(pixels, segments)

    #calculate mean colors per cluster
    mean_colors = calculate_mean_colors(pixels, cluster_assignments, segments)

    #update the pixels with their means
    replaced_pixels = update_pixels(cluster_assignments, mean_colors)

    #show final image
    show_image(replaced_pixels, width, height)

    # original.show()

def show_image(image_vector, width, height):
    redarr = np.asarray(column(image_vector, 0))
    greenarr = np.asarray(column(image_vector,1))
    bluearr = np.asarray(column(image_vector,2))
    rgbArray = np.zeros((height,width,3), 'uint8')
    rgbArray[..., 0] = redarr.reshape((height,-1))
    rgbArray[..., 1] = greenarr.reshape((height,-1))
    rgbArray[..., 2] = bluearr.reshape((height,-1))
    image = Image.fromarray(rgbArray)
    image.show()

def column(matrix, i):
    return [row[i] for row in matrix]

def get_pixels_from_image(image):
    print("### Extracting pixels from Image")
    pixels = image.getdata()
    width, height = image.size
    print("   Image is {} by {}".format(width,height))
    return list(pixels), width, height

def cluster_pixels(points, k):
    print("### Performing EM clustering, k={}".format(k))

    predictions, centers = initialize(points, k)

    #uncomment this line to return straight kmeans clusters and centers
    #return predictions, centers

    print("   Refining with EM algorithm")

    #setup
    max_iter = 100
    min_change = 0.1
    
    #initial values
    w_ij = [[1/k] * len(points)] * k  
    mu = np.asarray(centers)
    pi =calc_initial_weights(predictions, k)

    iteration = 0
    converged = False
    while(iteration < max_iter and converged==False):
        print("EM iteration {}".format(iteration))

        #E step
        w_ij = expectation(np.asarray(points), mu, pi)

        #M step
        old_mu = mu
        mu, pi = maximization(np.asarray(points), w_ij, k)

        #measure change
        change = measure_center_change(old_mu, mu)
        print("Observed change to cluster centers: {}".format(change))
        if change < min_change:
            converged=True
        iteration += 1

    #assign predictions
    print("   Assigning Clusters...")
    for doc in range(0, len(predictions)):
        index, value = max(enumerate(w_ij[doc]), key=operator.itemgetter(1))
        predictions[doc] = index
    return predictions, mu

def expectation(data, mu, pi):
    print("   Estimation...")
    # assign every data point to its most likely cluster
    n = data.shape[0]
    d = data.shape[1]
    k = len(pi)

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
    return W


def maximization(data, W, k, alpha=10):
    print("   Maximization...")
    # new_mu = []
    # new_pi = np.zeros((k,1))

    n = data.shape[0]
    d = data.shape[1]
    pi_new = np.log(np.matmul(W.T,np.ones((n,1))))
    pi_new = np.subtract(pi_new, np.log(n))
    W = np.multiply(alpha,W)
    mu_new = np.matmul(W.T,data)
    mu_new = np.divide(mu_new,np.matmul(W.T,np.ones((n,d))))
    pi_new = np.matmul(W.T,np.ones((n,1)))
    pi_new = np.divide(pi_new,n)
    return mu_new,pi_new 

def measure_center_change(old_mu, new_mu):
    dist = np.linalg.norm(new_mu-old_mu)
    return dist

def calc_initial_weights(cluster_assignments, k):
    print("   Calculating initial weights...")
    weights = np.zeros((k,1))
    N = len(cluster_assignments)
    for i in range(0, k):
        mask = [(pix==i) for pix in cluster_assignments]
        num_pixels = sum(mask)
        weights[i,0]= num_pixels/N
    return weights

def initialize(pixel_vectors, k):
    print("   Initializing Cluster centers with Kmeans")
    X = np.array(pixel_vectors)
    #start with kmeans to get cluster centers
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    predictions = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    return predictions, centers


def calculate_mean_colors(pixel_vectors, cluster_assignments, k):
    print("### Calculating new color assignmenets for {} clusters.".format(k))
    means= []
    for i in range(0,k):
        mask = [(pix==i) for pix in cluster_assignments]

        num_pixels = sum(mask)
        # print("   found {} pixels in cluster {}".format(num_pixels,i))
        pixels_in_cluster= np.asarray([pixel_vectors[j] for j in range(len(pixel_vectors)) if mask[j]])
        mean_color = sum(pixels_in_cluster)/num_pixels
        # print("   mean_color for cluster {} : {}".format(i,mean_color))
        means.append(np.array(mean_color))
    return means

def update_pixels(cluster_assignments, mean_colors):
    print("### Updating pixels...")
    modified_pixels = []
    for i in range(0, len(cluster_assignments)):
        cluster = cluster_assignments[i]
        color = mean_colors[cluster]
        modified_pixels.append(list(color))
    # print("   First Modified pixel {}".format(modified_pixels[0]))
    return np.asarray(modified_pixels)

if __name__ == "__main__":
    print("##### HW7 Image Segmentation #####")

    #reduce_image("./images/RobertMixed03.jpg", 3)
    #reduce_image("./images/RobertMixed03.jpg", 10)
    #reduce_image("./images/RobertMixed03.jpg", 20)
    #reduce_image("./images/RobertMixed03.jpg", 50)
    #reduce_image("./images/smallstrelitzia.jpg", 10)
    #reduce_image("./images/smallstrelitzia.jpg", 20)
    #reduce_image("./images/smallstrelitzia.jpg", 50)
    reduce_image("./images/smallsunset.jpg", 10)
    #reduce_image("./images/smallsunset.jpg", 20)
    # reduce_image("./images/smallsunset.jpg", 50)
