from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

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

    #comment this out
    return predictions, centers

    print("   Refining with EM algorithm")

    #setup
    max_iter = 10
    min_change = 0.01
    
    #initial values
    #w_ij = matrix(1/k, len(points), k)
    w_ij = [[1/k] * len(points)] * k
    mu = centers
    pi = calc_initial_weights(predictions, k)

    iteration = 0
    converged = False
    while(iteration < max_iter and converged==False):
        print("EM iteration {}".format(iteration))

        #E step
        w_ij = expectation(points, mu, pi)

        #M step
        old_mu = mu
        mu, pi = maximization(points, w_ij)

        #measure change
        change = measure_center_change(old_mu, mu)
        print("Observed change to cluster centers: {}".format(change))
        if change < min_change:
            converged=True
        iteration += 1

    #assign predictions
    print("   Assigning Clusters...")
    return predictions, centers

def expectation(points, cluster_centers, cluster_weights):
    print("   Estimation...")
    # assign every data point to its most likely cluster
    
    return dataFrame

def maximization(points, w_ij, n):
    print("   Maximization...")
    new_mu = []
    new_pi = []

    for j in range(0, 2):
        print("w_ij[{}] = {}".format(j, column(w_ij,j)))
        sum_wi = sum(column(w_ij,j))
        print("sum_wi {}".format(sum_wi))

        #calculate new centers
        mu = sum(column(points,j)*sum_wi)/sum_wi
        print("new mu[{}] = {}".format(j, mu))
        new_mu.append(mu)

        #calculate new weights
        pi = sum_wi/n 
        print("new pi[{}] = {}".format(j, pi))
        new_pi.append(pi)
    
    return new_mu, new_pi

def logsumexp(X):
    x_max = X.max(1)
    return x_max + np.log(np.exp(X - x_max[:, None]).sum(1))

def measure_center_change(old_mu, new_mu):
    dist = numpy.linalg.norm(new_mu-old_mu)
    return dist

def calc_initial_weights(cluster_assignments, k):
    print("   Calculating initial weights...")
    weights =[]
    N = len(cluster_assignments)
    for i in range(0, k):
        mask = [(pix==i) for pix in cluster_assignments]
        num_pixels = sum(mask)
        weights.append(num_pixels/N)
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

    reduce_image("./images/RobertMixed03.jpg", 10)
    #reduce_image("./images/RobertMixed03.jpg", 20)
    #reduce_image("./images/RobertMixed03.jpg", 50)
    # reduce_image("./images/smallstrelitzia.jpg", 10)
    # reduce_image("./images/smallstrelitzia.jpg", 20)
    # reduce_image("./images/smallstrelitzia.jpg", 50)
    # reduce_image("./images/smallsunset.jpg", 10)
    # reduce_image("./images/smallsunset.jpg", 20)
    # reduce_image("./images/smallsunset.jpg", 50)
