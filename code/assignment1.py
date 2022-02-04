import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.datasets as dt

# run the algorithm for 15 times but only plot for k = 4 and 7
k_of_results_to_be_ploted = [4, 7]


def generateSimpleDataset(cluster_count, size, seed):
    """ Generates and plots a dataset with 'cluster_count' number of clusters. Size of the dataset is determined by 'size' argument."""
    # for rand_state in [6, 7, 9, 10, 17]:
    dataset, label = dt.make_blobs(n_samples=size, n_features=2, centers=cluster_count, cluster_std=1, random_state=seed)
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.title('dataset')
    plt.show()
    return dataset

def generateNonconvexDataset(size):
    """ Generates and plots a dataset like two opposite directioned moons intertwined. Size of the dataset is determined by 'size' argument."""
    dataset, label = dt.make_moons(n_samples=size, noise=0.07)
    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.title('dataset')
    plt.show()
    return dataset

def euclideanDistSquared(point1, point2):
    """returns the euclidean distance between two points regardless of dimensionalty"""
    
    return sum([(dim_of_first - dim_of_second)**2 for dim_of_first, dim_of_second in zip(point1, point2)])

def kMeansInitialize(dataset, k, seed):
    """ 
    First generates 'k' number of random centroids. Then labels dataset. 
    Then calculates new positions of centroids by taking the average of 
    the data points belong to them. 
    If a centroid does not attract any data point, function returns -1 and called again.
    Finally plots the initial positions of the centroids and the labeled datasets.
    """
    np.random.seed(seed)

    # initialize centroids randomly
    centroids = np.random.rand(k, 2)  # rand creates numbers between 0 and 1
    # map [0,1] to [min of datapoints, max of datapoints] for x axis
    centroids[:, 0] = centroids[:, 0] * \
        (max(dataset[:, 0])-min(dataset[:, 0])) + min(dataset[:, 0])
    # map [0,1] to [min of datapoints, max of datapoints] for y axis
    centroids[:, 1] = centroids[:, 1] * \
        (max(dataset[:, 1])-min(dataset[:, 1])) + min(dataset[:, 1])
    centroid_labels = np.array([i for i in range(k)])

    # label datasets according to the closest centroid
    dataset_labels = np.zeros(len(dataset), dtype=int)
    point_idx = 0
    for point in dataset:
        centroid_idx = 0
        min_distance = math.inf
        for cluster_center in centroids:
            if(euclideanDistSquared(cluster_center, point) < min_distance):
                min_distance = euclideanDistSquared(cluster_center, point)
                dataset_labels[point_idx] = centroid_idx
            centroid_idx += 1
        point_idx += 1

    # if a centroid did not attract any data point return -1 so that caller function can call this function again.
    for i in range(k):
        if (dataset_labels == i).sum() == 0:
            return -1

    # update centroids by taking the average of datapoints belong to them
    new_centroid_sums = [0]*k
    point_idx = 0
    for point in dataset:
        new_centroid_sums[dataset_labels[point_idx]] += point
        point_idx += 1
    centroids = np.array([new_centroid_sums[i] / (dataset_labels == i).sum()
                         for i in range(k)], dtype=float)

    ## plot first versions of the labeled dataset and centroids only for given k's
    if len(centroids) in k_of_results_to_be_ploted:
        plt.figure()
        plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset_labels, vmin=min(
            dataset_labels), vmax=max(dataset_labels), cmap='tab20')
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c=centroid_labels, vmin=0, vmax=k-1, cmap='tab20b')
        for x, y in zip(range(len(centroids)), centroids):
            plt.text(y[0], y[1], x+1)

        plt.title('Initialized. Step: 0')
        plt.show()
    return centroids, centroid_labels, dataset_labels

def plotTwoDimensionalWithOptionalColors(title, data1, data2, data1_for_color, data2_for_color):
    """
    This function plots two different dataset to one figure. 
    First parameter is the name of the plot.
    Second parameter is "dataset", third parameter is "centroids".
    fourth and fifth parameters are for coloring.
    """
    plt.figure()
    plt.scatter(data1[:, 0], data1[:, 1], c=data1_for_color, vmin=min(
        data1_for_color), vmax=max(data1_for_color), cmap='tab20')
    plt.scatter(data2[:, 0], data2[:, 1],
                c=data2_for_color, vmin=0, vmax=len(data2_for_color)-1, cmap='tab20b')
    for x, y in zip(range(len(data2)), data2):
        plt.text(y[0], y[1], x+1)

    plt.title(title)
    plt.show()

def kMeans(dataset, k):
    """
    This is the main function of the k-means algorithm.
    Takes dataset and "K" value as parameters.
    Returns the objective function and silhouette score.

    It first tries to initialize centroids. Since initialization function called with random seeds, 
    output may differ in each run. After successfully creation of centroids, initializes other variables.
    Then begins the algorithm:
      - For each data point, finds closest centroid and labels it.
      - Calculates objective function by summing up the multiplication of sqaured eaucludian distances between datapoints and centroid by 2.
      - Updates the centroid by taking the average of datapoints belong to it.
      - Plots the labeled dataset and new centroid locations.
    Algorithm stops if Euclidean distance between two consecutive cluster centers (summed over all clusters) is less than 0.00001
    After the algorithm stops, it plots final result and objective function-iteration curve.
    """
    initialized_val = kMeansInitialize( dataset, k, seed=10) ## first try with seed = 10, if fails try random seeds
    # initialize clusters randomly until each of them has at least one data point
    while initialized_val == -1:
        initialized_val = kMeansInitialize(dataset, k, seed=np.random.randint(1000000))

    centroids, centroid_labels, dataset_labels = initialized_val
    prev_centroids = np.zeros([len(centroids), 2])
    centroid_changes = []
    # sum of within class variation of clusters.
    objective_function_at_each_step = []
    # within class variation is sum of squared distances between points to cluster center(centroid)

    step_count = 0
    while True:
        # label data points
        point_idx = 0
        for point in dataset:  # for each point
            label = -1  # label is centroid_id
            dist = math.inf  # dist is distance to centroids
            centroid_idx = 0
            for centroid in centroids:  # for each centroid
                # calculate eacludian distance
                temp = math.sqrt(euclideanDistSquared(centroid, point))
                if temp < dist:  # find centroid with minimum distance
                    dist = temp
                    label = centroid_idx
                centroid_idx += 1
            dataset_labels[point_idx] = label  # assign centroid label to point
            # find farthest point to centroid
            point_idx += 1

        ### calculate objective function
        # instead of summing up all the combination of distances between datapoints,
        # multiply (sum of distances between nodes and centroid) with 2 since they are same
        current_obj_sum = 0
        for point, label in zip(dataset, dataset_labels):
            current_obj_sum += 2 * \
                euclideanDistSquared(centroids[label], point)
        objective_function_at_each_step.append(current_obj_sum)
        
        #### update centroids
        # find sums of datapoints belong to centroids
        new_centroid_sums = [0 for i in range(k)]
        point_idx = 0
        for point in dataset:
            new_centroid_sums[dataset_labels[point_idx]] += point
            point_idx += 1
        
        """ # no need to plot this
        # plot datapoints with updated labels for the first three iterations
        if step_count < 3:
            plotTwoDimensionalWithOptionalColors('Datapoints labeled. Step: '+str(
                step_count+1), dataset, centroids, dataset_labels, centroid_labels)
        """

        # find average of datapoints belong to centroids, if no data point belong to a centroid, leave it unchanged
        # but since we made sure there is at least a point while initializating, this probability is zero.
        prev_centroids = centroids
        centroids = np.array([new_centroid_sums[i] / (dataset_labels == i).sum() if (
            dataset_labels == i).sum() != 0 else centroids[i] for i in range(k)])

        ### plot clusters with updated positions for the first three iterations
        if step_count < 3 and len(centroids) in k_of_results_to_be_ploted:
            plotTwoDimensionalWithOptionalColors('Centroids updated. Step: '+str(
                step_count+1), dataset, centroids, dataset_labels, centroid_labels)
        
        # break condition: sum of the changes in centroid positions is less than a threshold value. 0.00001 in this case.
        centroid_changes.append(
            sum([euclideanDistSquared(p, c) for p, c in zip(prev_centroids, centroids)]))
        if centroid_changes[-1] < 1e-5:
            break
        step_count += 1

    # Plot final result.
    if len(centroids) in k_of_results_to_be_ploted:
        # only plot for k = 4,7
        plotTwoDimensionalWithOptionalColors('Centroids updated. Final step: '+str(
            step_count+1), dataset, centroids, dataset_labels, centroid_labels)
        # Plot objective function with respect to iterations.
        plt.figure()
        plt.scatter(range(1,len(objective_function_at_each_step)+1),
                    objective_function_at_each_step)
        plt.plot(range(1,len(objective_function_at_each_step)+1),
                 objective_function_at_each_step)
        # for x,y in zip(range(len(objective_function_at_each_step)),objective_function_at_each_step):
        #    plt.text(x,y, "{:.2f}".format(y))
        plt.title('Objective function - iteration ')
        plt.show()

    return objective_function_at_each_step[-1]#, silhouetteMethod(dataset, dataset_labels, centroids, centroid_labels)

def scikitKMeans(dataset, k):
    """
    This function plots the k-means implementation of scikit-learn library.
    """
    scikit_kmeans = KMeans(n_clusters=k).fit(dataset)
    plotTwoDimensionalWithOptionalColors("scikit-learn library clustering", dataset,
                                         scikit_kmeans.cluster_centers_, scikit_kmeans.labels_, np.linspace(0, k, k, endpoint=False))

def elbowMethod(objective_values_for_differenk_ks):
    """
    It's easy to observe optimal K value from plot, but to determine it automatically I came up with an idea.
    My idea is to find the corner point where previoud slope / next slope is maximum. 
    That must be the corner of the elbow shaped curve.
    It works fine for elbow shaped curves but struggles if objective function is not a monotonically decreasing function.
    """
    differences_ratio = []
    for i in range(len(objective_values_for_differenk_ks)-2):
        differences_ratio.append((objective_values_for_differenk_ks[i] - objective_values_for_differenk_ks[i+1])/(
            objective_values_for_differenk_ks[i+1] - objective_values_for_differenk_ks[i+2]))
    # print(differences_ratio)

    idx1 = max(differences_ratio)
    optimal_k = differences_ratio.index(idx1)+3
    print("OPTIMAL K: ", optimal_k, "(found by elbow method)")
    plt.figure()
    plt.scatter(range(2, len(objective_values_for_differenk_ks)+2),
                objective_values_for_differenk_ks)
    plt.plot(range(2, len(objective_values_for_differenk_ks)+2),
             objective_values_for_differenk_ks)
    plt.text(optimal_k, objective_values_for_differenk_ks[optimal_k], "X")
    plt.title('the elbow method for determining optimal K')
    plt.show()

    """
def silhouetteMethod(dataset, dataset_labels, centroids, centroid_labels):
    """
    """
    NOTE: NOT WORKING, I don't know why, implementation seems right.

    I was not satisfied with the elbow method's problem when shape is not like an elbow.

    So, I implemented silhouette method too.
    Main idea of this method is measuring how well a data point is assigned to its cluster compared to other clusters.
    
    CAUTION:This slows down the algorithm greatly since its complexity is Theta(k*n^2).

    We first find a_value which is the mean distance of between a data point and other points in its cluster.
    Then, find b_value which is the minimum of the mean distances between a data point and other points in neighboring clusters.
    s_value is the (a-b) / max(a,b) and between -1 and 1

    Highest s_value represent the optimal k. In my case s_value is increasing and increasing.
    """
    """
    s_vals = []

    for point, label in zip(dataset, dataset_labels):
        a_val = 0
        b_vals = [0]*len(centroid_labels)

        for other_point, other_label in zip(dataset, dataset_labels):
            #for i in range(len(centroid_labels)):
            if label == other_label:
                a_val += math.sqrt(euclideanDistSquared(point1=point, point2=other_point))
            else:
                b_vals[other_label] += math.sqrt(euclideanDistSquared(point1=point, point2=other_point))
        a_val /= ((dataset_labels == label).sum() - 1) if (dataset_labels == label).sum() > 1 else 1
        b_vals = [b_vals[other_label] / (dataset_labels == other_label).sum()  for other_label in centroid_labels]
        b_val = b_vals.index(sorted(b_vals)[1])
        if (dataset_labels == label).sum() = 1:
            s_vals.append(0)
        else:
            s_vals.append((b_val-a_val) / max(a_val, b_val))

    print(len(s_vals))
    return sum(s_vals)/len(s_vals)


def plotSilhouetteScores(silhouette_scores):
    plt.figure()
    plt.scatter(range(2, len(silhouette_scores)+2),
                silhouette_scores)
    plt.plot(range(2, len(silhouette_scores)+2),
             silhouette_scores)
    plt.title('the silhouette method for determining optimal K')
    plt.text(silhouette_scores.index(max(silhouette_scores)) +
             2, max(silhouette_scores), "X")

    plt.show()
    print("OPTIMAL K: ", silhouette_scores.index(
        max(silhouette_scores))+2, "(found by silhouette method)")
"""

if __name__ == "__main__":

    ##### DATASET VARIABLES ########
    """
    NOTE: You can change these variables to test the code. 
    But if program runs longer than expected (30 seconds maybe), stop and change the seed. 
    It may not be able to initialize centroids.
    """
    CLUSTER_COUNT = 4
    K = 15
    SAMPLE_SIZE = 800
    SEED = 13
    ###################
    
    objective_values_for_differenk_ks = []
    #silhouette_scores = []
    ################################# CONVEX DATA #################################
    dataset = generateSimpleDataset(
        cluster_count=CLUSTER_COUNT, size=SAMPLE_SIZE, seed=SEED)
    for i in range(1, K):
        #objective_val, silhouett_val = kMeans(dataset, k=i+1)
        objective_val = kMeans(dataset, k=i+1)
        objective_values_for_differenk_ks.append(objective_val)
        #silhouette_scores.append(silhouett_val)

    elbowMethod(objective_values_for_differenk_ks)
    # ref: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    # ref: https://en.wikipedia.org/wiki/Silhouette_(clustering)
    #plotSilhouetteScores(silhouette_scores)
    for k in k_of_results_to_be_ploted:
        scikitKMeans(dataset, k)  # sk-learn library

    
    objective_values_for_differenk_ks = []
    #silhouette_scores = []
    ################################# NONCONVEX DATA #################################
    dataset = generateNonconvexDataset(size=SAMPLE_SIZE)
    for i in range(1, K):
        #objective_val, silhouett_val = kMeans(dataset, k=i+1)
        objective_val = kMeans(dataset, k=i+1)
        objective_values_for_differenk_ks.append(objective_val)
        #silhouette_scores.append(silhouett_val)

    elbowMethod(objective_values_for_differenk_ks)
    # ref: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    # ref: https://en.wikipedia.org/wiki/Silhouette_(clustering)
    #plotSilhouetteScores(silhouette_scores)
    for k in k_of_results_to_be_ploted:
        scikitKMeans(dataset, k)  # sk-learn library