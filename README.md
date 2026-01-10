## Cluster Estimation using 2D and 3D GPS Coordinates
A proof of concept for the localization for coloured targets during autonomous drone mission using the **Variational Bayesian Gaussian Mixture Model (VBGMM)** to condense clusters of raw, noisy telemetry data into an accurate estimate for the target's centre. 

# The Challenge
Designed for the **Waterloo Aerial Robotics Group** MVP for the Aerial Evolution Association of Canada competition, where after receiving a dataset containing the estimated centres of coloured targets, will return the correct number of targets as well as the target's centre within 0.5m.

<img width="1118" height="800" alt="image" src="https://github.com/user-attachments/assets/ef1b223f-0080-4052-88b5-c54e88cb0755" />

# How It's Made:
Python, SciKit, Numpy

1. Data Normalization: A NumPy-based **StandardScaler** is used to perform a Z-score normalization on the raw KML coordinates. This magnifies the small changes in latitude and longitude, so the model is more effective at evaluating the changes.

2. Variational Bayesian inference, where the points are put through the SciKit Model to determine cluster centres. This requires several pre-determined parameters:
- Covariance Type: The expected, general shape of the targets
- n_components: The maximum number of centroids for the model to return
- Init Parameters: Selected algorithm to make the initial centroid estimations.
- Weight Concentration Prior: A higher value indicates fewer, large clusters. A low value means more small clusters will be kept alive.
- Mean Precision Prior: How confident the model is with its initial estimations.

3. Post Processing: Included for higher accuracy of the model, especially with a variable number of centroids.
- Filter Points by Ownership:
  - Consider the number of points each estimated centroid has. If it is less than the pre-determined mininum points per cluster, remove that centroid.
- Filter by Covariance:
   - Covariance refers to the size of the cluster. States the largest acceptable size of a target, in which case it should removed and considered scattered sensor noise.
- Using a Weight Drop Threshold:
  - Weight is the fraction of points the cluster has of the entire dataset. Without noise, the combined weight of each centroid should be 1.
  - The centroids are sorted by weight from largest to smallest. If there is a steep drop in weight (by a factor of the weight drop threshold), all further centroids should be removed.

This program is also structured to be implemented through the MAVLink Protocol, featuring a worker to handle all of the calculations and logic, and can be treated as an object that will do this task. To validate the worker, this repository also includes a script to test the worker, using kml files collected from real-time drone flight data, or generated.
