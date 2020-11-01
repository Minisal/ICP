## Introduction 
C++ implementation of 3-dimensinal ICP (Iterative Closet Point) algorithm. 

#### Algorithm Input & Output
input : two sets of points in three-dimensional space A and B
output : rotation and translation of B to fit in A


#### Algorithm Steps
1. For each point in A, calculate the matching point (closest point) in B.
2. Minimize the matching error between matching points and calculate the pose.
3. Apply the calculated pose to the point cloud.
4. Recalculate matching points.
5. Iterate until the number of iterations > threshold or the value of the minimized energy function < threshold.

#### Project Instruction
- SVD-based least squares best-fitting algorithm is used for the corresponding point set. (Point-to-Point)
- Exhaustive search method is used to find the nearest neighbor of each point.
- Eigen library is used for matrices operations.


## Extension and Variants in the Future

#### Dataset

 - [ ] Import the real-world data sets.

#### Point Subsets
 - [ ] Random Sampling
 - [ ] Voxel Grid Sampling
 - [ ] NSS (Normal Space Sampling)
 - [ ] Feature Detaction

#### Data Association
 - [ ] Use KD-tree/OCtree
  to find the k-nearst neighbor of each point. 
 - [ ] Normal Shooting
 - [ ] Feature descriptor matching

#### Outlier Rejection
 - [ ] Remove correspondence with high distance for outlier rejection.
 - [ ] Remove worst x% of correspondences for outlier rejection.

#### Loss Function
 - [ ] Point-to-Plane

