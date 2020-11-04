# Introduction 
C++ implementation of 3-dimensinal ICP (Iterative Closet Point) algorithm. 

### Algorithm Input & Output
input : two sets of points in three-dimensional space A and B.

output : rotation and translation of B to fit in A.


### Algorithm Steps
1. For each point in A, calculate the matching point (closest point) in B.
2. Minimize the matching error between matching points and calculate the pose.
3. Apply the calculated pose to the point cloud.
4. Recalculate matching points.
5. Iterate until the number of iterations > threshold or the value of the minimized energy function < threshold.

### Project Instruction
- SVD-based least squares best-fitting algorithm is used for the corresponding point set. (Point-to-Point)
- Exhaustive search method is used to find the nearest neighbor of each point.
- Eigen library is used for matrices operations.
- Use [The Stanford Bunny Models](https://graphics.stanford.edu/data/3Dscanrep/) as dataset. 
- Use PCL 1.7 APIs to load datasets and visualize the PointClouds.

#### Requirements
- pcl > 1.7
- cmake > 2.8

#### Build
<pre><code>mkdir build
cd build 
cmake ..
make
</code></pre>
#### Run
<pre><code>./icp1_simple ../data/bun{000,045}mesh.ply
	./icp2_iterative_view ../data/bun{000,045}mesh.ply
</code></pre>


## Extension and Variants in the Future

#### Dataset

 - [x] Import the real-world datasets. 
 - [x] Visualize the datasets.(icp1_simple)
 - [x] Visualize the iterative process.(icp2_iterative_view)

#### Point Subsets
 - [ ] Random Sampling
 - [ ] Voxel Grid Sampling
 - [ ] NSS (Normal Space Sampling)
 - [ ] Feature Detaction

#### Data Association
 - [ ] Use KD-tree/OCtree to find the k-nearst neighbor of each point. 
 - [ ] Normal Shooting
 - [ ] Feature descriptor matching

#### Outlier Rejection
 - [ ] Remove correspondence with high distance for outlier rejection.
 - [ ] Remove worst x% of correspondences for outlier rejection.

#### Loss Function
 - [ ] Point-to-Plane

#### Optimization
 - [ ] CUDA : Data Parallelism

