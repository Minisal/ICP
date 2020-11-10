
#include <iostream>
#include <numeric>
#include "Eigen/Eigen"
#include "nanoflann.hpp"

using namespace std;
using namespace Eigen;
using namespace nanoflann;

/*
	matrix.block<p,q>(i,j);
	Extract blocks with a size of p*q started in (i,j)
*/

/*
	The output of ICP algorithm

	trans : transformation for best align
	dictances[i] : the distance between node i in src and its nearst node in dst
	inter : number of iterations
*/
typedef struct{
	Matrix4d trans;
	vector<float> distances;
	int iter;
} ICP_OUT;


/*
	The nearest neighbor point

	distance: the distence between src_matrix[sourceIndex] 
			  and dst_matrix[targetIndex]
*/
typedef struct{
	int sourceIndex;
	int targetIndex;
	float distance;
} Align;


/*
	The k nearest neighbor points

	distances[i]: the distence between src_matrix[sourceIndex] 
			  	  and dst_matrix[targetIndexes[i]]
			      i := 1~K
*/
typedef struct{
	int sourceIndex;
	vector<int> targetIndexes;
	vector<float> distances;
	float distanceMean;
} KNeighbor;



int cmpAlign(const void *a, const void *b)
{
	Align *a1 = (Align*)a;
	Align *a2 = (Align*)b;
	return (*a1).distance - (*a2).distance;
}

int cmpKNeighbor(const void *a, const void *b)
{
	KNeighbor *a1 = (KNeighbor*)a;
	KNeighbor *a2 = (KNeighbor*)b;
	return (*a1).distanceMean - (*a2).distanceMean;
}


/*
	Compute the distance between two 1*3 vector
	sqrt(sum(a[i]-b[i])^2), i:=0,1,2
*/
float dist(const Vector3d &a, const Vector3d &b){
	return sqrt((a[0]-b[0])*(a[0]-b[0])
			   +(a[1]-b[1])*(a[1]-b[1])
			   +(a[2]-b[2])*(a[2]-b[2]));
}


/*
	Transform A to align best with B
	(B has be correspondented to A)

	Input: 
		source     	A = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]

		destination B = [x1',y1',z1']
						|x2',y2',z2'|
						|...........|
						[xn',yn',zn']
					  # [xi',yi',zi'] is the nearest to [xi,yi,zi]
		
	Output: 
		T = [R, t]
			 R - rotation: 3*3 matrix
			 t - tranlation: 3*1 vector

	"best align" equals to find the min value of 
					sum((bi-R*ai-t)^2)/N, i:=1~N
	the solution is:
		centroid_A = sum(ai)/N, i:=1~N
		centroid_B = sum(bi)/N, i:=1~N
		AA = {ai-centroid_A}
		BB = {bi-centroid_B}
		H = AA^T*BB
		U*S*Vt = singular_value_decomposition(H)

		R = U*Vt
		t = controid_B-R*centroid_A
*/
Matrix4d best_fit_transform(const MatrixXd &A,const MatrixXd &B)
{
	size_t row = min(A.rows(),B.rows());

	Vector3d centroid_A(0,0,0);
	Vector3d centroid_B(0,0,0);
	
	MatrixXd AA;
	MatrixXd BB;
	if(A.rows()>B.rows()) AA = BB = B;
	else BB = AA = A;

	Matrix4d T = MatrixXd::Identity(4,4);

	for(int i=0; i<row; i++)
	{
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}

	centroid_A /= row;
	centroid_B /= row;

	for(int i=0; i<row; i++)
	{
		AA.block<1,3>(i,0) = A.block<1,3>(i,0)-centroid_A.transpose();
		BB.block<1,3>(i,0) = B.block<1,3>(i,0)-centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB;
	MatrixXd U;
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;
	Matrix3d R;
	Vector3d t;
	
	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
	// JacobiSVD decomposition computes only the singular values by default. 
	// ComputeFullU or ComputeFullV : ask for U or V explicitly.
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

	R = U*Vt;

	if(R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = U*Vt;
	}

	t = centroid_B-R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}
/*
	Input : A : n*3 matrix
			B : n*3 matrix
    	    neighbors : Indexes and distances of k closest points match.
		    remainPercentage x = [0 ~ 100] : Remove worst (100-x)% of 
		    					correspondence for outlier rejection. 
*/
Matrix4d best_fit_transform(const MatrixXd &A,const MatrixXd &B, 
							vector<KNeighbor> neighbors,int remainPercentage=100,int K=5)
{
	int num = (int) neighbors.size()*remainPercentage/100;
	Vector3d centroid_A(0,0,0);
	Vector3d centroid_B(0,0,0);
	Vector3d temp(0,0,0);
	MatrixXd AA = A;
	MatrixXd BB = A;
	Matrix4d T = MatrixXd::Identity(4,4);

	for(int i=0; i<num; i++)
	{
		int aIndex = neighbors[i].sourceIndex;
		centroid_A += A.block<1,3>(aIndex,0).transpose();
		for(int k=0;k<K;k++)
		{
			int bIndex = neighbors[i].targetIndexes[k];
			centroid_B += B.block<1,3>(bIndex,0).transpose();
		}
		centroid_B /= K;
	}

	centroid_A /= num;
	centroid_B /= num;

	for(int i=0; i<num; i++)
	{
		int aIndex = neighbors[i].sourceIndex;
		AA.block<1,3>(i,0) = A.block<1,3>(aIndex,0)-centroid_A.transpose();

		for(int k=0;k<K;k++)
		{
			int bIndex = neighbors[i].targetIndexes[k];
			BB.block<1,3>(i,0) = B.block<1,3>(bIndex,0)-centroid_B.transpose();
		}
		BB.block<1,3>(i,0) /= K;		
	}

	MatrixXd H = AA.transpose()*BB;
	MatrixXd U;
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;
	Matrix3d R;
	Vector3d t;

	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
	// JacobiSVD decomposition computes only the singular values by default. 
	// ComputeFullU or ComputeFullV : ask for U or V explicitly.
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

	R = U*Vt;

	if(R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = U*Vt;
	}

	t = centroid_B-R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}

/* 
	Input: 
		source        = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]

		destination   = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]
		threshold : remove correspondence with distance higher 
					than this threshold for outlier rejection.

	Output: 
		vector<Align> sorted by distance from smallest to largest
*/
vector<Align> best_alignment(const MatrixXd& source, const MatrixXd& target, float threshold=100)
{
	int SourceRow = source.rows();
	int targetRow = target.rows();
	Vector3d sourceVector;
	Vector3d targetVector;
	vector<Align> alignments;
	float minDistance = threshold; // initial to threshould
	int tempIndex = 0;
	float tempDistance = 0;

	for(int i=0; i<SourceRow; i++)
	{
		sourceVector = source.block<1,3>(i,0).transpose();
		minDistance = threshold;
		tempIndex = 0;
		tempDistance = 0;
		for(int j=0; j<targetRow; j++)
		{
			targetVector = target.block<1,3>(j,0).transpose();
			tempDistance = dist(sourceVector,targetVector);
			if(tempDistance < minDistance)
			{ // find the nearst
				minDistance = tempDistance;
				tempIndex = j;
			}
		}
		Align align; 
		align.sourceIndex=i;
		align.targetIndex=tempIndex;
		align.distance=minDistance;
		alignments.push_back(align);
	}

	qsort(&alignments[0],alignments.size(),sizeof(Align),cmpAlign);
	return alignments;
}

vector<KNeighbor> k_nearest_neighbors(const MatrixXd& source, const MatrixXd& target, float leaf_size=10, int K=5)
{
	int dimension = 3;
	int SourceRow = source.rows();
	int targetRow = target.rows();
	Vector3d sourceVector;
	Vector3d targetVector;
	vector<KNeighbor> neighbors;
	int tempIndex = 0;
	float tempDistance = 0;

	// build kdtree
	Matrix<float,Dynamic,Dynamic> targetMatrix(targetRow,dimension);
	for(int i=0; i<targetRow; i++)
		for(int d=0;d<dimension;d++)
			targetMatrix(i,d)=target(i,d);
	typedef KDTreeEigenMatrixAdaptor<Matrix<float,Dynamic,Dynamic> > kdtree_t;
	kdtree_t targetKDtree(dimension, cref(targetMatrix),leaf_size);
	targetKDtree.index->buildIndex();

	for(int i=0; i<SourceRow; i++)
	{
		vector<float> sourcePoint(dimension);
		float meanDis = 0.0f;

		for(size_t d=0;d<dimension;d++)
			sourcePoint[d]=source(i,d);

		vector<size_t> result_indexes(K);
		vector<float> result_distances(K);
		nanoflann::KNNResultSet<float> resultSet(K);
		resultSet.init(&result_indexes[0],&result_distances[0]);
		nanoflann::SearchParams params_igonored;
		targetKDtree.index->findNeighbors(resultSet,&sourcePoint[0],params_igonored);

		KNeighbor neigh;
		neigh.sourceIndex=i;
		for(int i=0;i<K;i++)
		{
			neigh.targetIndexes.push_back(result_indexes[i]);
			neigh.distances.push_back(result_distances[i]);
			meanDis += result_distances[i];
		}
		neigh.distanceMean = meanDis/K;
		neighbors.push_back(neigh);
	}

	qsort(&neighbors[0],neighbors.size(),sizeof(KNeighbor),cmpKNeighbor);
	return neighbors;
}


/*
	iterative closest point algorithm

	Input: 
		source      A = {a1,...,an}, ai = [x,y,z]
		destination B = {b1,...,bn}, bi = [x,y,z]
		max_iteration
		tolenrance
		outlierThreshold
	Output: 
		ICP_OUT->
			trans : transformation for best align
			dictances[i] : the distance between node i in src and its nearst node in dst
			inter : number of iterations

	Matrix:

		A = [x1,y1,z1]			B = [x1,y1,z1]
			|x2,y2,z2|				|x2,y2,z2|
			|........|				|........|
			[xn,yn,zn]				[xn,yn,zn]

		src = [x1,x2,x3, ...]		dst = [x1,x2,x3, ...]
			  |y1,y2,y3, ...|			  |y1,y2,y3, ...|
			  |z1,z2,z3, ...|			  |z1,z2,z3, ...|
			  [ 1, 1, 1, ...]			  [ 1, 1, 1, ...]
		* The last line is set for translation t, so that the T*src => M(3x4)*M(4*n)
		*  Notice that when src = T*src, the last line's maintain 1 and didn't be covered


		src3d = [x1,y1,z1]		
				|x2,y2,z2|		
				|........|		
				[xn,yn,zn]		
		* src3d : save the temp matrix transformed in this iteration
*/
ICP_OUT icp(const MatrixXd &A, const MatrixXd &B, 
			int max_iteration, float tolerance, int leaf_size=10, int Ksearch=5)
{
	size_t row = min(A.rows(),B.rows());
	MatrixXd src = MatrixXd::Ones(3+1,row); 
	MatrixXd src3d = MatrixXd::Ones(3,row); 
	MatrixXd dst = MatrixXd::Ones(3+1,row); 
	MatrixXd dst3d = MatrixXd::Ones(3,row);
    vector<KNeighbor> neighbors;
	Align alignments;
  	Matrix4d T;
  	Matrix4d T_all = MatrixXd::Identity(4,4);
	ICP_OUT result;
	int iter;

	for(int i=0; i<row; i++)
	{
		src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // line 4 for t:translate
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // save the temp data
		dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
		dst3d.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
	}

	double prev_error = 0;
	double mean_error = 0;

	// When the number of iterations is less than the maximum
	for(iter=0; iter<max_iteration; iter++)
	{ 
		neighbors = k_nearest_neighbors(src3d.transpose(),B); // n*3,n*3

		// save the transformed matrix in this iteration
		T = best_fit_transform(src3d.transpose(),dst.transpose(),neighbors);
		T_all = T*T_all;
		src = T*src; // notice the order of matrix product

		// copy the transformed matrix
		for(int j=0; j<row; j++)
			src3d.block<3,1>(0,j) = src.block<3,1>(0,j); 


		// calculate the mean error
		mean_error = 0.0f;
		for(int i=0; i<neighbors.size(); i++)
			mean_error += neighbors[i].distanceMean;
		mean_error /= neighbors.size();
		cout<<"error"<<prev_error-mean_error<<endl;
		if(abs(prev_error-mean_error)<tolerance)
			break;
	
		prev_error = mean_error;
	}

	vector<float> distances;
	for(int i=0; i<neighbors.size(); i++)
		distances.push_back(neighbors[i].distanceMean);

	result.trans = T_all;
	result.distances = distances;
	result.iter = iter+1;

	return result;
}










