#include <iostream>
#include <numeric>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;
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
	The nearest neighbor points

	distances[i]: the distence between src_matrix[i] and dst_matrix[indices[i]]
	indices[i]: the nearest point index in destination datasets
*/
typedef struct{
	vector<float> distances;
	vector<int> indices;
} NEIGHBOR;

typedef struct{
	int sourceIndex;
	int targetIndex;
	float distance;
} Align;



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
						[xi',yi',zi'] is the nearest to [xi,yi,zi]
		remainPercentage x = [0 ~ 100], remove worst (100-x)% of correspondence for outlier rejection. 
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
Matrix4d best_fit_transform(const MatrixXd &A,const MatrixXd &B,int remainPercentage=100){
	int row = (int) A.rows()*remainPercentage/100;
	Vector3d centroid_A(0,0,0);
	Vector3d centroid_B(0,0,0);
	MatrixXd AA = A;
	MatrixXd BB = B;
	Matrix4d T = MatrixXd::Identity(4,4);
	
	for(int i=0; i<row; i++){
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}

	centroid_A /= row;
	centroid_B /= row;

	for(int i=0; i<row; i++){
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

	if(R.determinant()<0){
		Vt.block<1,3>(2,0) *= -1;
		R = U*Vt;
	}

	t = centroid_B-R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}


/* 
	#TODO change to KNN
	Input: 
		source      A = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]

		destination B = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]

	Output: 
		T = [R, t]
			R - rotation: 3*3 matrix
			t - tranlation: 3*1 vector
*/
NEIGHBOR nearest_neighbor(const MatrixXd &src, const MatrixXd &dst){
	int row_src = src.rows();
	int row_dst = dst.rows();
	Vector3d vec_src;
	Vector3d vec_dst;
	NEIGHBOR neigh;
	float min = 100; // initial threshould
	int index = 0;
	float dist_temp = 0;

	for(int ii=0; ii<row_src; ii++){
		vec_src = src.block<1,3>(ii,0).transpose();
		min = 100;
		index = 0;
		dist_temp=0;
		for(int jj=0; jj<row_dst; jj++){
			vec_dst = dst.block<1,3>(jj,0).transpose();
			dist_temp = dist(vec_src,vec_dst);
			if(dist_temp<min){ // find the nearst
				min = dist_temp;
				index = jj;
			}
		}

		neigh.distances.push_back(min); // the dis of the min dis
		neigh.indices.push_back(index); // the index of the min dis
	}
	return neigh;
}


int cmpAlign(const void *a, const void *b)
{
	Align *a1 = (Align*)a;
	Align *a2 = (Align*)b;
	return (*a1).distance - (*a2).distance;
}

/* 
	#TODO change to KNN
	Input: 
		source        = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]

		destination   = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]
		threshold : remove correspondence with distance higher than threshold for outlier rejection.

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

	for(int i=0; i<SourceRowr; i++){
		sourceVector = source.block<1,3>(i,0).transpose();
		minDistance = threshold;
		tempIndex = 0;
		tempDistance = 0;
		for(int j=0; j<targetRow; j++){
			targetVector = target.block<1,3>(j,0).transpose();
			tempDistance = dist(sourceVector,targetVector);
			if(tempDistance < minDistance){ // find the nearst
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

	qsort(alignments,alignments.size(),sizeof(align),cmpAlign);
	return alignments;
}



/*
	#TODO the rows of A is not equal to B
	iterative closest point algorithm

	Input: 
		source      A = {a1,...,an}, ai = [x,y,z]
		destination B = {b1,...,bn}, bi = [x,y,z]
		max_iteration
		tolenrance
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


		src3d = [x1,y1,z1]		dst_chorder = [x1,y1,z1]
				|x2,y2,z2|					  |x2,y2,z2|
				|........|					  |........|
				[xn,yn,zn]					  [xn,yn,zn]
		* src3d : save the temp matrix transformed in this iteration
		* dst_chorder : save the temp nearst node in B for node in src3d(temp)


*/
ICP_OUT icp(const MatrixXd &A, const MatrixXd &B, 
			int max_iteration, float tolerance, float outlierThreshold=100){
	
	int row = A.rows();
	MatrixXd src = MatrixXd::Ones(3+1,row); 
	MatrixXd src3d = MatrixXd::Ones(3,row); 
	MatrixXd dst = MatrixXd::Ones(3+1,row); 
	MatrixXd dst_chorder = MatrixXd::Ones(3,row); 
	//NEIGHBOR neighbor;
	vector<Align> alignments;
	Matrix4d T;
	ICP_OUT result;
	int iter = 1;

	for(int i=0; i<row; i++){
		src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // line 4 for t,translate
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // save the temp data
		dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
	}

	double prev_error = 0;
	double mean_error = 0;
	for(int i=0; i<max_iteration; i++,iter=i+1){ // When the number of iterations is less than the maximum
		//neighbor = nearest_neighbor(src3d.transpose(),B); // n*3 , n*3
		alignments = best_alignment(src3d.transpose(),B, outlierThreshold); // n*3, n*3

/*
		for(int j=0; j<row; j++){
			dst_chorder.block<3,1>(0,j) = dst.block<3,1>(0,neighbor.indices[j]);
			// save the nearst node' index to node i
		}
*/
        //T = best_fit_transform(src3d.transpose(),dst_chorder.transpose()); // save the transformation in this iteration

        T = best_fit_transform(src3d.transpose(),dst.transpose(),alignments); // save the transformation in this iteration
		src = T*src; // notice the order of matrix product

		for(int j=0; j<row; j++){
			src3d.block<3,1>(0,j) = src.block<3,1>(0,j); // copy the transformed matrix
		}

		//mean_error = accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();


		mean_error = 0.0f;
		for(auto align : alignments){
			mean_error += align.distance;
		}
		mean_error /= alignments.size();

		if(abs(prev_error-mean_error)<tolerance){
			break;
		}
		
		prev_error = mean_error;
	}
	vector<float> distances;
	for(auto align : alignments){
		distances.push_back(align.distance);
	}
	
	result.trans = best_fit_transform(A,src3d.transpose()); // compute the total transforamtion for all iterations
	result.distances = distances;
	result.iter = iter;
	return result;
}










