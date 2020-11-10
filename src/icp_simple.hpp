#include <iostream>
#include <numeric>
#include "Eigen/Eigen"
#include "nanoflann.hpp"


using namespace std;
using namespace Eigen; //#TODO remove header importing
using namespace nanoflann;

typedef struct{
	Matrix4d trans;
	vector<float> distances;
	int iter;
} ICP_OUT;

typedef struct{
	vector<int> indexes;
	vector<float> distances;
	float mean_distances;
} NEIGHBOR;

float dist(const Vector3d &a, const Vector3d &b)
{
	return sqrt((a[0]-b[0])*(a[0]-b[0])
			   +(a[1]-b[1])*(a[1]-b[1])
			   +(a[2]-b[2])*(a[2]-b[2]));
}



NEIGHBOR nearest_neighbor_kdtree(const MatrixXd &src, const MatrixXd &tgt)
{
	int dim = 3;
	int leaf_size = 10;
	int K = 1;
	int row_src = src.rows();
	int row_tgt = tgt.rows();
	Vector3d vec_src;
	Vector3d vec_tgt;
	NEIGHBOR neighbor;

	// build kdtree
	Matrix<float,Dynamic,Dynamic> matrix_tgt(row_tgt,dim);
	for (int i=0; i<row_tgt; i++)
		for (int d=0; d<dim; d++)
			matrix_tgt(i,d) = tgt(i,d);
	typedef KDTreeEigenMatrixAdaptor<Matrix<float,Dynamic,Dynamic> >kdtree_t;
	kdtree_t kdtree_tgt(dim,cref(matrix_tgt),leaf_size);
	kdtree_tgt.index->buildIndex();

	for (int i=0; i<row_src; i++)
	{
		vector<float> point_src(dim);
		for(int d=0; d<dim; d++)
			point_src[d] = src(i,d);

		vector<size_t> index(K);
		vector<float> distance(K);
		nanoflann::KNNResultSet<float> result_set(K);
		result_set.init(&index[0], &distance[0]);
		nanoflann::SearchParams params_ignored;
		kdtree_tgt.index->findNeighbors(result_set,&point_src[0],params_ignored);
		neighbor.indexes.push_back(index[0]);
		neighbor.distances.push_back(distance[0]);
	}
	return neighbor; 
}

/*
NEIGHBOR nearest_neighbor(const MatrixXd &src, const MatrixXd &tgt)
{
	size_t row_src = src.rows();
	size_t row_tgt = tgt.rows();
	Vector3d vec_src;
	Vector3d vec_tgt;
	NEIGHBOR neighbor;
	float min = 100;
	int index = 0;
	float dist_temp = 0;

	for (size_t i=0; i<row_src; i++)
	{
		vec_src = src.block<1,3>(i,0).transpose();
		min=100;index=0;dist_temp=0;
		for (size_t j=0; j<row_tgt; j++)
		{
			vec_tgt = tgt.block<1,3>(j,0).transpose();
			dist_temp = dist(vec_src,vec_tgt);
			if (dist_temp<min)
			{
				min = dist_temp;
				index = j;
			}
		}
		neighbor.distances.push_back(min);
		neighbor.indexes.push_back(index);
	}
	return neighbor;
}
*/

Matrix4d best_fit_transform (const MatrixXd &A,const MatrixXd &B)
{
	Matrix4d T = MatrixXd::Identity(4,4);
	Vector3d centroid_A(0,0,0);
	Vector3d centroid_B(0,0,0);

	MatrixXd AA;
	MatrixXd BB;
	size_t row = min(A.rows(),B.rows());
	if(A.rows()>B.rows()) AA = BB = B;
	else 				  BB = AA = A;

	for (size_t i=0; i<row; i++)
	{
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}
	centroid_A /= row;
	centroid_B /= row;

	for (size_t i=0; i<row; i++)
	{
		AA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroid_A.transpose(); 
		BB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB;
	MatrixXd U;
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;
	Matrix3d R;
	Vector3d t;

	JacobiSVD<MatrixXd>svd(H,ComputeFullU|ComputeFullV);
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

	R = (U*Vt).transpose(); //#TODO

	if (R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = (U*Vt).transpose();
	}

	t = centroid_B - R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}


ICP_OUT icp (const MatrixXd &A,const MatrixXd &B,
	         int max_iteration, double tolenrance)
{
	size_t row = min(A.rows(),B.rows());
	MatrixXd src = MatrixXd::Ones(3+1,row);
	MatrixXd src3d = MatrixXd::Ones(3,row); // transformed temp
	MatrixXd tgt = MatrixXd::Ones(3+1,row);
	MatrixXd tgt3d = MatrixXd::Ones(3,row); // nearest
	MatrixXd T;
	MatrixXd Trans;
	NEIGHBOR neighbor;
	ICP_OUT result;
	int iter = 0;

	// alter the formulation of data
	for (size_t i=0; i<row; i++)
	{
		src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
		tgt.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
	}

	double prev_error = 0;
	double mean_error = 0;
	for (iter=0; iter<max_iteration; iter++)
	{
cout<<endl<<"iter:"<<iter<<endl;
		// find the correspondence
		neighbor = nearest_neighbor_kdtree(src3d.transpose(),B);

		for (size_t i=0; i<row; i++)
			tgt3d.block<3,1>(0,i) = tgt.block<3,1>(0,neighbor.indexes[i]);
cout<<"find best fit transform"<<endl;
		T = best_fit_transform(src3d.transpose(),tgt3d.transpose());
cout<<T<<endl<<endl;
		src = T*src;

		if(iter==0) Trans = T;
		else 		Trans = T*Trans;

		for (size_t i=0; i<row; i++)
			src3d.block<3,1>(0,i) = src.block<3,1>(0,i);
cout<<"calculate error diffenrence:";
		mean_error = 0.0f;
		for (size_t i=0; i<neighbor.distances.size();i++)
			mean_error += neighbor.distances[i];
		mean_error /= neighbor.distances.size();
		if (abs(prev_error - mean_error) < tolenrance)
			break;
		prev_error = mean_error;
cout<<mean_error<<endl;
	}

	T = best_fit_transform(A,src3d.transpose());
cout<<endl<<"final transformation"<<endl<<T<<endl<<endl;
	result.trans = T;
	result.distances = neighbor.distances;
	result.iter = iter;
	return result;
}




















