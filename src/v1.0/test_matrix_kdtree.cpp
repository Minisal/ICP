#include "nanoflann.hpp"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;
using namespace nanoflann;

const int SAMPLES_DIM = 15;

template <typename Der>
void generateRandomPointCloud(MatrixBase<Der> &mat, 
							  const size_t N, const size_t dim,
							  const typename Der::Scalar max_range = 10)
{
	cout<<"Generating "<<N<<" random points...";
	mat.resize(N, dim);
	for(size_t i=0;i<N;i++)
		for(size_t d=0;d<dim;d++)
			mat(i,d) = max_range*(rand()%1000)/ typename Der::Scalar(1000);
	cout<<"done\n";
}

template <typename num_t>
void kdtree_demo(const size_t nSamples, const size_t dim)
{
	Matrix<num_t,Dynamic,Dynamic> mat(nSamples, dim);

	const num_t max_range = 20;

	// Generate points:
	generateRandomPointCloud(mat, nSamples, dim, max_range);
	cout<<mat<<endl;

	// Query point:
	vector<num_t> query_pt(dim);
	for (size_t d=0; d<dim; d++)
		query_pt[d] = max_range*(rand()%1000)/num_t(1000);

	typedef KDTreeEigenMatrixAdaptor<Matrix<num_t,Dynamic,Dynamic> > my_kd_tree_t;

	my_kd_tree_t mat_index(dim, cref(mat), 10);
	mat_index.index->buildIndex();

	const size_t num_results = 3;
	vector<size_t> ret_indexes(num_results);
	vector<num_t> out_dists_sqr(num_results);

	nanoflann::KNNResultSet<num_t> resultSet(num_results);

	resultSet.init(&ret_indexes[0],&out_dists_sqr[0]);
	mat_index.index->findNeighbors(resultSet, &query_pt[0],nanoflann::SearchParams(10));

	cout<<"knnSreach(nn="<<num_results<<"): \n";
	for(size_t i=0; i<num_results; i++)
		cout<<"ret_index["<<i<<"]="<<ret_indexes[i]<<"out_dists_sqr="<<out_dists_sqr[i]<<endl;
}

int main(int argc, char**argv)
{
	srand(static_cast<unsigned int>(time(nullptr)));
	kdtree_demo<float>(1000, SAMPLES_DIM);
	return 0;
}


















