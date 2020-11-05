#include <iostream>
#include <vector>
#include <numeric>
#include <sys/time.h>
#include "Eigen/Eigen"
#include "icp.cpp"

using namespace std;
using namespace Eigen;

#define NUM_Points 30       // number of points in the test set
#define NUM_Tests 2         // number of iterations of the test
#define NOISE_SIGMA 0.01    // standard deviation error of the target points
#define MAX_Translation 0.1 // max translation of the target points
#define MAX_Rotation 0.1    // max rotation (radians) of the target points
#define MAX_iteration 20
#define TOLENRANCE 0.04     // the tolenrance of distance mean error


/*
	Return a random float variable with a value of 0 to 1.
	Three decimal places.
*/
float random_float(void)
{
	float f = rand()%100;
	return f/1000;
}


/*
	Input : a n*3 matrix
	Output : a n*3 matrix with diffenrent order of 3D vectors 
*/
void matrix_random_shuffle(MatrixXd &matrix)
{
	int row = matrix.rows();
	vector<Vector3d> temp;
	for(int jj=0;jj<row;jj++)
		temp.push_back(matrix.block<1,3>(jj,0));
	random_shuffle(temp.begin(),temp.end());
	for(int jj=0;jj<row;jj++)
		matrix.block<1,3>(jj,0)=temp[jj].transpose(); 

}


/*
	Input : 
		axis - the rotation axis
		theta - the rotation angle
	Output : 
		R - 3x3 rotation matrix
*/
Matrix3d rotation_matrix_genaration(Vector3d axis, float theta)
{
	AngleAxisd rotationVector(theta, axis.normalized());
	Matrix3d R = Matrix3d::Identity();
	R = rotationVector.toRotationMatrix();
	return R;

	/*
	axis/=sqrt(axis.transpose()*axis); // nomarlise 
	float a = cos(theta/2);
	Vector3d temp = -axis*sin(theta/2);
	float b,c,d;
	b = temp(0);
	c = temp(1);
	d = temp(2);
	Matrix3d R;
	R << a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c),
		 2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b),
		 2*(b*d-a*c), 2*(c*d+a*b), a*+d*d-b*b-c*c;
	return R;
	*/
}


/*
	Return the number of millisecondes 
	elapsed from Epoch to calling this function.
*/
unsigned GetTickCount()
{
/*
	struct timeval
	{
		__time_t tv_sec;        // Seconds
		__suseconds_t tv_usec;  // Microseconds
	};
*/
	struct timeval tv;
	if(gettimeofday(&tv,NULL)!=0)
		return 0;
	return (tv.tv_sec*1000)+(tv.tv_usec/1000);
}


/*
	#TODO : import the pcl
*/
void test_best_fit(void)
{
	MatrixXd A = MatrixXd::Random(NUM_Points,3); // src
	MatrixXd B;  // dst = src*[R t]
	MatrixXd C;  // src*[R1 t1]
	Vector3d t;  // <= random value
	Matrix3d R;
	Matrix4d T;  // =[R1 t1]
	Vector3d t1; // <= computed from best-fit
	Matrix3d R1;

	float total_time = 0;
	unsigned start, end;
	float interval;

	for(int i=0; i<NUM_Tests; i++)
	{
		B = A;

		// add random translation
		t = Vector3d::Random()*MAX_Translation;
		for(int jj=0; jj<NUM_Points; jj++)
			B.block<1,3>(jj,0) = B.block<1,3>(jj,0)+t.transpose();
		// add random rotation
		R = rotation_matrix_genaration(Vector3d::Random(), random_float()*MAX_Rotation);
		B = B*R;

		B += MatrixXd::Random(NUM_Points,3)*NOISE_SIGMA;

		start = GetTickCount(); 
		T = best_fit_transform(A,B);
		end = GetTickCount();
		interval = float((end-start))/1000;
		total_time+=interval;

		C = MatrixXd::Ones(NUM_Points,4);
		C.block<NUM_Points,3>(0,0) = A; 

		C = (T*C.transpose()).transpose();
		R1 = T.block<3,3>(0,0);
		t1 = T.block<3,1>(0,3);

		cout<<"BESTFIT_TEST"<<i<<":"<<endl;
		cout<<"position error"<<endl<<C.block<NUM_Points,3>(0,0)-B<<endl<<endl;
		cout<<"trans error"<<endl<<t1-t<<endl<<endl;
		cout<<"R error"<<endl<<R1-R<<endl<<endl;
		cout<<"------------------------------------------"<<endl;

	}
	cout<<"average fit time:"<<total_time/NUM_Tests<<endl;
	cout<<"=========================================="<<endl;
}


void test_icp(void)
{
	MatrixXd A = MatrixXd::Random(NUM_Points,3);
	MatrixXd B;
	MatrixXd C;
	Vector3d t;
	Matrix3d R;
	Matrix4d T;
	Vector3d t1;
	Matrix3d R1;
	ICP_OUT icp_result;
	vector<float> dist; 
	int iter;
	float mean;
	float total_time = 0;
	unsigned start, end;
	float interval;

	for(int i=0; i<NUM_Tests; i++)
	{
		B = A;
		t = Vector3d::Random()*MAX_Translation;
		for(int jj=0; jj<NUM_Points; jj++)
			B.block<1,3>(jj,0) = B.block<1,3>(jj,0) + t.transpose();
		R = rotation_matrix_genaration(Vector3d::Random(), random_float()*MAX_Rotation);
		B = (R*B.transpose()).transpose();

		B += MatrixXd::Random(NUM_Points,3)*NOISE_SIGMA;
		matrix_random_shuffle(B);
		start = GetTickCount();
		icp_result = icp(A,B,MAX_iteration,TOLENRANCE);
		end = GetTickCount();
		interval = float((end-start))/1000;
		total_time += interval;
		T = icp_result.trans;
		dist = icp_result.distances;
		iter = icp_result.iter;
		mean = accumulate(dist.begin(),dist.end(),0.0)/dist.size();
		C = MatrixXd::Ones(NUM_Points,4);
		C.block<NUM_Points,3>(0,0) = A;
		C = C*T.transpose();
		R1 = T.block<3,3>(0,0);
		t1 = T.block<3,1>(0,3);

		cout<<"ICP_TEST"<<i<<":"<<endl;
		cout<<"mean error"<<endl<<mean-6*NOISE_SIGMA<<endl<<endl;//????
		cout<<"icp trans error"<<endl<<t1-t<<endl<<endl;
		cout<<"icp R error"<<endl<<R1-R<<endl<<endl;
		cout<<"---------------------------------"<<endl;

	}
	cout<<"average icp time:"<<total_time/NUM_Tests<<endl;
	cout<<"=========================================="<<endl;
}


int main(int argc, char*argv[]){
	test_best_fit();
	test_icp();
	return 0;
}