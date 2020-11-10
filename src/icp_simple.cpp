#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>

#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>

#include "icp_simple.hpp"

void loadFile(const char* file_name, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
	pcl::PolygonMesh mesh;

	if(pcl::io::loadPolygonFile(file_name, mesh)==-1)
	{
		PCL_ERROR("File loading faild.");
		return;
	}
	else
	{
		pcl::fromPCLPointCloud2<pcl::PointXYZ>(mesh.cloud, cloud);
	}

	std::vector<int> index;
	pcl::removeNaNFromPointCloud(cloud, cloud, index);
}


int main(int argc, char**argv)
{
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>());

	{
		loadFile(argv[1], *cloud_source);
		loadFile(argv[2], *cloud_target);
	}


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans (new pcl::PointCloud<pcl::PointXYZ>());

	{
		Eigen::MatrixXf source_matrix = cloud_source->getMatrixXfMap(3,4,0).transpose();
		Eigen::MatrixXf target_matrix = cloud_target->getMatrixXfMap(3,4,0).transpose();
		
		int max_iteration = 10;
		float tolenrance = 0.000001;

		// call icp
		ICP_OUT icp_result = icp(source_matrix.cast<double>(),target_matrix.cast<double>(),
						 max_iteration,tolenrance);

		int iter = icp_result.iter;	
		Matrix4f T = icp_result.trans.cast<float>();	
		vector<float> distances = icp_result.distances;

		Eigen::MatrixXf source_trans_matrix = source_matrix;
		
		int row = source_matrix.rows();
		MatrixXf source_trans4d = MatrixXf::Ones(3+1,row);
		for(int i=0;i<row;i++)
			source_trans4d.block<3,1>(0,i) = source_matrix.block<1,3>(i,0).transpose();
		source_trans4d = T*source_trans4d;
		for(int i=0;i<row;i++)
			source_trans_matrix.block<1,3>(i,0)=source_trans4d.block<3,1>(0,i).transpose();


		pcl::PointCloud<pcl::PointXYZ> temp_cloud;
		temp_cloud.width = row;
		temp_cloud.height = 1;
		temp_cloud.points.resize(row);
		for (size_t n=0; n<row; n++) 
		{
			temp_cloud[n].x = source_trans_matrix(n,0);
			temp_cloud[n].y = source_trans_matrix(n,1);
			temp_cloud[n].z = source_trans_matrix(n,2);	
  		}
  		cloud_source_trans = temp_cloud.makeShared();
  
	}



	{ // visualization
		boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(255,255,255);

		// black
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(cloud_source,0,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_source,source_color,"source");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source");

		// blue
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_target,0,0,255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_target,target_color,"target");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"target");

		// red
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(cloud_source_trans,255,0,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_source_trans,source_trans_color,"source trans");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,"source trans");

		viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
		viewer->resetCamera();
		viewer->spin();
	}
	return(0);
}











