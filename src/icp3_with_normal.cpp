#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>

#include <vtkRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkCamera.h>

using namespace pcl;


void addNormal(PointCloud<PointXYZ>::Ptr cloud,
			   PointCloud<PointXYZRGBNormal>::Ptr cloud_with_normals)
{
	PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);

	pcl::search::KdTree<PointXYZ>::Ptr searchTree(new pcl::search::KdTree<PointXYZ>);
	searchTree->setInputCloud(cloud);

	NormalEstimation<PointXYZ,Normal> normalEstimator;
	normalEstimator.setInputCloud(cloud);
	normalEstimator.setSearchMethod(searchTree);
	normalEstimator.setKSearch(15);
	normalEstimator.compute(*normals);

	concatenateFields(*cloud, *normals, *cloud_with_normals);
}

void loadFile(const char* fileName, PointCloud<PointXYZ> &cloud)
{
	PolygonMesh mesh;

	if(io::loadPolygonFile(fileName, mesh) == -1)
	{
		PCL_ERROR("File loading failed.");
		return;
	}
	else
	{
		fromPCLPointCloud2<PointXYZ>(mesh.cloud, cloud);
	}

	std::vector<int> index;
  	pcl::removeNaNFromPointCloud ( cloud, cloud, index );
}

int main(int argc, char** argv)
{
	PointCloud<PointXYZ>::Ptr cloud_source(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr cloud_target(new PointCloud<PointXYZ>());
	{
    loadFile ( argv[1], *cloud_source );
    loadFile ( argv[2], *cloud_target );
  	}

  	PointCloud<PointXYZ>::Ptr cloud_source_trans(new PointCloud<PointXYZ>());
  	cloud_source_trans = cloud_source;

  	boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D Viewer"));
  	viewer->setBackgroundColor(0,0,0);

  	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color ( cloud_source, 0, 255, 0 );
  	viewer->addPointCloud<pcl::PointXYZ> (cloud_source, source_color, "source");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
  
  	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color ( cloud_target, 255, 255, 255 );
  	viewer->addPointCloud<pcl::PointXYZ> ( cloud_target, target_color, "target");
  	viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target" );
  
  	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color ( cloud_source_trans, 255, 0, 255 );
  	viewer->addPointCloud<pcl::PointXYZ> ( cloud_source_trans, source_trans_color, "source trans" );
  	viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source trans" );
  
  
  	viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection( 1 );
  	viewer->resetCamera();



  	// prepare could with normals
  	PointCloud<PointXYZRGBNormal>::Ptr cloud_source_normals(new PointCloud<PointXYZRGBNormal>());
  	PointCloud<PointXYZRGBNormal>::Ptr cloud_target_normals(new PointCloud<PointXYZRGBNormal>());
  	PointCloud<PointXYZRGBNormal>::Ptr cloud_source_trans_normals(new PointCloud<PointXYZRGBNormal>());

  	addNormal(cloud_source, cloud_source_normals);
  	addNormal(cloud_target, cloud_target_normals);
  	addNormal(cloud_source_trans, cloud_source_trans_normals);

  	IterativeClosestPointWithNormals<PointXYZRGBNormal,PointXYZRGBNormal>::Ptr icp(new IterativeClosestPointWithNormals<PointXYZRGBNormal,PointXYZRGBNormal>);
  	icp->setMaximumIterations(1);
  	icp->setInputSource(cloud_source_trans_normals);
  	icp->setInputTarget(cloud_target_normals);

  	while(!viewer->wasStopped())
  	{
  		icp->align(*cloud_source_trans_normals);

  		if(icp->hasConverged())
  		{
  			transformPointCloud(*cloud_source, *cloud_source_trans, icp->getFinalTransformation());
  			viewer->updatePointCloud(cloud_source_trans,source_trans_color,"source trans");
  			cout<<icp->getFitnessScore()<<endl;
  		}
  		else
  			cout<<"Not converged."<<endl;

  		viewer->spinOnce();
  	}

  	return 0;
}
















