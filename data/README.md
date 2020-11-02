This folder contains the [standford bunny data](http://graphics.stanford.edu/data/3Dscanrep/) (bun000.ply, bun045.ply)

Note: The pcl::io::loadPolygonFile() can not load the original bun{000,045}.ply files bacause the face element does not exist. Hence use meshlab to add face element.

bun000mesh.ply and bun045mesh.ply are created by
- reading the ply file by meshlab, and
- exporting the mesh into another ply file by meshlab 
	1. select "File"->"Export Mesh As..." then 
	2. choose options "Vert: Flags, Normal" and "Face: Flags") 







### DATA INSTRUCTION

			      Range Data

		      Stanford Range Repository
		     Computer Graphics Laboratory
			 Stanford University

			    August 4, 1996


These data files were obtained with a Cyberware 3030MS optical
triangulation scanner.  They are stored as range images in the "ply"
format.  

For more information, consult the web pages of the Stanford Graphics
Laboratory:

	http://www-graphics.stanford.edu

