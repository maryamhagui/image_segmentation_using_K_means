# k-means-implementation
The k-means method is a classic classification tool which allows a data set to be divided into k homogeneous classes. Most of the images (photos, 2D vector drawings, 
3D syntheses, etc.) locally verify properties of homogeneity, in particular in terms of light intensity. The k-means algorithm therefore makes it possible to provide
a solution to the segmentation of images.
The K-Means algorithm can be used to segment an image that has areas of relatively uniform color. We represent all the pixels of the image 
in a three-dimensional space based on their Red / Green / Blue components. We thus obtain a point cloud on which we apply the k-means algorithm.
To illustrate the use of k-means, we use a synthetic image made up of two zones with clearly distinct colors.
## Example of implementation
We will give our algorithm a multi-color input image which consists of different intensities: (we always do the test with the initial image and see the results while changing the number of clusters k)
![GitHub Logo](images/lena.jpg)
This image consists of 512 x 512 pixels
* For k = 2:
The figure below clearly illustrates the 3D point cloud graphical representation that represents the pixels of our input image.
