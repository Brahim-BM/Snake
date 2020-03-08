# Snake

An active contour is a set of points that we will try to move to make it fit a shape. It is a data extraction technique used in image processing. The idea of this method is to move the points to bring them closer to areas of strong gradient while maintaining characteristics such as the curvature of the contour or the distribution of points on the contour or other constraints related to the arrangement of the points.

The algorithm consist in : 
1) Opening the image and configure it in grayscale image
2) Instantiation of the parameters (alpha, beta, gamma, and the number of points of the active contour)
3) Definition of the initial form of the snake (a circular form)
4) Calcul of the external energy
5) Update the coordinates through iterations until convergence
6) Displaying the convergence of the algorithm 
7) Saving the final result
