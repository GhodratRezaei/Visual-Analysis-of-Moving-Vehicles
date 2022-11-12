# **Visual-Analysis-of-Moving-Vehicles**


# **Table Of Content**
1. [Introduction](#my_first_title)
2. [Method](#my-second-title)
3. [Result](#my-third-title)
4. [Usage](#my-fourth-title)




## **Introduction**


The use of computer vision provides a very inexpensive way of detecting the speed of the cars while at the same time enables the system to monitor the car over speeding, making it easier and inexpensive to fine the offender as compared to the traditional method where the highway patrolling cars detect the over speeding cars and follow the particular car in order to fine it. In addition to this, the monitoring of flow of cars in and out of a certain region via counter, helps us to monitor and effectively predict the traffic flow situation in the city.
Here, relevant authorities can monitor the flow of cars and based on those readings, can recommend the commuters to adopt an alternative route, allowing the diversification of traffic flow and avoiding the traffic jams.

Common Techniques used for Object Detection are classical computer vision techniques and Deep learning approaches. In classical computer Vision, the background removal technique is used to detect object. The idea of background removal is not a modern one; instead, it has existed in the field since the dawn of modern technology. Hence, there have been numerous techniques for background removal in the domain of computer vision. Nevertheless, classical computer vision techniques are constantly being overthrown by modern, artificial intelligence-based ones by lodging more innovative ideas giving more precise and accurate results. In modern approaches toward object detection using Artificial Neural Network, objects can be detected by higher accuracy with respect to classical computer vision techniques. How ever we used both methods to detect object.
Using cross ratio invariant principle, along with the position of the two reference points, vanishing point of the image and the known real-world values, we can estimate the distance of the car and the marker in the real world. By using this change in distance from the subsequent



## **Method**


The Implementation of computer vision processes and techniques is devided into two main parts:
*  **1. Vehicle Detection**
*  **2. Speed Estimation**

Initially we extracted a video taken from a street camera of a 3-lane street where lane markers are visible, and vehicles are traveling at high speed. As part of the image preprocessing part, firstly, the video was converted into the set of images with frequency of 30 frames per second which gives use with a total of 150 images during the span of random 15 seconds analysis from the input video. The conversion of video to images was done using the tool libraries FFMPEG and Magick on Linux. After converting the video into images, the image editing was carried out in which borders of the images were edited to keep the focus on the street and till the greatest of extent, subtract the other things happening in the video (like visibility of other streets, etc.) which can later be a cause of disturbance in our analysis leading to incorrect values.

### **Vehicle Detection**
#### **Classical Computer Vision Technique (Background Subtraction)


In this Method, each image first was converted to gray scale. Having gray scale of each frame, each pixel value was subtracted from value of corresponding pixels in other images. In pixels with high values of differences between image frames, color of corresponding pixels leads to white color. Similarly for pixels with less difference of values, color will be more toward dark. After implementing such mask on images frame, we set threshold, upon which color can be white and lower values will be considered as dark pixel. This threshold value was defined 60 (pixel color value) in our methodologies and was set experimentally to cover larger areas which have meaningful discrepancies. One important that issue to be solved is to allocate to each vehicles a unique index which is identifiable among other object in different image frame. To solve this issue, greedy approach was implemented. Other innovative and new solution can be [marriage problem or maximum weighted bipartite matching](https://ieeexplore.ieee.org/document/6726915). Marriage problem is tracking each unique object in sequence of image frames. Challenge of object detection appears when we have object with similar size and shape in different image frames. To solve this problem, global optimization algorithm can be implemented over whole sequence of image frame. This global optimization algorithm has sub-optimization part in each image frame. To obtain stable tracking results, we propose a tracking method based on global optimization. Particularly, we first detect each individual vehicle in each image frame by background removal method, then formulate the multiple objects tracking problem as a combinatorial optimization problem over a pair of consecutive frames and solve the problem by the greedy approach algorithm. The target would be allocating a unique index to each unique vehicle in image frames which it is appeared, so that overall score of allocations is maximum over the image frames or equivalently total dissimilarity of objects over images gets minimum.





