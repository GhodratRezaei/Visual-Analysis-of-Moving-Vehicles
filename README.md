# **Visual-Analysis-of-Moving-Vehicles**


# **Table Of Content**
1. [Introduction](#my_first_title)
2. [Method](#my-second-title)
3. [Implementation](#my-third-title)
4. [Result and Conclusion](#my-fourth-title)
5. [Usage](#my-fifth-title)




## **Introduction**


The use of computer vision provides a very inexpensive way of detecting the speed of the cars while at the same time enables the system to monitor the car over speeding, making it easier and inexpensive to fine the offender as compared to the traditional method where the highway patrolling cars detect the over speeding cars and follow the particular car in order to fine it. In addition to this, the monitoring of flow of cars in and out of a certain region via counter, helps us to monitor and effectively predict the traffic flow situation in the city.
Here, relevant authorities can monitor the flow of cars and based on those readings, can recommend the commuters to adopt an alternative route, allowing the diversification of traffic flow and avoiding the traffic jams.

Common Techniques used for Object Detection are classical computer vision techniques and Deep learning approaches. In classical computer Vision, the background removal technique is used to detect object. The idea of background removal is not a modern one; instead, it has existed in the field since the dawn of modern technology. Hence, there have been numerous techniques for background removal in the domain of computer vision. Nevertheless, classical computer vision techniques are constantly being overthrown by modern, artificial intelligence-based ones by lodging more innovative ideas giving more precise and accurate results. In modern approaches toward object detection using Artificial Neural Network, objects can be detected by higher accuracy with respect to classical computer vision techniques. How ever we used both methods to detect object.
Using cross ratio invariant principle, along with the position of the two reference points, vanishing point of the image and the known real-world values, we can estimate the distance of the car and the marker in the real world. By using this change in distance from the subsequent



## **Method**


The Implementation of computer vision processes and techniques is devided into two main parts:
*  **1. Vehicle Detection**
*  **2. Speed Estimation**

Initially we extracted a video taken from a street camera of a 3-lane street where lane markers are visible, and vehicles are traveling at high speed. As part of the image preprocessing part, firstly, the video was converted into the set of images with frequency of 30 frames per second which gives use with a total of 150 images during the span of random 15 seconds analysis from the input video. The conversion of video to images was done using the tool libraries [FFMPEG](https://www.ffmpeg.org/) and [Magick](https://cran.r-project.org/web/packages/magick/vignettes/intro.html) on Linux. After converting the video into images, the image editing was carried out in which borders of the images were edited to keep the focus on the street and till the greatest of extent, subtract the other things happening in the video (like visibility of other streets, etc.) which can later be a cause of disturbance in our analysis leading to incorrect values.

### **1. Vehicle Detection**
#### **1.1 Classical Computer Vision Technique (Background Subtraction)**


In this Method, each image first was converted to gray scale. Having gray scale of each frame, each pixel value was subtracted from value of corresponding pixels in other images. In pixels with high values of differences between image frames, color of corresponding pixels leads to white color. Similarly for pixels with less difference of values, color will be more toward dark. After implementing such mask on images frame, we set threshold, upon which color can be white and lower values will be considered as dark pixel. This threshold value was defined 60 (pixel color value) in our methodologies and was set experimentally to cover larger areas which have meaningful discrepancies. One important that issue to be solved is to allocate to each vehicles a unique index which is identifiable among other object in different image frame. To solve this issue, greedy approach was implemented. Other innovative and new solution can be [marriage problem or maximum weighted bipartite matching](https://ieeexplore.ieee.org/document/6726915). Marriage problem is tracking each unique object in sequence of image frames. Challenge of object detection appears when we have object with similar size and shape in different image frames. To solve this problem, global optimization algorithm can be implemented over whole sequence of image frame. This global optimization algorithm has sub-optimization part in each image frame. To obtain stable tracking results, we propose a tracking method based on global optimization. Particularly, we first detect each individual vehicle in each image frame by background removal method, then formulate the multiple objects tracking problem as a combinatorial optimization problem over a pair of consecutive frames and solve the problem by the greedy approach algorithm. The target would be allocating a unique index to each unique vehicle in image frames which it is appeared, so that overall score of allocations is maximum over the image frames or equivalently total dissimilarity of objects over images gets minimum.

Two sequential image frames are shown in figures belows:




![Capture](https://user-images.githubusercontent.com/75788150/201474437-01ee0a57-a423-4e1d-821e-6e8ca7bc08ae.PNG)



corresponding graph algorithm and greed search are also shown in figures below:

![Capture](https://user-images.githubusercontent.com/75788150/201474491-39d4e76b-c879-4c07-a0e5-84648614ed29.PNG)


#### **1.2 Deep Learning Method**


Object detection using deep learning method is faster and accurate. In this project YOLO version 5 was implemented, which is now the most advanced object identification algorithm available. It is a
novel convolutional neural network (CNN) that detects objects in real-time with great accuracy. This approach uses a single neural network to process the entire picture, then separates it into parts and predicts bounding boxes and probabilities for each bounding box candidate (patch). These bounding boxes are weighted by the expected probability. So, we are dealing with both classification and regression problem; classification of object and assign to it a certain label which are preserved during whole image frame sequences and regression which certify the coordinates of the object (length and wide of bounding box and coordinates of upper right part of rectangular). Using a pre-trained model, we implement transfer learning and fine tuning to train whole multi-Layer-perceptions and potions of convolutional layers. To be mentioned that we did not use the label of objects, because it is not important in this project. To match certain objects in each image frame, greedy approach was implemented which was explained in previous method.











### **2. Speed Esstimation**

Once the vehicles is detected using the artificial intelligence, the phenomena of cross ratio is used in order to calculate the speed of the vehicle. The cross ratio of tuple points used the ratio of 4 points in image plane and 4 points in the planar view and use the ratio of those points which is always equal to -1 as shown in figure below:




![Capture](https://user-images.githubusercontent.com/75788150/201474606-631f6cec-fb5a-4f58-a513-d9d7856758ac.PNG)



   In the real-world planar view, point A represents the point at infinity, whereas points B and D represents the Forward Marker and Rear Marker respectively on the road and point C represents the real-world position of the car. Similarly, the corresponding values of these points are projected on the
2-dimensional image plane where the point A’ represents the vanishing point of the image, B’ and D’ represents the position of the marker and C’ represents the position of the center of the car in the image.

  From the 3-Dimensional real world prospective, we have the point A located at infinity whereas the distance between the front and the rear marker (B-D) is known to use which is equal to 52m. The distance we are calculating is the distance of the car from the front marker.
  
  On the other hand, in the image space, point A’ which represents the vanishing point of the image is calculated by combining the parallel line features of the vehicle moving direction and the vehicle feature. The pair of parallel lines gives us the vanishing point. Here we assume that since the vehicle is travelling in the straight direction within our region of study, the vanishing point would remain the same from image to image. However, to reduce the error in this assumption to a significant level, we calculated the vanishing point using the average of multiple vanishing points of the images we are studying. Points B’ and D’ which represents the front marker and the rear marker respectively on the image plane, would also remain the same as the position of the street and hence the lane markers remain the same with respect to the camera with which the video is being taken. The point C’, which represents the position of the vehicle on the image, is taken from the midpoint of the box position in which the vehicle is detected. The box of the image is extracted from the first part of the project related to the vehicle detection part. Using these values along with the relevant formulas given below, we were able to estimate the distance of the car from the front lane marker.
  
  
  Once we know the distance travelled by a particular car from one image to the following one, along with the time taken which is given by the FPS frequency with which we converted the video to the image, we were able to estimate the speed of the vehicle (equation below).
  
  
  (𝐴 𝐵 𝐶 𝐷)=|𝐴𝐶∗𝐵𝐷| / |𝐵𝐶∗𝐴𝐷|
  (𝐴 𝐵 𝐶 𝐷)=(𝐴′𝐵′𝐶′𝐷′)
  
  In our code, the following notations were used along with the final equation:
  
  
  𝑑=(|𝑉𝑎−𝑀𝑅|∗|𝑀𝐹−𝐶𝑐|) / (|𝑀𝐹−𝑀𝑅|∗|𝑉𝑎−𝐶𝑐|)∗52 𝑚
  
d = Distance of the car from the front marker (m)
Va = Vanishing point of the image scene
MR = Marker rear in the image
MF = Marker front in the image
Cc = Center of the car in the image

## **Implementation**


### **Assumption**

Due to unknown camera position with respect to the world reference plane in addition to unknown camera parameters, our evaluation has some limitations as few assumptions are considered in the methodology applied:

*  Since the vanishing point is extracted using the features (parallel lines) of both the lane marker and vehicle features, it goes without saying that we assume that the vehicles are running perfecting in direction of the street whereas the results of vehicles changing the lanes may be affected by this assumption

*  Since for calculating the speed of the vehicles, we are using the average distance travelled in a certain time segment, the speed of the vehicle is the average speed hence we assume that the vehicle is traveling at a constant velocity.  
*  Since the point at infinity of the image is taken to be fixed as the position of camera is fixed, the point cannot be completely accurate due to the limitation of quality of image and human error. However, to mitigate that effect, we did take the average of point at infinity of random images.



### **Conversion of Video to JPEG:**

   As very first step we converted a video of 30 minutes to images with frequency of 30 frames per second. The conversion of video to images was done using the tool libraries FFMPEG and Image Magick on Linux. The command below was executed on the Linux to convert the video to the images (figure 6). For the ease of experimentation, we are just considering the time slot of 15 seconds, hence the consecutive number of images we study are 150.
   
   After extracting the images, we can use them in our next parts of experimentation.
   
   
### **Vehicle Detection**

#### **1. Background Removal**


As explained in previous parts, background removal is implemented over gray scale image, and then a threshold of 60 was defined to separate the image into two parts: white part mismatch pixels and dark parts for matched pixels. Whole procedure was implemented using Python which is shown in figure below (figure 7). To track each unique vehicle along image frames, a greedy approach method was used which is a well approximated approach of marriage problem or maximum weighted bipartite matching.



Background Removal does not yield accurate detection of the objection. As shown in figure 7 and 8, all parts of vehicles are not detected and the detected parts are not consistent, meaning that the detection accuracy is not so high.

![Capture](https://user-images.githubusercontent.com/75788150/201477519-07b2ee77-15c5-42b5-a70b-d90832fff36e.PNG)


#### **Deep Learning (Yollo5)**

Using yolo for object Detection allows us to get the bounding box which is maximally matched with the object. As explained before, we use a pretrained model and did fine tunned it to the get the final network model for object detection. Input is image frame, and output is bounding box with defined circle and coordinates. 



Following figures two examples of the output from the object detection by YoloV5.

![Capture](https://user-images.githubusercontent.com/75788150/201477655-5377b530-8890-4050-9254-855ed51e8434.PNG)




![Capture](https://user-images.githubusercontent.com/75788150/201477683-07f2a12d-9065-49ec-84b8-b15a3f8e56b0.PNG)




### **Vehicle Counting**

Vehicle counting is done inside function Process_bounding_box in [main.py](https://github.com/GhodratRezaei/Visual-Analysis-of-Moving-Vehicles/blob/main/main.py) file.




![Capture](https://user-images.githubusercontent.com/75788150/201477783-51bdad51-9607-43f5-a2d4-a051f8e4cd8a.PNG)


   
### **Average of Vanishing Points:**



In order minimize the error of the vanishing point we estimated for our experimentation, we calculated vanishing points of random images and then took the average of it. This way any human error, or error due to selection of the parallel lines is minimized.
In figure below, we constructed a pair of parallel lines, one using the vehicle feature and the other using the street feature.


![Capture](https://user-images.githubusercontent.com/75788150/201477843-8f13a7bd-2f5e-426e-b15a-82cf36797a4c.PNG)

Following the construction of parallel lines, we can estimate the vanishing point of the image with the help of the intersection point of these two parallel lines. In the figure below, we can see that the vanishing point is located at the top of the image, that is in the direction of vehicle movement. In addition to this, here we can also see the location of the markers located on the street which is taken as the reference point. The front marker is labelled as ‘b’ and the rear marker is labelled as ‘a’. The distance between these two points is 52m which was calculated using the standard distance of the markers. The two marker points and the vanishing point is considered fixed for all the images; hence it is given as input to the system.







![Capture](https://user-images.githubusercontent.com/75788150/201477870-bec2dd4d-f9ea-4909-b0c0-388de16e731f.PNG)



### **Speed Estimation**

Using the vehicle detection ability, we can track the vehicles from one image to the subsequent ones. This feature, combined with the cross ratio of tuple points, where the four points of the image plane is compared with the real-world points (details explained in the methodology section above), we can estimate the distance of the detected vehicle and the reference markers. Since a particular vehicle is detected until it crosses the second marker on the street, we can get the time taken by the vehicle to travel the specified distance (between the two labelled markers). This allows us to estimate the average speed of that vehicle. 


In the following Figures, the average speed of each car is visible which is calculated using the time taken by the car to cross the two markers whose distance is pre-defined in the previous part.




![Capture](https://user-images.githubusercontent.com/75788150/201477915-f68c0b36-2eb6-4684-9236-be151bfabd6e.PNG)



As can be seen in figure above, average speed of each car has is shown in red rectangle and the unit is m/s. The speed values make sense in real-world. As an example, speed of 20 (m/s) is equal to 72(km/h) and in highway it is standard speed for car.





## **Result and Conclusion**


The results of the average speed estimated of the cars along with the counter value, is somehow acceptable. Furthermore, while implementing the cross-ratio algorithm, the distance between the front marker and the card was also checked with the estimated value of the real-world distance using the values of distance between the markers, which were also acceptable. This concludes that the methodologies implemented in this experimentation which includes the conversion of the video to images, followed by vehicle detection, estimation of vanishing points and tracking of the vehicle using the greedy approach some what gives an acceptable set of results.
However, due to the initial assumptions we considered, there may be some errors, specially for cars changing lane in between the region of study as it would no longer hold on to the assumption that the car is parallel to the street in the real world. Furthermore, there might be cars which are accelerating or decelerating in the region between two markers, resulting in incorrect average speed estimation.
Improvements can be brought to the experiment in many ways. Firstly, a better-quality camera with higher resolution would enable better estimation of the vanishing point as it would reduce the error while drawing the parallel lines. Secondly, we can use an alternative way of vehicle tracking known as marriage problem instead of greedy approach which would be more intrinsic but will give better results. Thirdly, smaller, and multiple segments of regions can be chosen to study instead of just one as in our case. This would increase the mathematical complexity of the program but would give a better estimation of speed specially for vehicle who are accelerating / decelerating while driving. Lastly, the camera intrinsic parameters along with the position of camera with respect to world coordinates can be easily calculated in the real world for situations like these. Once they are calculated, the camera geometry can be put to use which would also give a better result in terms of image projection and rectification which then can be further processed to estimate our desired parameters.



## **Usage**
For Running the code, following code lines should be ran.


`pip3 -r requirements.txt`
`python3 main.py`



