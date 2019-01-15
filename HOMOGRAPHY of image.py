import numpy as np
import cv2
import matplotlib.pyplot as plt

#Taking input image1 for reference to image2
image1=cv2.imread(r"D:\project\OMR\images\new omr\new1.png")
image2=cv2.imread(r"D:\project\OMR\images\new omr\new11.png")
#image22=cv2.imread(r"D:\project\OMR\images\new omr\new11.png")
cv2.imshow("image1",image1)
cv2.imshow("image2",image2)
#Converting this image into Gray Scale
gray1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

#SIFT used for finding keypoints we can also use SURF
sift=cv2.xfeatures2d.SIFT_create()

#kp holds the data of any particular keypoint and des holds the data of how to describe that key
#kp be a list of keypnts and des is numpy array of shape No._of_Keypoints×128.
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)

image1=cv2.drawKeypoints(image1,kp1,None)
#cv2.imshow("imgage",image1);
image2=cv2.drawKeypoints(image2,kp2,None)
#cv2.imshow("image2",image2);


#----------- tried with brute force---------
#bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#matches=bf.match(des1,des2)
#matches=sorted(matches,key=lambda x:x.distance)
#image3=cv2.drawMatches(image1,kp1,image2,kp2,mathces[:10],None,flags=2)
#plt.imshow(image3)
#plt.show()

index_params=dict(algorithm=0, trees=5)
search_params=dict()
flann =cv2.FlannBasedMatcher(index_params,search_params)

#here matches will hold the four values regarding for two separate descriptors as
#...values are....
#m.distance   Distance between descriptors. The lower, the better it is.
#m.imgIdx   Index of the descriptor in train descriptors
#m.queryIdx  Index of the descriptor in query descriptors
#m.trainIdx  Index of the train image.
#..we are talking about two images
matches=flann.knnMatch(des1,des2,k=2)

#m.distance   Distance between descriptors. The lower, the better it is.
#m.imgIdx   Index of the descriptor in train descriptors
#m.queryIdx  Index of the descriptor in query descriptors
#m.trainIdx  Index of the train image.

#we create mask as we want to draw on;y good matches
good_points=[]

#ratio test as per lowe's paper

for m,n in matches:
    #print(m.distance,n.distance);
    if m.distance<0.6*n.distance:
        good_points.append(m)
image3=cv2.drawMatches(image1,kp1,image2,kp2,good_points,image2)

cv2.imshow("good match image",image3)
img=cv2.resize(image3,(1250,600))

##----------homography---------------
##queryIdx "query index" gives us the position of points of the query image
if len(good_points)>10:
    query_pts=np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
    train_pts=np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
#we got the matrix to show the image in his prespective
    matrix,mask=cv2.findHomography(query_pts,train_pts,cv2.RANSAC,5.0)
    matches_mask=mask.ravel().tolist()  #ravel() help us to convert 2d array in 1d array
    #print(image1)
##-----perspective transfoem---- watch video
    h,w,s=image1.shape   #passing height and width of original image so that
                #second image can adopt the image 
    pts=np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,matrix)
##np.int32(dst)  we use this because we want to specify point which can not be in decimal
##True   we want ot close line polyline true
##(255,0,0)  we use this for color of bloundary
    homography=cv2.polylines(image2,[np.int32(dst)],True,(255,0,0),3)
    cv2.imshow("homography",homography)
else:
    cv2.imshow("homography2",image2)
print(image2.shape)
