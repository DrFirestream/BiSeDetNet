# BiSeDetNet
Real-time detection via segmentation

Based on https://github.com/Blaizzy/BiSeNet-Implementation

Changed backbone network to ResNet and upsampling route to Conv2DTranspose route for better accuracy 

![Alt text](temp.png?raw=true "Original image")  ![Alt text](segm_2.png?raw=true "Segmented image")

Detects small cars much better than Yolo v3
![Alt text](cars2.gif?raw=true "Detection of cars")

Works also for very low quality video
![Alt text](cars.gif?raw=true "Detection of cars")
