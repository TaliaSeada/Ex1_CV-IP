# Computer Vision and Image Processing 
* Python version: 3.9.5 </br>
* Platform: PyCharm </br>

### Submission files:
1. ex1_utils.py
2. ex1_main.py (given file)
3. gamma.py
4. testImg1.jpg (test image 1)
5. testImg2.jpg (test image 2)
6. README.md

### Descriptions and Functions:
1. __ex1_utils.py__ - In this file we wrote the main functions for this assignment.</br>
&emsp;~ imReadAndConvert - reads a given image file and converts it into a given representation.</br>
&emsp;~ imDisplay - utilizes imReadAndConvert to display a given image file in a given representation.</br>
&emsp;~ transformRGB2YIQ - transform an RGB image into the YIQ color space.</br>
&emsp;~ transformYIQ2RGB - transform an YIQ image into the RBG color space.</br>
&emsp;~ histogramEqualize - performs histogram equalization of a given grayscale or RGB image.</br>
&emsp;~ quantizeImage - performs optimal quantization of a given grayscale or RGB image.</br>
&emsp;~ _QuanMain - help function that responsible to the main quantization </br>

2. __gamma.py__ - In this file represent the GUI for gamma correction.</br>
&emsp;~ gammaDisplay - performs gamma correction on an image with a given gamma.</br>
&emsp;~ on_trackbar - follows the values of the trackbar we built.</br>

# Python’s basic syntax and some of its image processing facilities. </br>
This exercise covers:</br>
1. Loading grayscale and RGB image representations. 
2. Displaying figures and images.
3. Transforming RGB color images back and forth from the YIQ color space.
4. Performing intensity transformations: histogram equalization.
5. Performing optimal quantization

### Loading grayscale and RGB image representations and Displaying figures and images:
![image](https://user-images.githubusercontent.com/78349342/159690913-d0daf730-badd-47de-a1eb-a6178a1b31f7.png)

### Transforming RGB color images back and forth from the YIQ color space:
![image](https://user-images.githubusercontent.com/78349342/159686790-e9acff83-1d6f-47e8-a63c-88e093720922.png)

### Performing intensity transformations- histogram equalization:
<b> Histogram equalization </b> is a method in image processing of contrast adjustment using the image's histogram. - https://en.wikipedia.org/wiki/Histogram_equalization <br/>
  
Gray Scale: </br>
![image](https://user-images.githubusercontent.com/78349342/159687764-f7618bde-26b0-46fa-9351-f737fa0e694f.png) </br>
RGB: </br>
![image](https://user-images.githubusercontent.com/78349342/159688213-4568614b-d8c4-4b55-98e8-59b000adf73d.png) </br>

### Performing optimal quantization:
<b> Quantization </b>  is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers). - 
https://en.wikipedia.org/wiki/Quantization <br/>
Gray Scale: </br>

![image](https://user-images.githubusercontent.com/78349342/159689796-cdab485b-061f-42dd-a9c7-def7a5cb78eb.png) </br>
RGB: </br>
![image](https://user-images.githubusercontent.com/78349342/159690250-fa87598d-afc9-46b6-8a3d-4eaa85816d08.png)

### Gamma correction:
1. https://en.wikipedia.org/wiki/Gamma_correction
2. https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
![image](https://user-images.githubusercontent.com/78349342/159947656-a92742cb-8711-48df-81ce-21202ab091ac.png) </br>
__Examples:__ </br>
![image](https://user-images.githubusercontent.com/78349342/159948666-be396bc4-b33f-4d3b-814f-ee1cd40178a4.png) </br>





