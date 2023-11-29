# image-to-sketch
it's basic machine learning project which is used to convert any image into relative sketch type image.

it was build using ___python___ and for operations on images ___OpenCV___ Framework was used. for color quantization part ___K-Means clustering___ Machine learning algorithm was used

steps involed in this process:
1. [read image [using OpenCV]](#read-image-and-grayscale-convertion)
2. [grayscale conversion](#read-image-and-grayscale-convertion)
3. [edge masking](#color-quantization)
4. [color quantization [using K Means]](#edge-masking)
5. [noise reduction using Bilateral filter](#noise-reduction)
6. [combining edge mask and reconstructed color image](#applying-mask-on-colored-image)
7. [write resultant image in output folder](#end-results)

## read image and grayscale convertion 
firstly we read an input image using OpenCV and convert into __Grayscale image__.

![orginal to gray image](code/assets/process/oggray.png)

[Note: OpenCV uses BGR color schem by default] 
## edge masking

![gray to edge image](code/assets/process/grayedge.png)

## color quantization 

![orignal to recolored image](code/assets/process/ogrecolored.png)

## noise reduction 

![bi-laterl filtered image](code/assets/process/recoloredfiltered.png)

## applying mask on colored image

![masking on image](code/assets/process/maskingoutput.png)

## end results

![output image w.r.t orignal image](code/assets/process/endoutput.png)

### sample Examples  
