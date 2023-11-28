<h1> image-to-sketch </h1>
<p>it's basic machine learning project which is used to convert any image into relative sketch type image.</p>
<p>it was build using <i><b>python</b></i> and for operations on images <i><b>OpenCV</b></i> Framework was used. for color quantization part <i><b>K-Means clustering</b></i> Machine learning algorithm was used</p>
<p>steps involed in this process:</p>
<ul>
  1. read image<br>
  2. grayscale conversion<br>
  3. edge masking<br> 
  4. color quantization [using K-Means]<br>
  5. noise reduction using Bilateral filter<br> 
  6. combining edge mask and reconstructed color image<br>
</ul>
