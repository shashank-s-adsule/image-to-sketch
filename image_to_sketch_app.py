import numpy as np
import cv2
from flask import Flask, render_template, request, url_for, redirect

app=Flask(__name__, template_folder="./templates/")

def color_quantize(img,K=8):
    data=np.float32(img).reshape((-1,3))
    
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    file=request.files.get("image")
    
    print(file)



if __name__=="__main__":
    app.run(debug=True)