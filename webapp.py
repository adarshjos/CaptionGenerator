import os
from flask import Flask,render_template,request,redirect
import sys
import cPickle
sys.path.append("/Users/adarshjoseph/ImageCaptioning/ProjectImageCaptioning")
from imageFeatureExtraction import extractfeatureInception
from captionGenerator import generateCap


UPLOAD_FOLDER = "/Users/adarshjoseph/ImageCaptioning/ProjectImageCaptioning/static/testImg"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/',methods=["GET", "POST"])
def index():
   return render_template("index.html")

@app.route('/generateCaption',methods=["GET", "POST"])
def generateCaption():
   if request.method == "POST":
      image = request.files["image"]
      image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))
      imgPath=[]
      imgPath.append(str(os.path.join(app.config["UPLOAD_FOLDER"], image.filename)))
      imgIDvsImageFeatureVector = extractfeatureInception(imgPath,testing=True)
      genCaption = generateCap(imgIDvsImageFeatureVector[0])
      return render_template("generateCaption.html",imgName="testImg/"+str(image.filename),caption=genCaption)


if __name__ == '__main__':   
   app.run(debug=True)