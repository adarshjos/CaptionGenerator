'''
input:  
        1. coco/images/train2014_1
        2. coco/images/val2014_1
'''

import os
import numpy as np
import tensorflow as tf
import cPickle
from tensorflow.python.platform import gfile


dataDir = "coco/data/"
modelDir = os.path.abspath("inception")
pathToSavedModel = os.path.join(modelDir,"classify_image_graph_def.pb")

#this can also be written as pathttisavedmodel="inception/classify_image_graph_def.pb"

def extractfeatureInception(pathofImages,testing=False):

    #loading pretrained CNN#
    with gfile.FastGFile(pathToSavedModel,"rb") as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        g_in=tf.import_graph_def(graph_def, name="")
    ##########################
    print pathofImages
    imgIDvsImageFeatureVector = {}
    with tf.Session(graph=g_in) as sess:
        #print sess.graph.get_operations()
        i=0
        second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")
        for imgPath in pathofImages:
            print imgPath
            img_data = gfile.FastGFile(imgPath, "rb").read()
            try:
                # get the img's corresponding feature vector:
                feature_vector = sess.run(second_to_last_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
            except:
                print "JPEG error for:"
                print img_path
                print "******************"
            else:
                i=i+1
                if(i%200==0):
                    print "ImageVector Extraction done for "+str(i)
                # # flatten the features to an np.array:
                feature_vector = np.squeeze(feature_vector)
                if not testing:
                    # get the image id:
                    imgName = imgPath.split("/")[3]
                    img_id = imgName.split("_")[2].split(".")[0].lstrip("0")
                    img_id = int(img_id)
                else:
                    # we're only extracting features for one img
                    # set the img id to 0:
                    img_id = 0
                imgIDvsImageFeatureVector[img_id]=feature_vector
                
        return imgIDvsImageFeatureVector

def fetchImagePaths(folderName):
    imgPaths=[]
    for root, dirs, files in os.walk(folderName):
        for file_name in files:
            if ".jpg" in file_name:
                #print(os.path.join(root, name))
                imgPaths.append(os.path.join(root, file_name))
    return imgPaths

def main():
    '''
    the following steps are to load each image in val/train and extract those imaghe features 
    using the inception model 
    '''
    val_image_paths = cPickle.load(open("coco/data/val_image_paths"))
    train_image_paths = cPickle.load(open("coco/data/train_image_paths"))

    valIDvsImageFeatureVector = extractfeatureInception(val_image_paths)
    cPickle.dump(valIDvsImageFeatureVector,open(os.path.join(dataDir,"valIDvsImageFeature"),"wb"))
    print ("#####Validation Image Feature Extraction done####")
    

    trainIDvsImageFeatureVector = extractfeatureInception(train_image_paths)
    cPickle.dump(trainIDvsImageFeatureVector,open(os.path.join(dataDir,"trainIDvsImageFeature"),"wb"))
    print ("#####Train Image Feature Extraction done####")

if __name__ =='__main__':
    main()