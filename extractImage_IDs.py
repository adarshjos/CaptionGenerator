"""
Pre-Requisite : dataset, ie test and val are placed in the proper hierarchial order
ProjectTryOut-->coco-->images-->test
ProjectTryOut-->coco-->images-->val

Result: Two files are created with contents:imageID
"""
import os
import numpy as np
import cPickle
import random

#path where the images are located
test_image_folder = "coco/images/test2014_1/"
val_image_folder = "coco/images/val2014_1/"
train_image_folder = "coco/images/train2014_1/"


#a list is created containing the paths of each image

def fetchImagePaths(folderName):
    imgPaths=[]
    for root, dirs, files in os.walk(folderName):
        for file_name in files:
            if ".jpg" in file_name:
                #print(os.path.join(root, name))
                imgPaths.append(os.path.join(root, file_name))
    return imgPaths

def getImgIDs(imgPaths):
    imgIDs=np.array([])
    for imgPath in imgPaths:
        img_name = imgPath.split("/")[3]
        img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
        img_id = int(img_id)
        imgIDs = np.append(imgIDs,img_id)
    return imgIDs



test_image_paths = fetchImagePaths(test_image_folder)
np.random.shuffle(test_image_paths)
test_image_paths=test_image_paths[1:datasetSplitSize]

train_image_paths = fetchImagePaths(train_image_folder)
np.random.shuffle(train_image_paths)
train_image_paths=train_image_paths[1:75000]

val_image_paths = fetchImagePaths(val_image_folder)
np.random.shuffle(val_image_paths)
val_image_paths=val_image_paths[1:5000]

print train_image_paths[0]

train_image_ids = getImgIDs(train_image_paths)
test_image_ids = getImgIDs(test_image_paths)
val_image_ids = getImgIDs(val_image_paths)


#store these list we created into the local disk inside coco/data/
cPickle.dump(val_image_ids,open(os.path.join("coco/data/","valImgIDs"),"wb"))
cPickle.dump(val_image_paths,open(os.path.join("coco/data/","val_image_paths"),"wb"))
print "<---------VALIDATION image IDs Saved in Disk----count:"+str(len(val_image_ids))+"-->"

cPickle.dump(test_image_ids,open(os.path.join("coco/data/","testImgIDs"),"wb"))
print "<---------TEST image IDs Saved in Disk---------count:"+str(len(test_image_ids))+"-->"

# #########
# #for testing not sure ,required in future
cPickle.dump(train_image_ids,open(os.path.join("coco/data/","trainImgIDs"),"wb"))
cPickle.dump(train_image_paths,open(os.path.join("coco/data/","train_image_paths"),"wb"))
print "<---------TRAIN image IDs Saved in Disk--------count:"+str(len(train_image_ids))+"-->"

# ###########