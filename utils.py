import cPickle
import numpy as np
import os
import random
import sys
sys.path.append("/Users/adarshjoseph/ImageCaptioning/ProjectImageCaptioning/coco/PythonAPI")
from pycocotools.coco import COCO
#sys.path.append("/Users/adarshjoseph/ImageCaptioning/ProjectImageCaptioning/coco/coco-caption")
from pycocoevalcap.eval import COCOEvalCap


def createModelDirs(LSTMobj):
     # create the main model directory:
    if not os.path.exists(LSTMobj.config.model_dir):
        os.makedirs(LSTMobj.config.model_dir)
    # create the dir where model weights will be saved during training:
    if not os.path.exists("%s/weights" % LSTMobj.config.model_dir):
        os.mkdir("%s/weights" % LSTMobj.config.model_dir)
    # create the dir where epoch losses will be saved during training:
    if not os.path.exists("%s/losses" % LSTMobj.config.model_dir):
        os.mkdir("%s/losses" % LSTMobj.config.model_dir)
    if not os.path.exists("%s/generated_captions" % LSTMobj.config.model_dir):
            os.mkdir("%s/generated_captions" % LSTMobj.config.model_dir)    

def getBatches(LSTMobj):
    """
    Objective: randomly shuffles all train caption ids and groups them into batches
    (of size model_obj.config.batch_size) where all captions in any given batch
    has the same length.
    """
    batchSize = LSTMobj.config.batch_size
    batchesOfCaptionIDs = []
    for captionLength in LSTMobj.captionLenghtVSnoOfCaptions:
        captionIDs = LSTMobj.trainCaptionLengthVSCaptionIDs[captionLength]
        #simply shuffle the ids
        random.shuffle(captionIDs)

        noOfCaptions =  LSTMobj.captionLenghtVSnoOfCaptions[captionLength]
        noOfBatches = int(noOfCaptions/batchSize)

        for i in range(noOfBatches):
            eachBatch = captionIDs[i*batchSize:(i+1)*batchSize]
            batchesOfCaptionIDs.append(eachBatch)
    random.shuffle(batchesOfCaptionIDs)
    return batchesOfCaptionIDs


def detokenizeCaption(tokenizedCaption, vocabulary):
    captionVec=[]
    for wordID in tokenizedCaption:
        word = vocabulary[wordID]
        captionVec.append(word)
    captionVec.pop(0)
    captionVec.pop()
    caption = " ".join(captionVec)
    return caption

def getBatchPHdata(LSTMobj,eachBatch):
    """
    - DOES: takes in a batch of caption ids, gets all corresponding data
    (img feature vectors and captions) and returns it in a format ready to be
    fed to the model (LSTM) placeholders in a feed_dict.
    """
    batchSize = LSTMobj.config.batch_size
    imgDim = LSTMobj.config.img_dim
    
    captionLen = len(LSTMobj.traincaptionIDvsCaption[eachBatch[0]])
    #initialize return datas ie caption,imgvector,target
    captions = np.zeros((batchSize,captionLen))
    imgVectors = np.zeros((batchSize,imgDim))
    target = -np.ones((batchSize,captionLen+1))

    for i in range(len(eachBatch)):
        captionID = eachBatch[i]
        imgID = LSTMobj.captionIDvsImgID[captionID]
        if imgID in LSTMobj.trainIDvsImageFeature:
            imgVector = LSTMobj.trainIDvsImageFeature[imgID]
        caption = LSTMobj.traincaptionIDvsCaption[captionID]
        captions[i]=caption
        imgVectors[i]=imgVector
        target[i,1:captionLen]=caption[1:]
        # captions=[eos,adarsh,is,eos]
        
        # taeget=[adarsh,is,eos,-1]
        
    return captions,imgVectors,target

def getTrainDataIterator(LSTMobj):
    batchesOfCaptionIDs = getBatches(LSTMobj)
    for eachBatch in batchesOfCaptionIDs:
        # get the batch's data in a format ready to be fed into the placeholders:
        captions, imgVectors, targets = getBatchPHdata(LSTMobj,eachBatch)
        yield(captions, imgVectors, targets)

def evaluateCaptions(captionFile):
    """
    DOES: computes the evaluation metrics BLEU-1 - BLEU4, CIDEr,
    METEOR and ROUGE_L for all captions in captions_file (generated on val or
    test imgs).
    """
    originalCaptionsFile = "coco/annotations/captions_val2014.json"
    coco = COCO(originalCaptionsFile)
    cocoRes = coco.loadRes(captionFile)#this will create a coco object with this captionsfile
    cocoEval = COCOEvalCap(coco,cocoRes)
    # set the imgs to be evaluated to the ones we have generated captions for:
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()
    # get the dict containing all computed metrics and metric scores:
    toRetDict = cocoEval.eval
    return toRetDict