import numpy as np
import os
import cPickle
import re

#################
#refer:  https://www.fir3net.com/Programming/Python/how-do-you-import-a-python-module-from-another-folder.html
#for the below from a.b import c
import sys
sys.path.append("/Users/adarshjoseph/ImageCaptioning/ProjectImageCaptioning/coco/PythonAPI")
from pycocotools.coco import COCO
#################


test_imgIDs = cPickle.load(open("coco/data/testImgIDs"))
val_imgIDs = cPickle.load(open("coco/data/valImgIDs"))
##########
train_imgIDs = cPickle.load(open("coco/data/trainImgIDs"))
##########

captionsDIR = "coco/annotations/"
dataDir = "coco/data/"

def load_captions(datatSetType):
    
    ''''
    we need to load both train captions as well as val captions , we will be passing the dataset
    ie either train or val and by the variable name passed we will load the corresponding captions
    '''
    if datatSetType == "train":
        captionsFile = "coco/annotations/captions_train2014.json"
    else:
        captionsFile = "coco/annotations/captions_val2014.json"

    cocoObj = COCO(captionsFile)
    imgIDS = cocoObj.getImgIds()

    for img_id in imgIDS:
        
        # get the ids of all captions for the image using the img_id:
        captionIDs = cocoObj.getAnnIds(imgIds=img_id)
        captionObjs = cocoObj.loadAnns(captionIDs)

        for captionObj in captionObjs:
            captionID = captionObj["id"]
            '''
            we will store the captioID vs img ID
            here we need to note that each captioID is unique 
            and five captions are there for one img id
            '''
            captionIDvsimgID[captionID] = img_id
            caption = captionObj["caption"]
            caption = caption.lower()
            caption = re.sub("[^a-z0-9 ]+", "", caption)
            caption = re.sub("  ", " ", caption)
            caption = caption.split(" ")


            if img_id in val_imgIDs:
                valcaptionIDvsCaption[captionID] = caption
            elif img_id in train_imgIDs:
                traincaptionIDvsCaption[captionID] = caption

    print "!!!!!Caption Loading Done!!!!!"        

def countWords(traincaptionIDvsCaption):   
    word_counts = {}
    for captionID in traincaptionIDvsCaption:
        caption = traincaptionIDvsCaption[captionID]
        for word in caption:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] +=1 
    return word_counts     
###################################

captionIDvsimgID = {}
valcaptionIDvsCaption = {}
traincaptionIDvsCaption = {}


load_captions("train")
load_captions("val")
print len(valcaptionIDvsCaption)
print len(traincaptionIDvsCaption)

cPickle.dump(captionIDvsimgID,open(os.path.join(dataDir,"captionIDvsimgID"),"wb"))

pretrained_words = []
with open(os.path.join(captionsDIR, "glove.6B.300d.txt")) as file:
    for line in file:
        line_elements = line.split(" ")
        word = line_elements[0]
        pretrained_words.append(word)

word_counts = countWords(traincaptionIDvsCaption)
print "!!!!!WORD COUNT TOOK!!!"
# create a vocabulary of all words that appear 5 or more times in the
# training set
vocabulary = []
for word in word_counts:
    word_count = word_counts[word]
    if word_count >= 5 and str(word) in pretrained_words:
        vocabulary.append(word)
print "VOCABULARY CREATED"+str(len(vocabulary))
# replace all words in train that are not in the vocabulary with an
# <UNK> token AND prepend each caption with an <SOS> token AND append
# each caption with an <EOS> token:
#eg : <SOS> she is throwing a ball <EOS>  --> we will be adding the tokens <EOS> and <SOS>
i=0
for captionID in traincaptionIDvsCaption:
    if(i%500==0):
        print "step"+str(i)
    i=i+1
    caption = traincaptionIDvsCaption[captionID]
    lenghtOFCaption = len(caption)
    for indexOfWord in range(lenghtOFCaption):
        word = caption[indexOfWord]
        if word not in vocabulary:
            caption[indexOfWord] = "<UNK>"
    #inseetrting <sos> in front of caption and <eos> in back of caption
    caption.insert(0,"<SOS>")
    caption.append("<EOS>")
# add "<SOS>", "<UNK>" and "<EOS>" to the vocabulary:
vocabulary.insert(0, "<EOS>")
vocabulary.insert(0, "<UNK>")
vocabulary.insert(0, "<SOS>")
cPickle.dump(vocabulary,open(os.path.join(dataDir,"vocabulary"),"wb"))

# prepend each caption in val with an <SOS> token AND append each
# caption with an <EOS> token:

for captionID in valcaptionIDvsCaption:
    caption = valcaptionIDvsCaption[captionID]
    #inseetrting <sos> in front of caption and <eos> in back of caption
    caption.insert(0,"<SOS>")
    caption.append("<EOS>")
#same for test also
for captionID in testcaptionIDvsCaption:
    caption = testcaptionIDvsCaption[captionID]
    #inseetrting <sos> in front of caption and <eos> in back of caption
    caption.insert(0,"<SOS>")
    caption.append("<EOS>")
print "!!!!!!!!CAPTION APPENDING <EOS>, <SOS>!!!!!!!"

#tokenize captions of training images
#tokenization is breaking into a set of words and then here we are transforming that splited words
#into np array for deep learning and np array is crated from the corresponding indx of the word in
#vocabularys

for captionID in traincaptionIDvsCaption:
    caption = traincaptionIDvsCaption[captionID]
    tokenized_caption = []
    for word in caption:
        indexOfWord = vocabulary.index(word)
        tokenized_caption.append(indexOfWord)
    tokenized_caption = np.array(tokenized_caption)
    traincaptionIDvsCaption[captionID] = tokenized_caption
print "!!!!TOKENIZED!!!!!"
cPickle.dump(traincaptionIDvsCaption, open(os.path.join(dataDir,"traincaptionIDvsCaption"), "wb"))
cPickle.dump(valcaptionIDvsCaption, open(os.path.join(dataDir,"valcaptionIDvsCaption"), "wb"))

#next : mapping betwn the length and captions having that length
#say three captions with captionID {21,32,34} are having length 8
#then they will be stored in an array a[8] = {21,32,34}

trainCaptionLengthVSCaptionIDs = {}
for captionID in traincaptionIDvsCaption:
    caption = traincaptionIDvsCaption[captionID]
    captionLength = len(caption)
    if captionLength not in trainCaptionLengthVSCaptionIDs:
        temp = []
        temp.append(captionID)
        trainCaptionLengthVSCaptionIDs[captionLength]=temp
    else:
        trainCaptionLengthVSCaptionIDs[captionLength].append(captionID)
print trainCaptionLengthVSCaptionIDs

# map each train caption length to the number of captions of that length:
trainCapLenVScaptionCount = {}
for captionLength in trainCaptionLengthVSCaptionIDs:
    captionIDs = trainCaptionLengthVSCaptionIDs[captionLength]
    captionCount = len(captionIDs)
    trainCapLenVScaptionCount[captionLength]=captionCount


###Save evrything to local disk
cPickle.dump(trainCapLenVScaptionCount,open(os.path.join("coco/data/","trainCapLenVScaptionCount"), "wb"))
cPickle.dump(trainCaptionLengthVSCaptionIDs,open(os.path.join("coco/data/","trainCaptionLengthVSCaptionIDs"), "wb"))

