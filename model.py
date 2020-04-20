"""
https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

"""
import numpy as np
import tensorflow as tf
import cPickle
import os
import time
from utils import *
import json
import random

class LSTM_config(object):
    def __init__(self):
        self.embed_dim = 300
        self.dropout = 0.75
        self.hidden_dim = 400
        self.lr = 0.001
        self.img_dim = 2048
        self.batch_size = 256
        self.vocab_size = 8323#get the size
        self.no_of_layers = 1
        self.max_caption_length = 40
        self.max_no_of_epochs = 20
        self.model_dir = "models/LSTM"


class LSTM_model(object):

    def __init__(self,config,gloveEmbeddings,mode="training"):
        self.config = config
        self.gloveEmbeddings = gloveEmbeddings
        if mode is not "testing":
            self.loadUtilitiesData()
            createModelDirs(self)
        #create the placeholders for the tensorflow graph
        self.addPlaceholders()
        self.combineInputs()
        self.addLayers()
        if mode is not "testing":
            self.lossFunction()
            self.add_training_op()

    def loadUtilitiesData(self):
        self.vocabulary = cPickle.load(open("coco/data/vocabulary"))
        ###### load data to map from caption id to img feature vector:
        self.captionIDvsImgID = cPickle.load(open("coco/data/captionIDvsimgID"))
        self.trainIDvsImageFeature = cPickle.load(open("coco/data/trainIDvsImageFeature"))
        self.traincaptionIDvsCaption =cPickle.load(open("coco/data/traincaptionIDvsCaption"))
        self.trainCaptionLengthVSCaptionIDs = cPickle.load(open("coco/data/trainCaptionLengthVSCaptionIDs"))            
        self.captionLenghtVSnoOfCaptions = cPickle.load(open("coco/data/trainCapLenVScaptionCount"))
        print "Utility Data Loading COMPLETED!!"
        
    def addPlaceholders(self):
        '''
        adds placeholders for captions, imgs, labels and keep_prob to
        the computational graph. These placeholders will be fed with the actual data
        at the time of training
        '''
        self.captionPH = tf.placeholder(tf.int32,shape=[None,None],name="captionPH")
        #this placeholder's total size is the batchsize and row i of the captionPH
        #is the tokenized caption for eg image i in the batch
        self.imgPH = tf.placeholder(tf.float32,shape=[None,self.config.img_dim],name="imgPH")

        #next placeholder is for the target or output
        self.targetPH = tf.placeholder(tf.int32,shape=[None,None],name="targetPH")
         # ([batch_size, caption_length+1])
        self.dropoutPH = tf.placeholder(tf.float32,name="dropoutPH")
        print "addPlaceholders!!"
        
    def constructFeedDict(self,captionsBatch,imgsBatch,targetBatch=None,dropOut=1):
        feed_dict={}
        feed_dict[self.captionPH] = captionsBatch
        feed_dict[self.imgPH] = imgsBatch
        feed_dict[self.dropoutPH] = dropOut
        if targetBatch is not None:
            #in caotion generation for testing we wont have anu target 
            feed_dict[self.targetPH] = targetBatch
        return feed_dict
        print "constructFeedDict!!"

    def combineInputs(self):
        with tf.variable_scope("captionEmbVec"):
            wrdEmbedding = tf.get_variable("wordEmbedding",initializer=self.gloveEmbeddings)
            captionInput = tf.nn.embedding_lookup(wrdEmbedding,self.captionPH)
            # print "########"
            # print tf.shape(captionInput)
        with tf.variable_scope("imgVector"):
            weightsImg = tf.get_variable("weightsImg",shape=[self.config.img_dim, self.config.embed_dim],initializer=tf.contrib.layers.xavier_initializer())
            biasImg = tf.get_variable("biasImg", shape=[1,self.config.embed_dim],initializer=tf.constant_initializer(0))
            imgInput = tf.nn.sigmoid(tf.matmul(self.imgPH,weightsImg)+biasImg)
            # print "#################"
            # print tf.shape(imgInput)
            imgInput = tf.expand_dims(imgInput,1)
            # print "#################"
            # print tf.shape(imgInput)
        self.nodeFinalInput = tf.concat([imgInput,captionInput],1)
        print "combineInputs!!"

    def addLayers(self):
        # create an LSTM cell:
        LSTMnode = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim,name="basicLSTMell")
        # apply dropout to the LSTM cell:
        LSTMnode = tf.nn.rnn_cell.DropoutWrapper(LSTMnode,
                    input_keep_prob=self.dropoutPH,
                    output_keep_prob=self.dropoutPH)
        stackedLSTMnode= tf.nn.rnn_cell.MultiRNNCell(
                    [LSTMnode]*self.config.no_of_layers)
        initialState = stackedLSTMnode.zero_state(tf.shape(self.nodeFinalInput)[0],
                    tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(stackedLSTMnode,
                    self.nodeFinalInput, initial_state=initialState)
        outputs = tf.reshape(outputs, [-1, self.config.hidden_dim])
        with tf.variable_scope("hiddenLayer"):
            Whidden = tf.get_variable("Whidden",
                        shape=[self.config.hidden_dim, self.config.vocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            bhidden= tf.get_variable("bhidden",
                        shape=[1, self.config.vocab_size],
                        initializer=tf.constant_initializer(0))
            # compute the logits:
            self.hiddenLayers = tf.matmul(outputs, Whidden) + bhidden
        print "addLayers!!"

    def lossFunction(self):
    #target will be [-1,0.9776,0.9873,-1]
        target = tf.reshape(self.targetPH, [-1])
    #mask will become [False,True,True,False]
        mask = tf.greater_equal(target, 0)

    #  1-D example this is an exmaple take a look
    # tensor = [0, 1, 2, 3]
    # mask = np.array([True, False, True, False])
    # boolean_mask(tensor, mask)  # [0, 2]
        maskedTargets = tf.boolean_mask(target, mask)
        maskedHidden = tf.boolean_mask(self.hiddenLayers, mask)

        loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=maskedHidden, labels=maskedTargets)
        # average the loss over all words to get the batch loss:
        loss = tf.reduce_mean(loss_per_word)

        self.loss = loss
        print "lossFunction!!"

    def add_training_op(self):
        """
        - DOES: creates a training operator for optimizing the loss.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.trainOp = optimizer.minimize(self.loss)

    def runEpoch(self,sess):
        lossInBatch=[]
        for captions, imgVectors, targets in getTrainDataIterator(self):
            feedDict = self.constructFeedDict(captions,imgVectors,targets,self.config.dropout)
            eachLoss, _ = sess.run([self.loss, self.trainOp],
                        feed_dict=feedDict)
            lossInBatch.append(eachLoss)
        return lossInBatch
        """
        captions=[[sos,a,adarsh,is,good,eos],[jcsajfjkahfka]]
        imgvectors=[[567567],[878778]]
        targets=[[a,adarsh,is,goof,eos,-1],[bcmsabjfksahk]]
        """
    def imgCaptionGenerator(self,sess,vocabulary,imgVector):
        predIndex = 1
        #initialise all the vectors
        image = np.zeros((1,self.config.img_dim))
        image[0] = imgVector
        caption = np.zeros((1,1))
        caption[0]=np.array(vocabulary.index("<SOS>"))
        while caption.shape[1]<self.config.max_caption_length and int(caption[0][-1]) is not vocabulary.index("<EOS>"):
            feedDict = self.constructFeedDict(caption,image)
            hiddenLayer = sess.run(self.hiddenLayers, feed_dict=feedDict)
            predictionHiddenLayer = hiddenLayer[predIndex]
            predictedWordIndex = np.argmax(predictionHiddenLayer)

            nextWord = np.zeros((1,1))
            nextWord[0]=np.array(predictedWordIndex)
            caption = np.append(caption,nextWord,axis=1)
            predIndex+=1
        caption=caption[0].astype(int)
        caption = detokenizeCaption(caption,vocabulary)
        return caption

    def valCaptionGenerator(self,epoch,sess,vocabulary):
        captionsFile = "%s/generated_captions/captions_%d.json"\
                    % (self.config.model_dir, epoch)
        captions=[]
        valImgIDvsFeatureVec = cPickle.load(open("coco/data/valIDvsImageFeature"))
        valImgIDvsFeatureVecList = valImgIDvsFeatureVec.items()
        valSet = valImgIDvsFeatureVecList[0:2500]
        for imgId,imgVec in valSet:
            captionObj = {}
            imgCap = self.imgCaptionGenerator(sess,vocabulary,imgVec)
            captionObj["image_id"] = imgId
            captionObj["caption"] = imgCap
            captions.append(captionObj)
        with open(captionsFile, "w") as file:
            json.dump(captions, file, sort_keys=True, indent=4)
        return captionsFile

def main():
    config = LSTM_config()
    gloveEmbeddings = cPickle.load(open("coco/data/embeddingsMatrix"))
    print "gloveembeddings loaded"
    gloveEmbeddings = gloveEmbeddings.astype(np.float32)
    model = LSTM_model(config,gloveEmbeddings)
    print "#######LSTM MODEL CREATED#######"
    lossperEpoch = []
    evalMetricsperEpoch = []

    # create a saver for saving all model variables/parameters:
    saver = tf.train.Saver(max_to_keep=model.config.max_no_of_epochs)

    with tf.Session() as sess:
        # initialize all variables/parameters:
        initVar = tf.global_variables_initializer()
        sess.run(initVar)
        # //writer = tf.summary.FileWriter("models/LSTM/graph", sess.graph)
        for epoch in range(model.config.max_no_of_epochs):
            print "---------------------------------"
            print "!!<<<<<<<<<<NEW EPOCH>>>>>>>>>>!!"
            print "epoch no: %d/%d" %(epoch,config.max_no_of_epochs-1)
            print "---------------------------------"

            batchLosses = model.runEpoch(sess)
            epochLoss = np.mean(batchLosses)
            lossperEpoch.append(epochLoss)
            valCaptions = model.valCaptionGenerator(epoch,sess,model.vocabulary)
            evalDict = evaluateCaptions(valCaptions)
            if evalDict["CIDEr"]>0.80 or epoch == model.config.max_no_of_epochs-1:
                print "found at epoch:"+str(epoch)
            saver.save(sess, "%s/weights/model" % model.config.model_dir,
                            global_step=epoch) 

if __name__ == '__main__':
    main()