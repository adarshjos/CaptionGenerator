
from model import LSTM_config,LSTM_model
from imageFeatureExtraction import extractfeatureInception
import tensorflow as tf
import cPickle
import numpy as np
# import pyttsx3 
  
# initialisation 
# engine = pyttsx3.init() 
  
# testing 

# pathimg=["coco/images/test2014_1/COCO_test2014_000000000304.jpg"]

def generateCap(imgIDvsImageFeatureVector):
    vocabulary = cPickle.load(open("coco/data/vocabulary"))
    config = LSTM_config()
    dummy_embeddings=np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_model(config, dummy_embeddings, mode="testing")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "models/LSTM/weights/model-19")
        # tf.reset_default_graph()
        img_caption = model.imgCaptionGenerator(sess,vocabulary, imgIDvsImageFeatureVector)

    return img_caption
    
# def main():
#     print "#$$#@#@#"
#     print type(pathimg)
#     vectorFeature=extractfeatureInception(pathimg,True)
#     caption="Hello all"
#     caption=generateCap(vectorFeature[0])  
#     print caption
#     # engine.say(caption)
#     # engine.runAndWait() 


# if __name__ == '__main__':
#     main()
