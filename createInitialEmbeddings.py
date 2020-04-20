"""
creation of an initial embedding matrix
"""

import numpy as np
import os
import cPickle

captionsDIR = "coco/annotations"
dataDIR = "coco/data"
pretrainedWords = []
wordsVector = []
#according to the dimension  specified in the file name
word_vec_dim = 300

vocabulary = cPickle.load(open(os.path.join(dataDIR,"vocabulary")))
vocabSize = len(vocabulary)
print vocabSize
with open(os.path.join(captionsDIR,"glove.6B.300d.txt")) as file:
    for eachLine in file:
        eachLine = eachLine.strip()
        lineContents = eachLine.split(" ")
        word = lineContents[0]
        wordVector = lineContents[1:]

        pretrainedWords.append(word)
        wordsVector.append(wordVector)


''' create an embedding matrix where row i is the pretrained word vector
 corresponding to word i in the vocabulary'''
 
embeddingMatrix = np.zeros((vocabSize,word_vec_dim))
for vocabIndex,word in enumerate(vocabulary):
    if str(word) not in ["<SOS>", "<UNK>", "<EOS>"]:
        wordVectorIndex = pretrainedWords.index(str(word))
        eachWordVector = wordsVector[wordVectorIndex]
        eachWordVector = np.array(eachWordVector)
        eachWordVector = eachWordVector.astype(float)
        embeddingMatrix[vocabIndex,:] = eachWordVector
cPickle.dump(embeddingMatrix,open(os.path.join(dataDIR,"embeddingsMatrix"),"wb"))

print np.shape(embeddingMatrix)