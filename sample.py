from softmax_ncg import Softmax_Ncg
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gzip, cPickle

def getMNISTData(filePath='mnist.pkl.gz'):
    f = gzip.open(filePath,'rb')
    train, valid, test = cPickle.load(f)
    f.close()
    return train[0], train[1], valid[0], valid[1], test[0], test[1]
    
train, trainLabels, valid, validLabels, test, testLabels= getMNISTData()

train = np.row_stack((train,valid))
trainLabels = np.concatenate((trainLabels, validLabels))

trainLabels = trainLabels[:,np.newaxis]
enc = OneHotEncoder()
enc.fit(trainLabels)
y = enc.transform(trainLabels).todense().A
softmaxNode = Softmax_Ncg()
softmaxNode.fit(train, y, batch_size=0.75, maxiter=10, valid_frac=0.166667, 
                grad_batch=True, alpha=0, verbose=True)
pred = softmaxNode.predict(test)
pred = enc.active_features_[pred]
error = np.sum(pred != testLabels) / np.double(len(testLabels))
print('\n\nTest error: ' + str(error))