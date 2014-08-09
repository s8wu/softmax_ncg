from softmax import Softmax
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
softmaxNode = Softmax()
softmaxNode.fit(train, y, batch_size=0.3333, maxiter=12, valid_frac=0.16667, 
                grad_batch=False, alpha=0, verbose=True, reset_params=True)
pred = softmaxNode.predict(test)
pred = enc.active_features_[pred]
error = np.sum(pred != testLabels) / np.double(len(testLabels))
print('Test error: ' + str(error))