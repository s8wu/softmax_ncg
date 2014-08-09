import numpy as np
from scipy.optimize import minimize

class Softmax():
    
    _params = None
    _validError = float('inf')
    _intercept = True
    
    def __init__(self):
        pass

    def predict(self, x):
        """Predict class number for samples in x."""        
        prob = self.predict_proba(x)
        pred = np.argmax(prob, axis=1)
        return pred
    
    def predict_proba(self, x):
        augmentedX = self._augment_data(x)
        pred = self._activation_func(self._params, augmentedX)
        return pred
        
    def _activation_func(self, params, x):
        K = params.shape[0] / x.shape[1]
        M = x.shape[1]
        newParams = np.reshape(params, (M,K), order='F')
        newParams = np.reshape(newParams, (M*K), order='C')
        newParams = np.reshape(newParams, (M,K), order='C')
        eta = np.dot(x, newParams)
        expEta = np.exp(eta)
        normConst = np.sum(expEta, axis=1)[:, np.newaxis]
        y_hat = expEta / normConst
        return y_hat
    
    def fit(self, x, y, obs_weight=1, alpha=0, batch_size=0.25, 
            maxiter=10, valid_frac=0.1, grad_batch=True, intercept=True,
            reset_params=True, valid_type='class_error', verbose=False):
        """
        Fit maximum likelihoood parameters of a multinomial distribution using 
        Newton-CG. 
    
        Parameters
        ----------
        x: 2-d array of size [n_samples, n_features]
            Training data with observations along rows and predictors along 
            columns (N observations with M predictors).
        y: 2-d array of size [n_samples, n_classes]
            Probabilities of sample belonging to each class. 
        obs_weight: 1-d array of size [n_samples]
            Vector of observations weights or 1 for no weighting.
        alpha: float
            Penalty value for L2-regularization. Defaults to 0.
        batch_size: float
            Fraction of training samples in the range (0,1] to use for 
            estimating the hessian and/or gradient for each iteration of 
            Newton-CG.
        maxiter: int
            Maximum number of iterations to run Newton-CG.
        valid_frac: float
            Fraction of training samples to use for validation. The last 
            n_samples * valid_frac will be used as validation set. After each
            Newton-CG iteration the proposed update to parameters is only 
            accepted if it reduces the validation error. Defaults to 0.1. 
        grad_batch: boolean 
            If true, then gradient will be estimated with mini-batch as well.
            If False, then gradient will be estimated using all observations.
            Defaults to True.
        intercept: boolean
            If True then intercept terms are also fitted. Defaults to True.
        reset_params: boolean
            If False then use current parameter set as starting point of new
            fit. Otherwise reset to vector of zeroes. Defaults to True.
        valid_type: string
            How to evaluate results of validation; either 'class_error' for 
            classification error or 'cross_ent' for cross entropy. Defaults
            to 'class_error'
        verbose: boolean
            If True then prints extra information during and after fitting.
    """
        def obj_fun(params, x, y, obs_weight, alpha, *args):
            y_hat = self._activation_func(params, x)
            M = x.shape[1]
            crossEntropy = np.multiply(y, np.log(y_hat))
            crossEntropy = np.multiply(obs_weight, crossEntropy)
            crossEntropy = -np.sum(crossEntropy)
            crossEntropy = crossEntropy + np.dot(params, params) * alpha
            #subtract the regularization penalty on the intercept terms.
            if(self._intercept):
                crossEntropy = crossEntropy - np.sum(params[0::M]**2) * alpha
            return crossEntropy
        
        def grad(params, x, y, obs_weight, alpha, y_hat, batch, gBatch):
            if(gBatch):
                batchSlice = slice(batch[0], batch[-1]+1)
                x = x[batchSlice,:]
                y = y[batchSlice,:]
                if(not(type(obs_weight) is int)):
                    obs_weight = obs_weight[batchSlice]
            y_hat = self._activation_func(params, x)
            M = x.shape[1]
            N = x.shape[0]
            K = y.shape[1]
            err = np.multiply(y - y_hat, obs_weight)
            err = np.reshape(err, N*K, order='F')
            err = np.reshape(err, (K,N), order='C')
            grad = -np.dot(err, x)
            grad = grad.ravel(order='C')
            regBeta = np.multiply(params, alpha)
            if(self._intercept):
                regBeta[0::M] = 0
            grad = grad + regBeta
            return grad
        
        #Input "params" remains constant per Newton-CG iteration. As such, it 
        #is assumed that yhat does not need to be recomputed within hessp.
        def hessp(params, vec, x, y, obs_weight, alpha, yhat, batch, *args):
            #From batch indices create a batch view of the data.
            batchSlice = slice(batch[0], batch[-1]+1)            
            x = x[batchSlice,:]
            yhat = yhat[batchSlice,:]
            if(not(type(obs_weight) is int)):
                obs_weight = obs_weight[batchSlice]
            N,M = x.shape
            K = y.shape[1]
            hessProd = np.zeros(M*K)
            #Reshape vec into an M,K matrix while preserving C contiguity. This
            #assumes inputs are already C-contiguous
            newVec = np.reshape(vec, (M,K), order='F')
            newVec = np.reshape(newVec, (M*K), order='C')
            newVec = np.reshape(newVec, (M,K), order='C')
            a_prime = np.dot(x, newVec)
            h_prime = np.multiply(-yhat, a_prime)
            h_prime = np.sum(h_prime, axis=1)
            h_prime = h_prime[:,np.newaxis] + a_prime
            h_prime = np.multiply(h_prime, yhat)
            h_prime = np.multiply(h_prime, obs_weight)
            h_prime = np.reshape(h_prime, N*K, order='F')
            h_prime = np.reshape(h_prime, (K,N), order='C')
            hessProd = np.dot(h_prime, x)
            hessProd = hessProd.ravel(order='C')
            #Add regularization effect without intercepts.
            regTerm = np.multiply(vec, alpha)
            if(self._intercept):
                regTerm[0::M] = 0
            hessProd = hessProd + regTerm
            return hessProd
        
        def callback(xk):
            """
            Callback function that is called after each Newton-CG iteration.
            Here the predicted probabilities and mini batch slice is updated.
            Checks based on validation set are also done here.
            """
            #Update yhat and batch. These are defined outside of this scope.            
            yhat[:,:] = self._activation_func(xk, train)
            batch[:] = batch + batchStep
            #If next batch goes over set then start again at offset position.
            if(batch[-1] > train.shape[0] - 1):
                stubLength = train.shape[0] - batch[0]
                batch[:] = np.array(range(stubLength, stubLength + batchStep))
            if(valid_frac > 0):
                xkOld = self._params
                self._params = xk 
                if(valid_type == 'class_error'):
                    valid = x[validInd,]
                    validLabels = y[validInd,]
                    pred = self.predict(valid)
                    error = percentError(pred, validLabels)
                elif (valid_type == 'cross_ent'):                    
                    valid = augmentedX[validInd,]
                    validLabels = y[validInd,]
                    error = obj_fun(xk, valid, validLabels, 1, alpha)
                else:
                    raise Exception("Unrecognized valid_type input")
                if(error < self._validError):
                    self._validError = error
                    if(verbose):
                        print('Validation error: ' + str(error))
                else:
                    self._params = xkOld
            else:
                self._params = xk
        
        def percentError(pred, response):
            labels = np.argmax(response, axis=1)
            error = np.sum(pred != labels) / np.double(len(pred))
            return error            
        
        self._intercept = intercept
        #Add intercept term to feature matrix if required. Copies data in x.
        augmentedX = self._augment_data(x)
        augmentedX = np.require(augmentedX, requirements='C')        
        #Define slice that belongs to validation set.
        numValid = int(x.shape[0] * valid_frac)
        validInd = slice(x.shape[0] - numValid, x.shape[0])
        #Define training set
        train = augmentedX[0:(x.shape[0] - numValid),]
        response = y[0:(x.shape[0] - numValid),]
        #Number of covariates
        M = augmentedX.shape[1]
        #Number of categories
        K = y.shape[1]
        if(not (reset_params) and not self._params is None):
            x0 = self._params
        else:
            self._validError = float('inf')
            x0 = np.zeros(M*K)
        yhat = self._activation_func(x0, train)
        batchStep = int(np.round(train.shape[0] * batch_size))
        batch = np.array(range(0,batchStep))
        #Extra arguments to be passed into Newton-CG
        extraArgs = (train, response, obs_weight, alpha, yhat, batch, 
                     grad_batch)
        options = {'maxiter':maxiter,'disp':verbose}
        results = minimize(obj_fun, x0, args=extraArgs, method='Newton-CG', 
                           jac=grad, hessp=hessp, options=options,
                           callback=callback)
        if(results.status == 0 or results.status == 1 or results.status == 2):
            if(verbose):
                print(results)
        else:
            raise Exception(results.message)
        
    def _augment_data(self, x):
        if(self._intercept):
            augmentedX = np.column_stack((np.ones(x.shape[0]),x))
            return augmentedX
        else:
            return x