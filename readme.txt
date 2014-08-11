Softmax regression for Python solved using scipy's Newton-CG. 

Example for MNIST data set (MNIST data obtained from http://deeplearning.net/data/mnist/mnist.pkl.gz):


python sample.py

Validation error: 0.2334
Validation error: 0.1618
Validation error: 0.1181
Validation error: 0.095
Validation error: 0.0803
Validation error: 0.0737
Validation error: 0.0717
Validation error: 0.0703
Warning: Maximum number of iterations has been exceeded.
         Current function value: 12002.015746
         Iterations: 10
         Function evaluations: 11
         Gradient evaluations: 20
         Hessian evaluations: 116
  status: 1
 success: False
    njev: 20
    nfev: 11
     fun: 12002.015745749864
       x: array([-0.85148077,  0.        ,  0.        , ...,  0.        ,
        0.        ,  0.        ])
 message: 'Maximum number of iterations has been exceeded.'
    nhev: 116
     jac: array([-17.54970361,   0.        ,   0.        , ...,   0.        ,
         0.        ,   0.        ])


Test error: 0.0736



