Welcome to the VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression wiki!

# VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression
https://github.com/VipinJain1/VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression/wiki/Some-Quick-Tips

Good Link:
http://shatterline.com/blog/2013/09/12/not-so-naive-classification-with-the-naive-bayes-classifier/

## Any model I use, I need to clean, train, test data. All are very common in all the APIs.scikit-lean has super brilliant libs to use. So really for ML engineer code to write is like nothing. What you need is to understand the real data and make changes in the data accordingly. Just most of the work is in modelling data. Fixing underfit or overfit data - super critical to have data normalized.

# Basic Concepts:

## Over Fitting:
small change  in train data results in large change in the model.  if alpha is 0, we have over fitting or high variance problem. 
 
## Under Fitting:
High Bias. Alpha is very large - problem of under flitting. 
How to find right Alpha: Using K in k-fold validation or Alpha in Naive Bayes theorem.. 

## How to get feature importance in Naive Bayes?
p(Wi/Y=1) sort by desc order, top features will have higher imp features for positive class. 
p(Wi/Y=0) sort by desc order, top features will have higher imp features for negative class.

## Intemperatibility:
Sq -> Yq
Sq = [w1,w,2,w3,w,4....sn]

I am concluding, Yq=1 because Xq containing  words w3,w6,w10 which have a high value of P(w3|y=1) ,  P(w6|y=1), P(w10|y=1)
similarly we can say for negative  where y=0

## Imbalance Data:
N has n1 and n2 data. if n1>>> n2, data is imbalance. 
P(y=1|w1,w2,w3,...wn) = P(y=1)P(w1|y=1)P(w2|y=1)P(w3|y=1)...P(wn|y=1) = p(n1)
P(y=0|w1,w2,w3,...wn) = P(y=0)P(w1|y=0)P(w2|y=0)P(w3|y=0)...P(wn|y=0) = p(n2)
say p(n1) = 0.9 so p(n2) =0.1
### Solution:
(1) Try up-sampling and down-sampling make P of both =0.5 ( I do not like this idea)
(2)  Modified NB to account for class imbalance. 

## Outliers:
Xq = w1,w2,w3,....w' if w' does not exist , Laplace smoothing can take care of it. 
w' is  the outlier since I have mot seen it in my training data set. 
(1) 
if w8 does not occur many times, we can say it is outlier.
If word(Wj) say occur fewer than 10 times, just ignore/drop that word. do not keep in your set. this is one way to reduce outlier.

(2) Laplace smoothing can take care of it.. 

## Missing Values:
(1) If I have text-data, no issue. no case of missing data.
(2) If I have categorical features, f1 belongs to (a1,a2,a3) categories.  if we have missing values of any categorical feature, we can say it is part of NaN.
(3) Numerical Features: we can use mean, median, naive bayes etc.

## NB Built Numerical Features:

very powerful way to get PF from Gaussian dist of Bayes data.

## Multi-Class Classification in NB.
we can do in NB.

## Similarity Matrix and Distance Matrix in NB:
can be done in K-neighbor.
NB can not handle it.  we need to compute prob for f1,f2,,,
NB does no use distances.  it is a probability based method. We need actual feature values.

## Large Dimensions in NB:

Can be used easily.  better use log probability since data is huge because of multiplication. D is large, prod will be super huge. we would not have underflow  or numerical stability issue if we use log. 
  
## Good and Bad about NB:
1. Conditional independence of features , if True NB works very well.  If false, NB  performance degrades.
2. Even if some features are dep, still NB works well. 
Text classification Problem:  :Like mail spam email, prod review  - works very well. 
Categorical or binary features: works very well.  Real-values features - seldom used. we do not use much Gaussian dist, but used power law. 
NB is super interpret able, in medical where to tell doc, super simple to understand. just about continue.  we can easily over fit if we do not do lap-lase smoothing to find right alpha using cross validation.  so we need to use laplase smoothing in every case.  sktlearn takes care if it. 
knn we need whole data at run time.   

## Scikit-learn  very good doc. 
https://scikit-learn.org/stable/modules/naive_bayes.html

### Logistic Regression (LR)

  It is a classification technique.  Geometric based, NB was probabilistic based. 
  It is a classification technique.  Geometric based, NB was probabilistic based. 
  LR can be derived by geometric, probability or loss-function, but best is geometric. There are three ways to interperate.  
  If data is linearly separable in a plane by line. line is  y =mx+c , in plan = W(transpose)x+b =0,
  x and W are vectors, b is a scaler, 
  Assumption: - Classes are almost linearly separable. That is the best way to use LR.
  NB has conditional independence. K-NN is neighbor based. In LR, plane is divided in two parts, plus or minus points.
  Task is to find a plane, that will separate positive and negative points.

  if Wtxi >0 Yi =+1  if   Wtxi <0 Yi =-1 , LR is decision based.
  Optimal W =  argMax(sum (YiWtXi))  --> Get Max value. 
  LR can have many planes, we need best Wi, best plane. This is optimization problem.
  Wtxi is the distance from Xi to the plane.
  YiW(transpose)xi : +ve so plane is correctly classifies. if <0 means incorrectly classifies. 
  Outlier can mess up the data. In LR we would want as many as positive values for positive place and as many as negative values for negative plane. This the optimization function of LR. Find out the optmimal W (max values of + and -)

  Extreme outlier points can impact the model badly. max sum is out outlier prone. 
  So our above fucntion has to fix to take care of outlier. Squashing is used for that purpose.

  ### Squashing in LR:
  Instead of using signed distances directly, if signed distance is small, use as it, and if singed distance is large, make it a smaller value. If distance is huge, I wil not use huge value, I will much smaller.
  so we remove outlier.
  We use Sigmoid Function.

  Try Plot(1/1+e^-x)  to see the impact. X is the signed distance. 
  sigma (0) = 0.5
  Values are in between 0 to 1 instead of -infinity to infinity. So, it comes to probabilistic interpretation.

  ### Monotonic Function:
   G(x) increases if x increases.
   if x1 > x2 then g(x1) > g(x2) then g(x) is set to be monotonic funtion. 
   log(x) is defined when x>0  it is monotonic function. try plot (log(x)) on google.
    
   
   ### Optimization Equations:
   W" = ArgMin( Sum (log(1+ E(-Yi W(t) Xi))
   W = {w1,w2,w3,w4....wn}
   w is a feature. every feature I have weight associated. 
   if W(t) X >0 then Y is positive else Y is negative.
   
   ### Weight Vector:
    Weight vector is a D dim points.
    W = {w1,w2,ws3....wn} we have D features. 
    I have f1,f2,f3...fn , for every feature I have weight associated. 
    Xq -> Yq
    if W(t) X >0 then Y is positive else Y is negative.
    If W(t) Xq >0 then Yq =1 else Y=-1
    If weight of X increases, probability of Y also increases. 
    
   ### Regularization: Overfitting and Underfitting:
   
   Zi = Yi Wi(t) Xi , W Transpose.
   if I pick W such that:
   (a) all traning  points are correctly classified.
   (b) Zi -> infinity. Wi has to be + infinity or - infinity.   We get best fit. 
   
   Overfitting  - doing perfect job. 
   
   #### Regularization
    W* = ArgMin ( Sum ( Log (1+exp(-Yi W(t) Xi)))) + Lambda Wt * W we are minimizing both. 
    Lambda part is regularization. first term is the loss term. 
    
   ### Sparsity:
    W = w1,w,2,w3....wn
    Solution to LR is set to be sparse  if many Ws are zero. 
    if W vectror is sparse, solution of LR is also sparse.
    L1 regularization creates sparsity since weight set to zero.
   
   #### Elastic-Net
   Use L1 norm and L2 norm
   W* = ArgMin ( Sum ( Log (1+exp(-Yi W(t) Xi)))) + Lambda Wt * W  + lambda ||W square||
   
   ## Probabilistic Interpretation or derivation of Logistic Regression:
   
   One way is geometry and simple algebra. another way is probability and third way is loss minimization.
   
   Naice Bayes:
    (i) if features are real valued, we had gaussian dist.
    (ii) Class label is random var.
    
    Read book  https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
    (iii) X and Y conditionalyy independent. 
    Linear Regression = Gaussian Naive Bayes  + Bernouli
    
   
   
   
    
   

 




 

 





  





