Welcome to the VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression wiki!

### Note:
Any help needed to execute the code or understanding the code, please send me message. I will soon upload the input data and expected output data.
More updates coming soon. Meanwhile Please look at the code. I have added some comments over there.  

## Note: How to convert categorical feature to numerical feature:
(i) Simply convert the string to number by either some ranking or order. if you have set of strings, give a number to each other. Not a good solution. As this way, number will not take any sense.  
(ii) One hot encoding. - Very popular.  one-hot encoding, I say I have set of 5 colors, it will replace one feature to 5 features of binary vector of a size of number of distinct colors. It is a sparse and large vector.   Hair color is a categorical feature as it has multiple category. If I need to figure out from Country, height of the person, I can do that.  Example 200 countries, I can create matrix of 200 features. Each column will have mean of the height of the people per country. so, I can replace the feature of country with avg height. so, I introduce new 200 features by one country feature.  so, I convert categorical feature to a set of numeric features.
(iii) BoW - Bag of words.  - Another popular. And very similar to one-hot encoding.  
(iv)  Using Domain Knowledge:  Given a country, replace food taste or food quality with number.  
(iv) Mean replacement. 

Try all the above algorithms to see which one works  for your data.


# VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression
https://github.com/VipinJain1/VIP-K-Nearest-Neighbors-Naive-Bayes-Logistic-Regression/wiki/Some-Quick-Tips

#### Good Links:
http://shatterline.com/blog/2013/09/12/not-so-naive-classification-with-the-naive-bayes-classifier/
https://stats.stackexchange.com/questions/188416/discriminant-analysis-vs-logistic-regression

#### Very good quick tricky guide:

https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor
https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761 
https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222 

### Refer This - Quick Informative:
https://www.kaggle.com/mayu0116/hyper-parameters-tuning-of-dtree-rf-svm-knn


#### Any model I use, I need to clean, train, test data. All are very common in all the APIs.scikit-lean has super brilliant libs to use. So really for ML engineer code to write is like nothing. What you need is to understand the real data and make changes in the data accordingly. Just most of the work is in modelling data. Fixing underfit or overfit data - super critical to have data normalized.

### Logistic regression:
https://www.philippeadjiman.com/blog/2017/12/09/deep-dive-into-logistic-regression-part-1/

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
   
   Naive Bayes:
    (i) if features are real valued, we had gaussian dist.
    (ii) Class label is random var.
    
    Read book  https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
    (iii) X and Y conditionalyy independent. 
    Linear Regression = Gaussian Naive Bayes  + Bernouli
    
    ### Loss minimization interpretation of LR:
    
    Remember  W* = ArgMin(Sum (Log (1+exp (-Yi W(t)Xi))))  from 1 to n
    Zi = Yi W(t) Xi = Yi * F(Xi)
    I want to minimize incorrectly classified points. that is the whole point of classification. 
    I want to minimize my loss. +1 to incorrectly classified  and 0 for correctly classified.  so we want to minimize the loss that is       the loss function. 
      W* = argmin(sum(0_1 loss(Xi,Yi,w)))) 
      
     F(x) is differentiable if x is continious, in order to solve loss function. 
     
     ### HyperParameter Search/Optimization:
     Lambda =0, Overfitting
     Lambda =1 , Underfitting.
     Lambda in LR is a real value. Find right lambda.
     One way is grid search like brute force technique:
     take limited set of numbers of lambda
     search lambda in >= [0.0001,0.01, 0.1,1,10.....100...1000], some folks search lamda in very wide window. 
     Check cross validation error for each lambda, that should be minimal. 
     
    
  ### Column/Feature Standradization:
    
   Matrix:  Calculate mean and std dev of as matrix.  this is Standardization. Even is logistic regression it is mandatory to perform column Standardization. if two features are on multiple scale, we need to standardize columns/features. It is also called mean centric or scale. So, all features are brought to the same scale.
   
  #### Feature Importance & Model Interpretability:
We have d features and optimal weight vector for each feature. so we get f1,f2,f3...fn  features and weight w1,w2,w3...wn.
assume we have all features independent as we know logistic regression from probabilistic stand point is - Gaussian naive base + Bernoulli distribution on class labels. If all features are independent then - we can get feature importance using Wj's. But in realistic it is not possible. 

Using K-NN :- We could get feature importance using forward feature section method.
in Naive base: - we could tell feature imp using probability:  using P(Xi|Y=+1) -> feature which are important.
in LR :-  we get  |Wj| abs value of wt corresponding to feature J. If abs value of Wj is large, then it's contribution to WtXj is large.
So, if |Wj| is large then (Wt Xi) is large so absolute value matters. even if W is negative it will impact because of abs value.
Example:

If we want to predict male or female, male +1 and female -1. we can get hair length.
since most women will have longer hair then men.

So, if weight (hair length) increases, probability of negative class increases since female is -1.
If another example of feature height, male is generally having more height then female. so, probability of height of positive (male) class label will also increase.

#### Model Interpretability:
If I want to tell model is sensible or not, if Yq is +1 or -1, class label is positive or negative.  I can pick up most important feature which has absolute weight value which is large. I can pick those features only. I can interpret based on weight if that person is male or female based on length of hair for example.  if height is tall and hair length is short, I can say person is male.

#### Colinearity or MultiColinearity:
If F(i) = A (Fj) + B so we can say Fi and Fj are colinear.
If features have multicollinear dependency, we cannot use abs (weight) of feature. Implement dependencies and weight vector will also change.

#### Perturbation Technique: use for detecting multicollinearity.
in a matrix, add some small noise on cell value, and train your logistic regression again after adding some noise, if weights values are different that means features are colinear.

#### Nonlinear planes: Convert to linear before applying logistic regression.

## The Most important aspects of ML/AI:
(i)   Feature engineering.
(ii)  Bias - variance 
(iii) Data Analysis & Visualization.

Examples:
XOR data - convert to linear plane before applying LR.

### Performance Measurement of Models:

Accuracy = # Correctly classified points/Total number of points in D test

##### Issues with accuracy:

(i) Case of imbalanced data. 

### Confusion Matrix: 
Does not take probability scores.
Binary Classification task:two class (0,1)

TPR = TP/P
TNR = TN/N
FPR = FT/N
FNR = FN/P

Precesion:TP/TP+FP
Recall = TP/P
f1 sCORE= 2* (Pre*Recall)/(Pre+Recall)

### AUC Curve:

















  

  

    
    
    
     
     
      
     
   
    

    
   
   
   
    
   

 




 

 





  





