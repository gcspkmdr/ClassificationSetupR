# ClassificationSetupR

# References
Watch videos mentioned below to build intution(....and that's all....don't worry about terminology...or maths). 
 Don't worry if you don't understand every single thing.
 
Focus on things like :
 1. What each algorithm is trying to do?
 2. How each algorithm is different ?
 3. Why classifiers output probablities?
 4. Which columns/features are significant predictors and helping explain the underlying phenomenon?
 5. Metrics like Accuracy, ROC, Sensitivity, Specificity(.....very important)
 6. Understand what a confusion matrix is
 6. Why accuracy can be a misleading metric to use(You will understand this once you understand class imbalance prolem)
 7. Understand ROC curve and how it is built(Understand confusion matric before this). As a business student you should know how to interpret this. 

Logistic Regression : https://www.youtube.com/watch?v=yIYKR4sgzI8&vl=en

Logistic Regression : https://www.youtube.com/watch?v=vN5cNN2-HWE

Confusion Matrix : https://www.youtube.com/watch?v=Kdsp6soqA7o

Sensitivity/Specificity : https://www.youtube.com/watch?v=vP06aMoz4v8

ROC Curve : # https://www.youtube.com/watch?v=4jRBRDbJemM

https://www.datacamp.com/community/tutorials/decision-trees-R  -> Read...it's fairly simple

Decision Tree : https://www.youtube.com/watch?v=7VeUPuFGJHk

Ensembling : https://www.youtube.com/watch?v=Un9zObFjBH0

Random Forest : https://www.youtube.com/watch?v=J4Wdy0Wc_xQ

Bagging : https://www.youtube.com/watch?v=2Mg8QD0F1dQ

Boosting(AdaBoost) : https://www.youtube.com/watch?v=LsK-xG1cLYA

Boosting(Gradient Boost) : https://www.youtube.com/watch?v=jxuNLH5dXCs  ---> Skip if you want ---> Get intution from adaboost video

KNN : https://www.youtube.com/watch?v=HVXime0nQeI

LDA :https://www.youtube.com/watch?v=azXCzI57Yfc

Class Imbalance : https://towardsdatascience.com/class-imbalance-problem-in-classification-a2ddaba98f4a
 

# Data imputation

Missing data is a common problem in practical data analysis. They are simply observations that we intend to make but did not. In datasets, missing values could be represented as ‘?’, ‘nan’, ’N/A’, blank cell, or sometimes ‘-999’, ’inf’, ‘-inf’.  Most machine learning algorithms (kNN is a notable exception) cannot deal with this problem intrinsically, as they are designed for complete data. There are many ways to approach missing data. One common approach is imputation. Imputation simply means replacing the missing values with an estimate, then analyzing the full data set as if the imputed values were actual observed values.


# Encoding Categorical Variable

Label Encoding : This is the simplest form of encoding where each value is converted to an integer.The maximum value is the equal to the number of unique values of the variable.

One hot encoding : This is probably the most common form of encoding and is often referred to as creating dummy or indicator variables. It creates a new column for each unique value of the categorical variable. Each of these columns are binary with values 1 or 0 depending on whether the value of the variable is equal to the unique value being encoded by this column.

# Data Standardization

Do this for dataset that has multiple features spanning varying degrees of magnitude, range, and units. This is a significant obstacle as a few machine learning algorithms are highly sensitive to these features. Distance based algorithms like KNN, K-means, and SVM are most affected by the range of features. This is because behind the scenes they are using distances between data points to determine their similarity. Tree-based algorithms, on the other hand, are fairly insensitive to the scale of the features. If you don't perform standardization the algorithm takes more time(iterations) to arrive at optimum point(i.e the best classifier), which isn't considered a good practice in data science circles. There are many approaches(eg: Stndardization, Normalization etc.). We are going to use Stnndardization. Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.

# Train/Val/Test Split

![alt text](https://miro.medium.com/max/1400/1*Nv2NNALuokZEcV6hYEHdGA.png)

For any model building process daataset is divided into 3 parts:

1. Train
2. Validation
3. Test

A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters.The validation dataset is different from the test dataset that is also held back from the training of the model, but is instead used to give an unbiased estimate of the skill of the final tuned model when comparing or selecting between final models.

Validation dataset is used for model tuning, which most likely you won't be doing in your exams, and Test set is used to evaluate your model.So, for your purposes test and validation set are one and the same thing. In your exams most likely you will be given one file. Divide it in two parts train/val set(demonstrated in code)



# Must Read
# Why accuracy can be a misleading metric?

The most commonly reported measure of classifier performance is accuracy: the percent of correct classifications obtained. This metric has the advantage of being easy to understand and makes comparison of the performance of different classifiers  trivial, but it ignores many of the factors which should be taken into account when honestly assessing the performance of a classifier.
Classifier performance is more than just a count of correct classifications.Consider, for interest, the problem of screening for a relatively rare condition such as cervical cancer, which has a prevalence of about 10% (actual stats). If a lazy Pap smear screener was to classify every slide they see as “normal”, they would have a 90% accuracy. Very impressive! But that figure completely ignores the fact that the 10% of women who do have the disease have not been diagnosed at all.

# Why ROC curves serve as a better metric?

ROC curve is a plot between the True Positive Rate and False Positive Rate.For a perfect classifier the ROC curve will go straight up the Y axis and then along the X axis. A classifier with no power will sit on the diagonal, whilst most classifiers fall somewhere in between. It is immediately apparent that a ROC curve can be used to select  a threshold for a classifier which maximises the true positives, while minimising the false positives. However, different types of problems have different optimal classifier  thresholds. For a cancer screening test, for example, we may be prepared to put up with a relatively high false positive rate in order to get a high true positive,  it is most  important to identify possible cancer sufferers.For a follow-up test after treatment, however, a different threshold might be more desirable, since we want to minimise  false negatives, we don’t want to tell a patient they’re clear if this is not actually the case. ROC curves also give us the ability to assess the performance of the  classifier over its entire operating range. The most widely-used measure is the area under the curve (AUC). The AUC can be used to compare the performance of two or more classifiers. A single threshold can be selected and the classifiers’ performance at that point compared, or the overall performance can be compared by considering the AUC.

