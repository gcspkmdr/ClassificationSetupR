# References
# Watch videos to build intution(....and that's all....forget terminology...or maths). Don't worry if you don't understand every single thing.
# Focus on things like :
# 1. How each algorithm is different
# 2. Why classifiers output probablities?
# 3. Which columns are significant predictors?
# 4. Metrics like Accuracy, ROC, Sensitivity, Specificity(.....very important), and why accuracy can be a misleading metric to use(You will understand this once you understand class imbalance prolem)
# 5. Understand ROC curve. As a business student you should know how to interpret this. 
# Logistic Regression : https://www.youtube.com/watch?v=yIYKR4sgzI8&vl=en
# Logistic Regression : https://www.youtube.com/watch?v=vN5cNN2-HWE
# Confusion Matrix : https://www.youtube.com/watch?v=Kdsp6soqA7o
# Sensitivity/Specificity : https://www.youtube.com/watch?v=vP06aMoz4v8
# ROC Curve : # https://www.youtube.com/watch?v=4jRBRDbJemM
# https://www.datacamp.com/community/tutorials/decision-trees-R  -> Read...it's fairly simple
# Decision Tree : https://www.youtube.com/watch?v=7VeUPuFGJHk
# Ensembling : https://www.youtube.com/watch?v=Un9zObFjBH0
# Random Forest : https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
# Bagging : https://www.youtube.com/watch?v=2Mg8QD0F1dQ
# Boosting(AdaBoost) : https://www.youtube.com/watch?v=LsK-xG1cLYA
# Boosting(Gradientoost) : https://www.youtube.com/watch?v=jxuNLH5dXCs  ---> Skip if you want ---> Get intution from adaboost video
# KNN : https://www.youtube.com/watch?v=HVXime0nQeI
# LDA :https://www.youtube.com/watch?v=azXCzI57Yfc
# Class Imbalance : https://towardsdatascience.com/class-imbalance-problem-in-classification-a2ddaba98f4a
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load Libraries---------------------------------------------------------------------------------------------------------------------------------
library(tidyverse)

library(ggplot2) # For Data Visulaization

library(dummies) # For one-hot encoding

library(data.table)

`%notin%` <- Negate(`%in%`)

library(rpart) # For Decision Tree
library(rpart.plot) # For Plotting Decision Tree

library(ipred) # For Bagging

library(randomForest) # For Random Forest

library(gbm) # For Gradient boosting

library(class) # For KNN

library(MASS) # For LDA

library(ROCR) # For ROC curve

library(caret) #for variable importance
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load Data------------------------------------------------------------------------------------------------------------------------------------------------------------------
train_data <- read.csv(file.choose(), na.strings = c("", "NA"))

head(train_data, 10)

print("The following line prints the shape of dataframe")
dim(train_data)
summary(train_data)

#-----------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# In real business problem you would observe that each row is given a
# sort of unique id.You can also call it as a Primary Key. Generally it doesn't
# has any bussines significance. It just tells you that each record is unique and not a duplicate.
# The following section does nothing except creating a variable
# which holds the name of that primary key. It helps later on during handling of dataframes.
uuid_col <- ''
target <- 'hd'
#------------------------------------------------------------------------------------------------------------------------------------

# Check Data type of each column----------------------------------------------------------------------------------------------------
# At times what happens is that a column would be categorical, but beacause it contains numbers R assume them to be integer
# An example would be years of work ex because it can't take any value. Its limited in it range
# Use this space to change data type of any particular column you want
sapply(train_data, class) # this will tell you the data type of columns so make changes to columns data type according to your understanding

# Algorithms in R expects that the target column in classification problem is of type Factor.Convert Target Variable to factor(in case it encoded as intger in dataset)
train_data[, target] = as.factor(train_data[, target])


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Missing Values--------------------------------------------------------------------------------------------------------------------
# Following Function evaluates the % values missing in each columns
nullColumns <- function(df) {
  emptyCols = data.frame(colSums((is.na(df)))) / dim(df)[1]
  colnames(emptyCols) = c("Percentage_Missing")
  print(emptyCols)
  barplot(emptyCols$Percentage_Missing
          ,names.arg = row.names(emptyCols)
          ,main = "Percetange Values Missing in Dataframe"
          ,xlab = 'Columns'
          ,ylab = 'Percentage Values Missing'
          ,col = "darkblue"
          )
}
nullColumns(train_data)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Data imputation-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Missing data is a common problem in practical data analysis. They are simply observations that we intend to make but did not.
# In datasets, missing values could be represented as ‘?’, ‘nan’, ’N/A’, blank cell, or sometimes ‘-999’, ’inf’, ‘-inf’.
# Most machine learning algorithms (kNN is a notable exception) cannot deal with this problem intrinsically, as they are designed for complete data.
# There are many ways to approach missing data. One common approach is imputation. Imputation simply means replacing the missing values with an estimate,
# then analyzing the full data set as if the imputed values were actual observed values.

# Following Function generates mode of a given categorical column
Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1)
    xmode <- ">1 mode"
  return(xmode)
}

# Following function imputes missing values.Mean/Median for numeric columns.Mode for character or categorical columns
dataImputation <- function(df) {
  for (var in 1:ncol(df)) {
    if (class(df[, var]) %in% c("numeric", "integer")) {
      #df[is.na(df[,var]),var] <- mean(df[,var], na.rm = TRUE)
      df[is.na(df[, var]), var] <- median(df[, var], na.rm = TRUE)
    } else if (class(df[, var]) %in% c("character", "factor")) {
      df[is.na(df[, var]), var] <- Mode(df[, var], na.rm = TRUE)
    }
  }
  return(df)
}

train_data = dataImputation(train_data)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Data visualization-----------------------------------------------------------------------------------------------------------------------------------------------------------

# For a classification problem it is of utmost importance that the first thing one sees is if there is a class imbalance
# What is a class imbalance problem?
# This is a scenario where the number of observations belonging to one class is significantly lower than those belonging to the other classes
ggplot(train_data) + geom_bar(aes(x = train_data[, target])) + xlab(target) + ylab("Class Count")

# Now start analyzing individual variables wrt  the target variable(The following are example. Modify them according to the dataset given to you)
# Plot target variable against numeric variables
ggplot(train_data) + geom_boxplot(aes(x = train_data[, target] , y = train_data[, 'sbp'])) + xlab(target) + ylab('sbp')
ggplot(train_data) + geom_boxplot(aes(x = train_data[, target] , y = train_data[, 'tobacco'])) + xlab(target) + ylab('tobacco')
ggplot(train_data) + geom_boxplot(aes(x = train_data[, target] , y = train_data[, 'obesity'])) + xlab(target) + ylab('obesity')

# Plot two categorical variales against each other ( Stacked barchart , Side-by-Side bar chart)
ggplot(train_data, aes(x = train_data[, target], fill = famhist)) + geom_bar(position = "stack") + xlab(target) + ylab('famhist')
ggplot(train_data, aes(x = train_data[, target], fill = famhist)) + geom_bar(position = position_dodge(preserve = "single")) + xlab(target) + ylab('famhist')
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Encoding Categorical Variable--------------------------------------------------------------------------------------------------------------------------------------------------------
# Following are the two methods to encode categorical variable.
# Option 1 : Label Encoding
# Option 2 : One Hot Encoding

# Option 1
# Label Encoding : This is the simplest form of encoding where each value is converted to an integer.The maximum value is the equal to the number of unique values of the variable.

# Follwing function takes a single categorical column and label encodes it
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  return(x)
}

# Following function calls the above function for each categorical column and label encodes them
encode_label <- function(df) {
  for (var in colnames(df)) {
    if (class(df[, var]) %in% c("character", "factor") &&
        var != uuid_col & var != target) {
      df[, var] <- encode_ordinal(df[, var])
    }
  }
  return(df)
}


# Option 2
# One hot encoding : This is probably the most common form of encoding and is often referred to as creating dummy or indicator variables. It creates a new column for each unique
# value of the categorical variable. Each of these columns are binary with values 1 or 0 depending on whether the value of the variable is equal to the unique value being encoded by this column.

# Following function ctakes each categorical column in a dataframe and  one hot encodes them  and returns the one hot encoded dataframe
encode_one_hot <- function(df) {
  cols = c()
  for (var in colnames(df)) {
    if (class(df[, var]) %in% c("character", "factor") &&
        var != uuid_col & var != target) {
      cols = c(cols, var)
    }
  }
  df = dummy.data.frame(df, names = cols, sep = "_")
  names(df) <- make.names(names(df))
  return(df)
}

# Choose any one-> set encode_type

encode_type = 2 # 1 for option 1, 2 for option 2

# Depending on whatever option you choose earlier the following function encodes the categorical columns accordingly
encode_categorical <- function(encode_type, df) {
  if (encode_type == 1) {
    return(encode_label(df))
  }
  if (encode_type == 2) {
    return(encode_one_hot(df))
  }
}

train_data = encode_categorical(encode_type , train_data)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Data Standardization------------------------------------------------------------------------------------------------------------------------------------------------------
# Do this for dataset that has multiple features spanning varying degrees of magnitude, range, and units. This is a significant obstacle as
# a few machine learning algorithms are highly sensitive to these features. Distance based algorithms like KNN, K-means, and SVM are most affected
# by the range of features. This is because behind the scenes they are using distances between data points to determine their similarity.
# Tree-based algorithms, on the other hand, are fairly insensitive to the scale of the features. If you don't perform standardization
# the algorithm takes more time(iterations) to arrive at optimum point(i.e the best classifier), which isn't considered a good practice in data science circles.
# There are many approaches(eg: Stndardization, Normalization etc.). We are going to use Stnndardization.
# Standardization is a scaling technique where the values are centered around the mean with a unit standard deviation.
# This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.

# Following function takes a single numeric/integer column and standardize it using its mean and sd
standardizeColumn <- function(df, col) {
  new_col_vec <- (df[, col] - mean(df[, col],,na.rm = TRUE)) / sd(df[, col],na.rm = TRUE)
  return(new_col_vec)
}

# Following function calls the above function for each numeric/integer column in dataset and standardizes it
standardizeDataFrame <- function(df) {
  for (var in colnames(df)) {
    if (class(df[, var]) %in% c("numeric", "integer")
        && var %notin% c(uuid_col, target)) {
      df[, var] <- standardizeColumn(df, var)
    }
  }
  return(df)
}
train_data = standardizeDataFrame(train_data)

#-------------------------------------------------------------------------------------------------------------------------------------------------------


# Train/Test Split------------------------------------------------------------------------------------------------------------------------
#For any model building process daataset is divided into 3 parts:
  
# 1. Train
# 2. Validation
# 3. Test

# A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters.
# The validation dataset is different from the test dataset that is also held back from the training of the model, but is instead used to give an unbiased estimate 
# of the skill of the final tuned model when comparing or selecting between final models.

# Validation dataset is used for model tuning, which most likely you won't be doing in your exams, and Test set is used to evaluate your model.So, for your purposes 
# test and validation set are one and the same thing. In your exams most likely you will be given one file. Divide it in two parts train/val set(demonstrated in code)

# In problems where you have to perform classification chances are that there is class imbalance(Hopefully in exam you won't see this). In such cases you should perform
# stratifed train test split because you want similar representaion of classes in train and test/val set

set.seed(99)
train.index <- createDataPartition(train_data[, target], p = .7, list = FALSE)
X_train <- train_data[train.index, which(names(train_data) %notin% c(uuid_col, target))] # Train data
y_train <- train_data[train.index, target] # Train labels
X_val  <- train_data[-train.index, which(names(train_data) %notin% c(uuid_col, target))] # Val data
y_val <- train_data[-train.index, target] #Val labels
##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Check target encoding-------------------------------------------------------------------------------------------------------------------------------------------------------

target_encoding <- contrasts(y_train)
target_codes <- row.names(target_encoding)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Logistic Regression
set.seed(31)
model.logit <-glm (y_train ~ .,
                   data = X_train,
                   family = binomial,
                   control = glm.control(maxit = 25)
                   )
summary(model.logit)
# Check which features were the best predictors
varImp(model.logit)

preds.logit <- model.logit %>% predict(X_val, type = "response")  ##probablities
preds.classes.logit <- ifelse(preds.logit > 0.5, target_codes[2], target_codes[1])

# Confusion Matrix
cm.logit = confusionMatrix(y_val, as.factor(preds.classes.logit))
print(cm.logit)

# Plot ROC curve
pred_obj <- prediction(preds.logit, y_val)
auc.logit <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.logit)

roc.perf.logit <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(
  roc.perf.logit,
  col = rainbow(10),
  colorize = TRUE ,
  main = "Logistic Regression"
)
abline(a = 0, b = 1)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------



# The following algorithms are from the Decision Tree Family

# Decision Tree
# Complexity parameter is the tunable parameter. It's related to the depth to which you have grown the tree.
# This value represents the optimum value at which you get the best value of the metric(eg: Accuracy , AUC) used during CV. We will plot this against the depeth of tree in next steps to get a better idea
set.seed(53)
model.dt <- rpart(y_train ~ .,
                  data = X_train,
                  method = "class")
summary(model.dt)
printcp(model.dt)
plotcp(model.dt) # Plots complexity parameter and the corresponding depth of the tree against Relative error

# Check at what cp in xerror in minimum and using that build the model again
set.seed(57)
model.dt <- rpart(
  y_train ~ .,
  data = X_train,
  method = "class",
  control = rpart.control(cp = 0.012)
)

# Check which features were the best predictors
model.dt$variable.importance

preds.dt <-model.dt %>% predict(X_val, type = "prob") ## probablities
preds.classes.dt <- ifelse(preds.dt[, 2] > 0.5, target_codes[2], target_codes[1])
rpart.plot(model.dt,box.palette = "BuGn",shadow.col = "gray",nn = TRUE,main = "Decision Tree")

# Confusion Matrix
cm.dt = confusionMatrix(y_val, as.factor(preds.classes.dt))
print(cm.dt)

# Plot ROC Curve
pred_obj <- prediction(preds.dt[, 2], y_val)
auc.dt <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.dt)
roc.perf.dt <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(roc.perf.dt,
     col = rainbow(10),
     colorize = TRUE,
     main = "Decision Tree")
abline(a = 0, b = 1)
#----------------------------------------------------------------------------------------------------------------------------------------------------




# Random Forest---> Builds n-number of independent trees and combines them using majority predictiosns
model.rf <- randomForest(y_train ~ .,
                         data = X_train,
                         importance = TRUE,
                         ntree = 100)

print(model.rf)

# Check which features were the best predictors(use the plot drawn below)
importance(model.rf)
varImpPlot(model.rf,type = 2)

preds.rf <- model.rf %>% predict(X_val, type = "prob") ## probablities
preds.classes.rf <- as.factor(ifelse(preds.rf[, 2] > 0.5, target_codes[2], target_codes[1]))

# Confusion Matrix
cm.rf = confusionMatrix(y_val, preds.classes.rf)
print(cm.rf)

# Plot ROC Curve
pred_obj <- prediction(preds.rf[, 2], y_val)
auc.rf <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.rf)
roc.perf.rf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(roc.perf.rf,
     col = rainbow(10),
     colorize = TRUE,
     main = "Random Forest")
abline(a = 0, b = 1)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Bagging --> This also builds independent trees.
# For each tree it samples data from whole dataset using a sampling techinque known as bootstap aggregation i.e. bagging
model.bag <- bagging(
  y_train ~ .,
  data = X_train,
  nbagg = 100,
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)
print(model.bag)

# Check which features were the best predictors 
varImp(model.bag)

preds.bag <- model.bag %>% predict(X_val, type = "prob") ## probablities
preds.classes.bag <- as.factor(ifelse(preds.bag[, 2] > 0.5, target_codes[2], target_codes[1]))

# Confusion Matrix

cm.bag = confusionMatrix(y_val, preds.classes.bag)
print(cm.bag)

# Plot ROC Curve
pred_obj <- prediction(preds.bag[, 2], y_val)
auc.bag <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.bag)
roc.perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(roc.perf,
     col = rainbow(10),
     colorize = TRUE,
     main = "Bagged Tree")
abline(a = 0, b = 1)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# The following algorithm is also a variant of decision tree but uses a technique known as boosting
# Gradient Boosting
set.seed(31)
model.gbm <- gbm(as.character(factor(y_train, labels=c("0","1"))) ~ .,
                 data = X_train,
                 distribution = 'bernoulli',
                 n.trees = 100
                 )
print(model.gbm)

# Check which features were the best predictors 
summary(model.gbm)

preds.gbm <- model.gbm %>% predict(X_val, type = "response") ## probablities
preds.classes.gbm <- as.factor(ifelse(preds.gbm > 0.5, target_codes[2], target_codes[1]))

# Confusion Matrix

cm.gbm = confusionMatrix(y_val, preds.classes.gbm)
print(cm.gbm)

# Plot ROC Curve
pred_obj <- prediction(preds.gbm, y_val)
auc.gbm <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.gbm)
roc.perf <-
  performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(roc.perf,
     col = rainbow(10),
     colorize = TRUE,
     main = "Gradient Boosting")
abline(a = 0, b = 1)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# KNN-------------------------------------------------------------------------------------------------------------------------------------------
# The following for-loop will give you the optimum value of K
set.seed(37)
i=1
k.optm=1
for (i in 1:28){
   knn.mod <- knn(train=X_train, test=X_val, cl=y_train, k=i)
   k.optm[i] <- 100 * sum(y_val == knn.mod)/NROW(y_val)
   k=i
   cat(k,'=',k.optm[i],'')
}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

preds.classes.knn <- knn(train=X_train, test=X_val, cl=y_train, k=which.max(k.optm))
# There are no probabilities for such K-nearest classification method because it is discriminative classification

# Confusion Matrix
cm.knn = confusionMatrix(y_val, preds.classes.knn)
print(cm.knn)
#---------------------------------------------------------------------------------------------------------------------------



#LDA------------------------------------------------------------------------------------------------------------------------------
set.seed(59)
model.lda <- lda(y_train~., data = X_train)
preds.lda <- model.lda %>% predict(X_val)
preds.classes.lda <- as.factor(ifelse(preds.lda$posterior[,2] > 0.5, target_codes[2],target_codes[1]))
# Confusion Matrix

cm.lda = confusionMatrix(y_val, preds.classes.lda)
print(cm.lda)

# Plot ROC Curve
pred_obj <- prediction(preds.lda$posterior[,2], y_val)
auc.lda <- as.numeric(performance(pred_obj, "auc")@y.values)
print(auc.lda)
roc.perf.lda <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(roc.perf.lda,
     col = rainbow(10),
     colorize = TRUE,
     main = "Linear Discriminant Analysis")
abline(a = 0, b = 1)


# Models Evaluation-----------------------------------------------------------------------------------------------------------------------------------------

# Accuracy
accuracy_list <- c(cm.logit$overall['Accuracy'],cm.dt$overall['Accuracy'],cm.rf$overall['Accuracy'],
                      cm.bag$overall['Accuracy'],cm.gbm$overall['Accuracy'],cm.knn$overall['Accuracy'],cm.lda$overall['Accuracy'])*100

classifier_list_accuracy <- c("Logistic Regression","Decision Tree","Random Forest","Bagged Trees","GBM","KNN",'LDA')
df.accuracy <- as.data.frame(cbind(classifier_list_accuracy,accuracy_list))
ggplot(data=df.accuracy, aes(x=classifier_list_accuracy, y=accuracy_list)) + geom_bar(stat="identity", width=0.5,fill="steelblue") + xlab('Classifiers') + ylab("Accuracy") +theme_minimal()



# Plot the ROC curves for all models -> This will give you an idea which model was the best
preds_list <- list(preds.logit, preds.dt[, 2], preds.rf[, 2], preds.bag[, 2], preds.gbm,preds.lda$posterior[,2])
m <- length(preds_list)
actuals_list <- rep(list(y_val), m)
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Val Set ROC Curves")
legend(
  x = "bottomright",
  legend = c(
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Bagged Trees",
    "GBM",
    "LDA"
  ),
  fill = 1:m
)
abline(a = 0, b = 1)

# AUC(Area Under the ROC curve)
auc_list <- c(auc.logit,auc.dt,auc.rf,auc.bag,auc.gbm,auc.lda)
classifier_list_auc <- c("Logistic Regression","Decision Tree","Random Forest","Bagged Trees","GBM",'LDA')
df.auc <- as.data.frame(cbind(classifier_list_auc,auc_list))
ggplot(data=df.auc, aes(x=classifier_list_auc, y=auc_list)) + geom_bar(stat="identity", width=0.5,fill="steelblue") + xlab('Classifiers') + ylab("AUC") +theme_minimal()


# Must Read
# Why accuracy can be a misleading metric?
# The most commonly reported measure of classifier performance is accuracy: the percent of correct classifications obtained.
# This metric has the advantage of being easy to understand and makes comparison of the performance of different classifiers 
# trivial, but it ignores many of the factors which should be taken into account when honestly assessing the performance of 
# a classifier.

# Classifier performance is more than just a count of correct classifications.Consider, for interest, the problem of screening 
# for a relatively rare condition such as cervical cancer, which has a prevalence of about 10% (actual stats). If a lazy Pap
# smear screener was to classify every slide they see as “normal”, they would have a 90% accuracy. Very impressive! 
# But that figure completely ignores the fact that the 10% of women who do have the disease have not been diagnosed at all.

# Why ROC curves serve as a better metric?
# ROC curve is a plot between the True Positive Rate and False Positive Rate.For a perfect classifier the ROC curve will go straight up the Y axis and then along the X axis.
# A classifier with no power will sit on the diagonal, whilst most classifiers fall somewhere in between. It is immediately apparent that a ROC curve can be used to select 
# a threshold for a classifier which maximises the true positives, while minimising the false positives. However, different types of problems have different optimal classifier 
# thresholds. For a cancer screening test, for example, we may be prepared to put up with a relatively high false positive rate in order to get a high true positive,  it is most 
# important to identify possible cancer sufferers.For a follow-up test after treatment, however, a different threshold might be more desirable, since we want to minimise 
# false negatives, we don’t want to tell a patient they’re clear if this is not actually the case. ROC curves also give us the ability to assess the performance of the 
# classifier over its entire operating range. The most widely-used measure is the area under the curve (AUC). The AUC can be used to compare the performance of two or more classifiers.
# A single threshold can be selected and the classifiers’ performance at that point compared, or the overall performance can be compared by considering the AUC.

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run this part only if a seperate test data is provided

test_data <-read.csv(file.choose(),na.strings = c("", "NA"))
head(test_data, 10)
dim(test_data)
nullColumns(test_data)

X_test <- test_data[, which(names(test_data) %notin% c(uuid_col, target))]


X_test = dataImputation(X_test)
X_test = encode_categorical(encode_type ,X_test)
X_test = standardizeDataFrame(X_test)

#Prediction
# Choose Model

preds.test <- model.logit %>% predict(X_test, type = "response")  ##probablities
preds.classes <- ifelse(preds.test > 0.5, target_codes[2], target_codes[1])
test_data[,target] <- preds.classes
