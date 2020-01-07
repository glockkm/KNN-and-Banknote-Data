#MSDS 5213 Lab 1 KNN
#Kimberly Glock
#November 5, 2019

#https://towardsdatascience.com/k-nearest-neighbors-algorithm-with-examples-in-r-simply-explained-knn-1f2c88da405c

library(class)
library(caret)
library(e1071)

#http://archive.ics.uci.edu/ml/datasets/banknote+authentication
#Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
#For digitization, an industrial camera usually used for print inspection was used. 
#The final images have 400x 400 pixels. 
#Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. 
#Wavelet Transform tool were used to extract features from images

#1. variance of Wavelet Transformed image (continuous)
#2. skewness of Wavelet Transformed image (continuous)
#3. curtosis of Wavelet Transformed image (continuous)
#4. entropy of image (continuous)
#5. class (integer) 

bank = read.csv("banknote.csv", head=FALSE, sep=",")
dim(bank)
summary(bank)

preProcess(bank, method=c("center", "scale"))
# center data
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/preProcess

split_size = 0.8
train_size = floor(nrow(bank) * split_size)
#https://www.tutorialgateway.org/r-floor-function/
set.seed(123)
train_data_pool = sample(1:nrow(bank), train_size)

class_var = bank[train_data_pool,]$V5 #target from training data for knn function cl argument
class_var2 = bank[-train_data_pool,]$V5 #target variable from testing data for accuracy/cm
# pull out only class/target variable
View(class_var)

train = subset(bank[train_data_pool,], select=-V5)
#####train = subset(bank[train_data_pool,]) #???makes error in knn as dims differ between test and train data
test = subset(bank[-train_data_pool,], select=-V5)
# extract train and test data without class column
best_k = train(train, class_var, method="knn", 
               tuneGrid=data.frame(.k = 1:20),
               trControl = trainControl(method="cv"))
# uses caret library to find optimal number of k
#https://www.rdocumentation.org/packages/caret/versions/6.0-84/topics/train
best_k


#####pred = predict(best_k, test)
#https://stats.stackexchange.com/questions/245385/how-to-make-a-confusion-matrix-when-testing-a-model-on-data-with-only-positive-c
# test best model using test data
#pred2 = knn(train, test, cl = class_var, k=2, prob = FALSE, use.all = TRUE)
pred3 = knn(train, test, cl = class_var, k=3, prob = FALSE, use.all = TRUE)
#https://www.rdocumentation.org/packages/class/versions/7.3-15/topics/knn

#confusionMatrix(pred, bank[-train_data_pool,]$V5)
#######confusionMatrix(factor(pred2), factor(bank[-train_data_pool,]$V5))
confusionMatrix(factor(pred3), factor(class_var2))
#confusionMatrix(pred2, bank[-train_data_pool,]$V5)


tab = table(pred2, class_var2)
accuracy = function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)


# to save and reuse model in future
#https://www.rdocumentation.org/packages/base/versions/3.6.1/topics/readRDSjkjkjkjkjkjkjkjkjkx