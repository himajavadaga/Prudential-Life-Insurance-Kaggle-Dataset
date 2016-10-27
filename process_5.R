#
# Prudential Dataset
# - Midterm - Team 4
#
#install.packages("readr")
library(caret)
library(readr)

#helps the user to pick the file
x=file.choose(new = FALSE)
#importing the file
train_sample<-read.csv(x, header = TRUE)
#set.seed(1337)
# smp_size <- floor(0.9 * nrow(train))
# train_ind <- sample(seq_len(nrow(train)), size = smp_size)
# train_sample <- train[train_ind, ]
#

# Preprocessing
#Function for Count of NAs
no_of_na=function(vector){
  s=0
  for(i in 1:length(vector)){
    if(is.na(vector[i]==TRUE)) s=s+1
  }
  return(s)
}

sum(is.na(train_sample))

#Percentage of missigness per predictor(missingness rate)
C=data.frame(lapply(train_sample, no_of_na))*100/nrow(train_sample)
# C

#Excluding variables that have missingness rate over 50%
selected=subset(colnames(C),C[1,]<50)
train_sample<-train_sample[,selected]


# Fix NA's
feature.names <- names(train_sample)[2:(ncol(train_sample)-1)]
for (f in feature.names) {
  if (class(train_sample[[f]])=="integer" || class(train_sample[[f]])=="numeric") {
    median <- median(train_sample[[f]], na.rm = T)
    train_sample[[f]][is.na(train_sample[[f]])] <- median
  }
}


# Convert binary columns into 1-0
feat.bin <- c("Product_Info_6", "Product_Info_5", "Product_Info_1", "InsuredInfo_2", "Employment_Info_3",
              "Employment_Info_5",  "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
              "Insurance_History_1", "Medical_History_4", "Medical_History_22",
              "Medical_History_38")

for (f in feat.bin) {
  levels <- unique(train_sample[[f]])
  train_sample[[f]] <- as.integer(factor(train_sample[[f]], levels=levels))-1
}


# Convert character columns to ids
for (f in feature.names) {
  if (class(train_sample[[f]])=="character") {
    levels <- unique(train_sample[[f]])
    train_sample[[f]] <- as.integer(factor(train_sample[[f]], levels=levels))
  }
}


# Categorical Variables
catVars <- c('Product_Info_7', 'Product_Info_2',
             'InsuredInfo_1', 
             'Insurance_History_2',
             'Insurance_History_3', 'Insurance_History_4',
             'Insurance_History_7', 'Insurance_History_8',
             'Insurance_History_9', 'Family_Hist_1',
             'Medical_History_3', 'Medical_History_5',
             'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
             'Medical_History_9', 'Medical_History_11',
             'Medical_History_12', 'Medical_History_13', 'Medical_History_14',
             'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
             'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
             'Medical_History_23', 'Medical_History_25',
             'Medical_History_26', 'Medical_History_27', 'Medical_History_28',
             'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
             'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
             'Medical_History_36', 'Medical_History_37',
             'Medical_History_39', 'Medical_History_40', 'Medical_History_41')

#convert Categorical Variables to dummy variable
for (var in catVars) {
  #For every unique value in the string column, create a new 1/0 column
  #This is what Factors do "under-the-hood" automatically when passed to function requiring numeric data
  for(level in unique(train_sample[[var]])){
    train_sample[paste(var, level, sep = "_")] <- ifelse(train_sample[[var]] == level, 1, 0)
  }
  train_sample[[var]] <- NULL
}


#Write the processed file to csv
write.csv(train_sample, "processed2.csv")

#Check if ther are any NA's in the data
sum(is.na(train_sample))

# Split the data

smp_size <- floor(0.9 * nrow(train_sample))
train_ind <- sample(seq_len(nrow(train_sample)), size = smp_size)
train_data <-  train_sample[train_ind, ]
test_data = train_sample[-train_ind,]  #200 rows


# To find important features
set.seed(9)
# load the library
# install.packages("mlbench")
library(mlbench)
library(caret)
# load the dataset
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Response ~., data=train_data, method="lm",preProcess="scale", trControl=control)
# estimate variable importance

?train
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
plot(importance)


# Running LR on the entire dataset to find the important features

test <- lm(Response ~., data=train_data)
summary(test)


# After selecting the sifnificant features from both the models, build the 3 different models


# ********************************************************
# 1. LINEAR REGRESSION  
# ********************************************************

train.fit <- lm(Response ~ Product_Info_2_10+ Product_Info_2_12+ 
                  BMI	+Employment_Info_3	+Family_Hist_2	+Family_Hist_4	+Ht	+Ins_Age	+Insurance_History_1	
                +InsuredInfo_2	+InsuredInfo_5	+InsuredInfo_6	+InsuredInfo_7	+Medical_History_1	
                +Medical_History_11_1	+Medical_History_11_3	+Medical_History_22	+Medical_History_30_2	
                +Medical_History_31_3	+Medical_History_35_1	+Medical_History_39_3	+Medical_History_4	
                +Medical_History_40_3	+Medical_History_7_1+Medical_Keyword_9 +Medical_Keyword_15	+Medical_Keyword_19	+Medical_Keyword_2	
                +Medical_Keyword_22	+Medical_Keyword_25	+Medical_Keyword_3	+Medical_Keyword_33
                +Medical_Keyword_34+Medical_Keyword_37+Medical_Keyword_38	+Medical_Keyword_39	+Medical_Keyword_41	+Medical_Keyword_45
                +Medical_Keyword_6	+Medical_Keyword_9	+Product_Info_4	+Product_Info_6	+Wt, data = train_data)

summary(train.fit)
plot(train.fit)
library(forecast)
prediction <- predict(train.fit, test_data)
head(prediction)
head(test_data$Response)
SSE <- sum((test_data$Response - prediction) ^ 2)
SST <- sum((test_data$Response - mean(test_data$Response)) ^ 2)
1 - SSE/SST
library(Metrics)
kap <- ScoreQuadraticWeightedKappa(round(prediction),test_data$Response,1,8)
kap



#*************************************************
# 2. Decision Tree Regressor
#*************************************************

#

library(forecast)
library(rpart)
library(XLConnect)

attach(train_data)

#  Grow the tree
tree_fit_I <- rpart(Response ~ 
                      Product_Info_2_10+ Product_Info_2_12+ 
                      BMI	+Employment_Info_3	+Family_Hist_2	+Family_Hist_4	+Ht	+Ins_Age	+Insurance_History_1	
                    +InsuredInfo_2	+InsuredInfo_5	+InsuredInfo_6	+InsuredInfo_7	+Medical_History_1	
                    +Medical_History_11_1	+Medical_History_11_3	+Medical_History_22	+Medical_History_30_2	
                    +Medical_History_31_3	+Medical_History_35_1	+Medical_History_39_3	+Medical_History_4	
                    +Medical_History_40_3	+Medical_History_7_1+Medical_Keyword_9 +Medical_Keyword_15	+Medical_Keyword_19	+Medical_Keyword_2	
                    +Medical_Keyword_22	+Medical_Keyword_25	+Medical_Keyword_3	+Medical_Keyword_33
                    +Medical_Keyword_34+Medical_Keyword_37+Medical_Keyword_38	+Medical_Keyword_39	+Medical_Keyword_41	+Medical_Keyword_45
                    +Medical_Keyword_6	+Medical_Keyword_9	+Product_Info_4	+Product_Info_6	+Wt
                    ,method = "anova", data=train_data, control=rpart.control(minsplit = 10, cp=0.01, maxdepth=10))
summary(tree_fit_I)
printcp(tree_fit_I)

# Predict the ouput
predictionsI <- predict(tree_fit_I,test_data)
predict_trainI <- predict(tree_fit_I,train_data)


min(predictionsI)
max(predictionsI)

round(predictionsI)
aI <- accuracy(predictionsI, test_data$Response)
aI
bI <- accuracy(predict_trainI, train_data$Response)
bI

table(predictionsI,test_data$Response)

bestcp <- tree_fit_I$cptable[which.min(tree_fit_I$cptable[,"xerror"]),"CP"]
bestcp


# Prune the tree
pfit<- prune(tree_fit_I, cp=bestcp) # from cptable   

# Plot the pruned tree
plot(pfit, uniform=TRUE, 
     main="Pruned Regression Tree for Prudential")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
post(pfit, file = "D:/NEU Course Materials/Big Data Analytics/prudential.ps", 
     title = "Pruned Regression Prudential")

summary(pfit)

#prediction after pruning
pred_prune <- predict(pfit,test_data)

head(pred_prune)

#RMSE
RMSE.rtree.prune <- sqrt(mean((pred_prune-test_data$Response)^2))
RMSE.rtree.prune

#MAE
MAE.rtree.prune <- mean(abs(pred_prune-test_data$Response))
MAE.rtree.prune

#Accuracy after pruning
accuracy(pred_prune,test_data$Response)

library(Metrics)
# 
# length(predictionsI)
# head(predictionsI)
# nrow(predictionsI)
# length(test_data$Response)

kap_tree <- ScoreQuadraticWeightedKappa(round(pred_prune),test_data$Response,1,8)
kap_tree


#**********************************************
# 3. SVM  
#**********************************************

##########SVM#########

## choosing only the 10% of the data to train for the SVM model
tendata <- train_sample[sample(1:nrow(train_sample), 5938,replace=FALSE),]

#training the data
smp_size_svm <- floor(0.75 * nrow(tendata))
train_ind_svm <- sample(seq_len(nrow(tendata)), size = smp_size_svm)
train_data_svm <-  tendata[train_ind_svm, ]
test_data_svm = tendata[-train_ind_svm, ]  #200 rows

library(e1071) 
library(rpart)
data(train_ind_svm, package="mlbench")

#svm model
model = svm(Response ~ BMI	+Employment_Info_3	+Family_Hist_2	+Family_Hist_4	+Ht	+Ins_Age	+Insurance_History_1	
            +InsuredInfo_2	+InsuredInfo_5	+InsuredInfo_6	+InsuredInfo_7	+Medical_History_1	
            +Medical_History_11_1	+Medical_History_11_3	+Medical_History_22	+Medical_History_30_2	
            +Medical_History_31_3	+Medical_History_35_1	+Medical_History_39_3	+Medical_History_4	
            +Medical_History_40_3	+Medical_History_7_1+Medical_Keyword_9 +Medical_Keyword_15	+Medical_Keyword_19	+Medical_Keyword_2	
            +Medical_Keyword_22	+Medical_Keyword_25	+Medical_Keyword_3	+Medical_Keyword_33
            +Medical_Keyword_34+Medical_Keyword_37+Medical_Keyword_38	+Medical_Keyword_39	+Medical_Keyword_41	+Medical_Keyword_45
            +Medical_Keyword_6	+Medical_Keyword_9	+Product_Info_4	+Product_Info_6	+Wt, kernel = "linear", cost = 1, gamma = 1, data = train_data_svm, scale = F)

# predicting the data from the svm model
predictions <-  predict(model, test_data_svm)

kap <- ScoreQuadraticWeightedKappa(round(predictions),test_data_svm$Response,1,8)
kap

library(Metrics)


#function to calculate RMSE
rmse <- function(error)
{
  sqrt(mean(error^2))
}
error <- train_data_svm$Response - predictions
svmPredictionRMSE <- rmse(error) 

#Besttune method
obj = best.tune(svm, Response ~ BMI	+Employment_Info_3	+Family_Hist_2	+Family_Hist_4	+Ht	+Ins_Age	+Insurance_History_1	
                +InsuredInfo_2	+InsuredInfo_5	+InsuredInfo_6	+InsuredInfo_7	+Medical_History_1	
                +Medical_History_11_1	+Medical_History_11_3	+Medical_History_22	+Medical_History_30_2	
                +Medical_History_31_3	+Medical_History_35_1	+Medical_History_39_3	+Medical_History_4	
                +Medical_History_40_3	+Medical_History_7_1+Medical_Keyword_9 +Medical_Keyword_15	+Medical_Keyword_19	+Medical_Keyword_2	
                +Medical_Keyword_22	+Medical_Keyword_25	+Medical_Keyword_3	+Medical_Keyword_33
                +Medical_Keyword_34+Medical_Keyword_37+Medical_Keyword_38	+Medical_Keyword_39	+Medical_Keyword_41	+Medical_Keyword_45
                +Medical_Keyword_6	+Medical_Keyword_9	+Product_Info_4	+Product_Info_6	+Wt, data = train_data_svm, kernel =
                  "linear")
