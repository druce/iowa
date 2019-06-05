library(tidyverse)
library(mice)
library(e1071)
library(Metrics)
library(randomForest)
library(glmnet)

# Reading data
train <- read.csv("~/Notebooks/iowa/data/train.csv", stringsAsFactors = F, sep=',')
test <- read.csv("~/Notebooks/iowa/data/test.csv", stringsAsFactors = F, sep=',')

# Combining test and train data
full <- bind_rows(train,test)
SalePrice <- train$SalePrice
N <- length(SalePrice)
Id <- test$Id
full[,c('Id','SalePrice')] <- NULL
rm(train,test)

# Converting predictors to factor or integer
chr <- full[,sapply(full,is.character)]
int <- full[,sapply(full,is.integer)]
fac <- chr %>% lapply(as.factor) %>% as.data.frame()
full <- bind_cols(fac,int)
# Running MICE based on random forest
micemod <- full %>% mice(method='rf')
full <- complete(micemod)
# Saving train and test sets
train <- full[1:N,]
test<-full[(N+1):nrow(full),]

write.csv(train,"~/train_imputed.csv",row.names = F)

write.csv(test,"~/test_imputed.csv",row.names = F)

# Adding dependent variable
train <- cbind(train,SalePrice)

# Modelling: SVM
svm_model <- svm(SalePrice~., data=train, cost = 3.2)
svm_pred_train <- predict(svm_model,newdata = train)
sqrt(mean((log(svm_pred_train)-log(train$SalePrice))^2))
svm_pred <- predict(svm_model,newdata = test)

# Writing final predictions to CSV file
solution <- data.frame(Id=Id,SalePrice=svm_pred)
write.csv(solution,"~/svm_solution_32.csv",row.names = F)
