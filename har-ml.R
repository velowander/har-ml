Original Research Citation:
# Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H.
# Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements.
# Proceedings of 21st Brazilian Symposium on Artificial Intelligence.
#Advances in Artificial Intelligence - SBIA 2012.
#In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012.
#ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

#Read more: http://groupware.les.inf.puc-rio.br/har#ixzz37WRMx5p1

#Remove predictors that are not sensor readings
#original research paper http://groupware.les.inf.puc-rio.br/har
#http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
#Issue: using OOB (out of bag) vs CV (cross validation)
#bagging methods for train(): bagEarth, treebag, bagFDA

listFactorColumn <- function(df) {
  #df - data frame to analyze
  out <- integer(0)
  for (i in 1:ncol(df)) if (is.factor(df[, i])) {
    out[length(out) + 1] = i
  }
  out
}

listColumnNA <- function(df) {
  #df - data frame to analyze
  #returns numeric proportions of NAs for each column
  out <- integer(0)
  for (i in 1:ncol(df)) out[length(out) + 1] = sum(is.na(df[, i]))/nrow(df)
  out
}

listUnacceptableColumnNA <- function(df) {
  #df - data frame to analyze
  #returns numeric indices of columns with NA proportions above thresh
  thresh <- 0.9
  out <- logical(0)
  for (i in 1:ncol(df)) {
    if (sum(is.na(df[, i]))/nrow(df) > thresh) out[length(out) + 1] = i
  }
  out
}

pml_write_files = function(x){
  #supplied function for creating all the submission files
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

library(caret)

fileName = c("pml-training", "pml-testing")
fileUrl = c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
            "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
for (i in 1:2) if (!file.exists(fileName[i])) download.file(fileUrl[i], fileName[i], method="curl")

#Load data
train.raw <- read.csv(fileName[1])[ , -c(1:7)]

#Preprocess
set.seed(32343)
train.sample <- train.raw[sample(nrow(train.raw), 7000), ]
classe <- train.sample$classe
train.sample <- train.sample[, -listFactorColumn(train.sample)]
train.sample <- train.sample[, -listUnacceptableColumnNA(train.sample)]

#train.sample <- data.frame( sapply(train.sample [ , -ncol(train.sample)], as.numeric) )
#train.sample[ , apply(train.sample[, -153], 2, var, na.rm=TRUE) != 0]
if (is.null(train.sample$classe)) train.sample <- cbind(classe = classe, train.sample)
#if (is.null(train.sample$classe)) train.sample <- as.data.frame(lapply(train.sample,as.numeric))

#For PCA:
preProc <- preProcess(train.sample[, -1], method=c("center", "scale", "pca"), thresh = 0.95)
trainPC <- predict(preProc, train.sample[, -1])
trainPC <- cbind(classe, trainPC)

#inTrain = createDataPartition(train.sample$classe, p = 0.7)[[1]]
inTrain = createDataPartition(trainPC$classe, p = 0.7, list = FALSE)
training.pc = trainPC[ inTrain, ]
training = train.sample[ inTrain, ]
validation.pc = trainPC[ -inTrain, ]
validation = train.sample [ -inTrain, ]
testing <- read.csv(fileName[2])

#also consider randomForest() instead of train(method = 'rf')...
#also consider train(training$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trControl, data=training)

#http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
#In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error.
#better use cv because rubric asks did we use cross validation for estimating errors
modFit = train(as.factor(classe) ~ . , data = training, method = 'rf', trControl = trainControl())
modFit.cv = train(as.factor(classe) ~ . , data = training,
               method = 'rf', trControl = trainControl(method='cv'))
modFit.pc <- train(as.factor(classe) ~ ., data = training.pc, method = "rf",
                     trControl = trainControl(method='cv'))
modFit.pc2 <- train(as.factor(classe) ~ ., data = training, method = "rf",
                    preProcess=c("center", "scale", "pca"), trControl = trainControl(method='cv'))
predict.cv <- predict(modFit.cv, validation[ , -1])
predict.pc <- predict(modFit.pc, validation.pc[ , -1])
predict.pc2 <- predict(modFit.pc2, validation[, -1])
confusionMatrix(predict.pc, validation.pc$classe)$overall
confusionMatrix(predict.cv, validation$classe)$overall
confusionMatrix(predict.pc2, validation$classe)$overall

predict.test <- predict(modFit.cv, testing[, -1])
pml_write_files(as.character(predict.test))
