# 1. Overview
## 1.1 Objective
### Predict survival on the Titanic

## 1.2 VARIABLE DESCRIPTIONS:
### survival Survival (0 = No; 1 = Yes)
### pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
### name Name
### sex Sex
### sibsp Number of Siblings/Spouses Aboard
### parch Number of Parents/Children Aboard
### ticket Ticket Number
### fare Passenger Fare
### cabin Cabin
### embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# 2. Set up Project
## 2.1 Set Working Directory
setwd('C:/Users/m339673/Desktop/Titanic')


## 2.2 Load Libraries
library(xgboost)
library(adabag)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(DMwR)
library(ROCR)
library(class)
library(ggplot2)
library(caret)
library(doBy)
library(dplyr)
library(gbm)
library(caret)
library(reshape2)
## 2.3 Import Data
train<-read.csv("train.csv", stringsAsFactors=F)
test<-read.csv("test.csv", stringsAsFactors=F)


train$dataset<-'train'
test$dataset<-'test'

titanic<-dplyr::bind_rows(train,test) #Combine data into one set so the data can be manipulated before modeling


## 2.4 Set up my helper functions


### Calculate the mode of a character/numeric variable
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

### Funtion to calculate the AUC of an ROC curve
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

### Functions to measure accuracy, F1 and deviance to compare models
loglikelihood <- function(y, py) {      # Note: 3 
  pysmooth <- ifelse(py==0, 1e-12,
                     ifelse(py==1, 1-1e-12, py))
  
  sum(y * log(pysmooth) + (1-y)*log(1 - pysmooth))
}

accuracyMeasures <- function(pred, truth, name="model") { 
  dev.norm <- -2*loglikelihood(as.numeric(truth), pred)/length(pred)
  ctable <- table(truth=truth,
                  pred=(pred>0.5))                            
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- 2*precision*recall/(precision+recall)
  data.frame(model=name, accuracy=accuracy, f1=f1, dev.norm)
}

### Convenience Functions to give class probabilities for class and numeric variables
mkPredC <- function(outCol,varCol,appCol) { 
  pPos <- sum(outCol==pos)/length(outCol) 
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[as.character(pos)] 
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[as.character(pos),]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred 
}

mkPredN<-function(outCol, varCol,appCol) {
  cuts<-unique(as.numeric(quantile(varCol,
                                   probs=seq(0,1,0.1),na.rm=T)))
  varC<-cut(varCol,cuts)
  ValC<-cut(appCol,cuts)
  mkPredC(outCol,varC,ValC)
}

## 2.5 Set seed
set.seed(1234)


# 3. Understand and manipulating the variables

## 3.1 Classify Variables
outcome<-'Survived'
pos<-'1'

## 3.2 Summarize variables
summary(titanic) #Age, Cabin and Embarked have missing values.  Also, the minimum for Fare is 0, so we need to investigate these variables
table(titanic$Survived)
table(titanic$Embarked)

### Look at those with Free fare
titanic[titanic$Fare==0,]   #All embarked from S and are older or age is NA for some, most likely an entry error at that Port.

### Lets add the Mean fare by Embarked and Pclass
titanic$Fare[titanic$Fare == 0] <- NA  #First set to NA

Avg_fare<-summaryBy(Fare~Pclass+Embarked, data=titanic,
          FUN=c(mean), na.rm=TRUE)



titanic<-merge(x=titanic,y=Avg_fare, by.x=c("Pclass","Embarked"), by.y=c("Pclass","Embarked"), all.x=TRUE)

titanic$Fare[is.na(titanic$Fare)] <- titanic$Fare.mean

titanic<-within(titanic, rm(Fare.mean))


### Look at those with missing Age (177 missing values)
sum(is.na(titanic$Age))

Avg_age<-data.frame(Age.mean=mean(titanic$Age, na.rm=TRUE))

titanic<-merge(x=titanic,y=Avg_age, by = NULL)

titanic$Age[is.na(titanic$Age)] <- titanic$Age.mean

titanic<-within(titanic, rm(Age.mean))

### Look at missing Cabin (687 missing values from Summary)
sum(is.na(titanic$Cabin))  #0 NAs they are just missing

sum(titanic$Cabin=='') #687

titanic$Cabin <- sapply(titanic$Cabin, as.character)


### Set to Unknown
titanic$Cabin[titanic$Cabin == ''] <- 'Unknown' 


### Look at those with missing Embarked (2 missing values from Summaryy)
sum(titanic$Embarked=='') #2
Embarked.mode<-as.character(Mode(titanic$Embarked))
titanic$Embarked<- sapply(titanic$Embarked, as.character)
titanic$Embarked[titanic$Embarked == ''] <- Embarked.mode
table(titanic$Embarked)



### Extract Deck from Cabin
titanic$Deck <- substr(titanic$Cabin,1,1)
table(titanic$Deck) #Since there is few G and T, classify it with R
titanic$Deck[titanic$Deck == 'T']<-'R'
titanic$Deck[titanic$Deck == 'G']<-'R'

### Create Family Size
titanic$Family_Size<-titanic$SibSp+titanic$Parch+1

# Grab title from passenger names
titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)


# Reassign rare titles

titanic$Title[titanic$Title %in% c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer',
                                   'Lady','Major','Mlle', 'Mme','Rev','Sir', 'the Countess')]<-'Rare'

titanic$Title[titanic$Title == 'Ms']<-'Miss'

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(titanic[titanic$dataset=='train',], aes(x = Family_Size, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') #After 4, we see chance of survival is lower

titanic$Large_family<-ifelse(titanic$Family_Size>4,1,0)

# Discretize family size
titanic$FsizeD[titanic$Family_Size == 1] <- 'single'
titanic$FsizeD[titanic$Family_Size< 5 & titanic$Family_Size > 1] <- 'small'
titanic$FsizeD[titanic$Family_Size > 4] <- 'large'

# Show family size by survival using a mosaic plot
mosaicplot(table(titanic$FsizeD, titanic$Survived), main='Family Size by Survival', shade=TRUE)
mosaicplot(table(titanic$Sex, titanic$Survived), main='Sex by Survival', shade=TRUE)

#Find children vs. adult
titanic$Child[titanic$Age < 18] <- 'Child'
titanic$Child[titanic$Age >= 18] <- 'Adult'

#Mother vs. not Mother
titanic$Mother <- 'Not Mother'
titanic$Mother[titanic$Sex == 'female' & titanic$Parch > 0 & titanic$Age > 18 & titanic$Title != 'Miss'] <- 'Mother'

#Make variables factors into factors
factor_vars <- c('Pclass','Sex','Embarked', 'Survived', 'Cabin','Deck',
                'Title','Large_family', 'FsizeD', 'Child','Mother')

titanic[factor_vars] <- lapply(titanic[factor_vars], function(x) as.factor(x))

## 3.3 Make transformations of Numeric Variables
vars<-colnames(titanic)
catVars<- vars[sapply(titanic[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(titanic[,vars],class) %in% c('numeric','integer')]

selVarsC<-catVars[c(-3,-4,-6,-7,-8)]
selVarsN<-numericVars[c(-1)]


### Function to transform
create_funs <- function() {
  funs <- new.env()
  funs$non <- function(x){x}
  funs$sqr <- function(x) {x^2}
  funs$cube <- function(x){x^3}
  funs$sqrt <- function(x){ifelse(x<0,NA,sqrt(x))}
  funs$curt <-function(x){x^.3333}
  funs$log <- function(x){ifelse(x <= 0, NA, ifelse(x == 1, log(1.0001), log(x))) }
  funs$exp <- function(x){exp(x)}
  funs$tan <- function(x) {tan(x)}
  funs$sin <- function(x) {sin(x)} 
  funs$cos <- function(x) {cos(x)}
  funs$inv <- function(x) {ifelse(x==0,NA,1/x)}
  funs$sqi <- function(x) {ifelse(x==0,NA,1/(x^2))}
  funs$cui <- function(x) {ifelse(x==0,NA,1/(x^3))}
  funs$sqri <- function(x) {ifelse(x==0,NA,1/sqrt(x))}
  funs$curi <- function(x) {ifelse(x==0,NA,1/(x^.3333))}
  funs$logi <- function(x) {ifelse(x <= 0, NA, ifelse(x == 1, 1/log(1.0001), 1/log(x))) }
  return(funs)
}

funs <- create_funs()



### 2 iteration stepwise regression to find best forms of the transformation
newvars<-character()

for (i in (1:length(selVarsN))){
  x<-titanic[,selVarsN[i]]
  res <- lapply(funs, function(f) {f(x)})
  df <- as.data.frame(res)
  names(df) <- paste(selVarsN[i],'_',names(funs), sep='')
  tst<-cbind(df,Survived=titanic$Survived)
  tst2<-tst[complete.cases(tst),]
  vars<-setdiff(colnames(tst2),list(outcome))
  fm<-paste(outcome,' ==1 ~ ',paste(vars,collapse=' + '),sep='')
  fm2<-paste(outcome,'==1 ~ ',1,sep='')
  nothing <- glm(fm2,data=tst2,family=binomial(link="logit"))
  step.model<-step(nothing, direction = "forward", trace = 0, scope = fm, steps=2)
  #merge titanic with df
  titanic<-cbind(titanic, df[setdiff(rownames(data.frame(step.model$coef)), list('(Intercept)'))])
  newvars<-append(newvars, setdiff(rownames(data.frame(step.model$coef)), list('(Intercept)'))) #new coefficients are stored in a variable
  }


#selVars<-union(selVarsC, numericVars)
selVars<-union(selVarsC, newvars)
dataVars<-union(selVars, outcome)

### Pull out test set
test<-titanic[titanic$dataset=='test',]


### Split training dataset into training and calibration
train<-titanic[titanic$dataset=='train',]
train<-train[,colnames(train) %in% (dataVars)]


train$gp<-runif(dim(train)[1])
train_new<-subset(train, train$gp>.3)
Cal<-subset(train, train$gp<=.3)


train_new<-within(train_new, rm(gp))
Cal<-within(Cal, rm(gp))



# 4. Begin Modeling (Use train to build and Cal to calibrate)

catVars<- selVars[sapply(train_new[,selVars],class) %in% c('factor','character')]
numericVars<- selVars[sapply(train_new[,selVars],class) %in% c('numeric','integer')]

### check output Class distribution for new training and calibration sets
table(train_new[,outcome])[2]/dim(train_new)[1]  #0.362851
table(Cal[,outcome])[2]/dim(Cal)[1]  #0.4065421 

# #Attempt to build a decision tree
### Variable selection by picking variables with deviance improvement on Calibration dataset
logLikelyhood <- function(outCol,predCol) {
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(Cal[,outcome],
                               sum(Cal[,outcome]==pos)/length(Cal[,outcome]))


for(v in catVars) { 
  pi <- paste('pred',v,sep='')
  train_new[,pi] <- mkPredC(train_new[,outcome],train_new[,v],train_new[,v])
  test[,pi] <- mkPredC(train_new[,outcome],train_new[,v],test[,v])
  Cal[,pi] <- mkPredC(train_new[,outcome],train_new[,v],Cal[,v])
  liCheck <- 2*((logLikelyhood(Cal[,outcome],Cal[,pi]) -
                   baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


for(v in numericVars) { 
  pi <- paste('pred',v,sep='')
  train_new[,pi] <- mkPredN(train_new[,outcome],train_new[,v],train_new[,v])
  test[,pi] <- mkPredN(train_new[,outcome],train_new[,v],test[,v])
  Cal[,pi] <- mkPredN(train_new[,outcome],train_new[,v],Cal[,v])
  liCheck <- 2*((logLikelyhood(Cal[,outcome],Cal[,pi]) -
                   baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


selVars_nopred<-gsub("pred", "", selVars) #Remove the pred prefix





### Using selected variables
fV <- paste(outcome,'==1 ~ ',
            paste(selVars,collapse=' + '),sep='')



tmodel <- rpart(fV,data=train_new)

prp(tmodel)


train_new$tpred<-predict(tmodel,newdata=train_new)
Cal$tpred<-predict(tmodel,newdata=Cal)
test$tpred<-predict(tmodel,newdata=test)

calcAUC(train_new$tpred,train_new[,outcome]) #0.8671868
calcAUC(Cal$tpred,Cal[,outcome]) #0.8803081




## Now lets build a nearest neighbor model
nK<-round(dim(train_new)[1]/10) #Number of neighbors analyzed (Total Observations/10)
knnTrain<-train_new[,selVars] #variables used for classification
knnCl<-train_new[,outcome]==pos #training outcomes


knnPred <- function(df) {
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T)
  ifelse(knnDecision==TRUE,
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}

train_new$nnpred<-knnPred(train_new[,selVars])
Cal$nnpred<-knnPred(Cal[,selVars])
test$nnpred<-knnPred(test[,selVars])


calcAUC(train_new$nnpred,train_new[,outcome]) #0.8742764
calcAUC(Cal$nnpred,Cal[,outcome]) #0.8728759

## Try Naive Bayes
ff <- paste('as.factor(',outcome,'==1) ~ ',
            paste(selVars,collapse=' + '),sep='')


nbmodel<-naiveBayes(as.formula(ff), data=train_new)
train_new$nbpred<-predict(nbmodel,newdata=train_new,type='raw')[,'TRUE']
Cal$nbpred<-predict(nbmodel,newdata=Cal,type='raw')[,'TRUE']
test$nbpred<-predict(nbmodel,newdata=test,type='raw')[,'TRUE']

calcAUC(train_new$nbpred,train_new[,outcome]) #0.8222325
calcAUC(Cal$nbpred,Cal[,outcome]) #0.8336543

## Try logistic regression
fm <- paste(outcome,'==1 ~ ',paste(c(catVars, numericVars),collapse=' + '),sep='')
model<-glm(fm, data=train_new, family=binomial(link="logit"))
train_new$logpred<-predict(model, newdata=train_new, type="response")
Cal$logpred<-predict(model, newdata=Cal, type="response")
test$logpred<-predict(model,newdata=test, type="response")

calcAUC(train_new$logpred,train_new[,outcome]) #0.8778028
calcAUC(Cal$logpred,Cal[,outcome]) #0.8714908




## Try Random Forest
fmodel<-randomForest(x=train_new[,c(catVars, numericVars)],
                     y=train_new[,outcome],
                     ntree=100,
                     nodesize=7,
                     importance=T)

varImp<-importance(fmodel)
varImpPlot(fmodel,type=1)


fresults<-predict(fmodel,newdata=train_new, "prob")
fresultsCal<-predict(fmodel,newdata=Cal, "prob")
fresultstest<-predict(fmodel,newdata=test[,c(catVars, numericVars)], "prob")

train_new$forestpred<-fresults[,2]
Cal$forestpred<-fresultsCal[,2]
test$forestpred<-fresultstest[,2]

calcAUC(train_new$forestpred,train_new[,outcome]) #0.9680052
calcAUC(Cal$forestpred,Cal[,outcome]) #0.8991081


#Try Adaboost
fb <- paste(outcome,' ~ ',
             paste(selVars,collapse=' + '),sep='')


adaboost<-boosting(as.formula(fb), data=train_new,boos=TRUE, mfinal=20,coeflearn='Breiman')
summary(adaboost)
adaboost$importance



train_new$apred<-predict(adaboost,train_new)$prob[,2]
Cal$apred<-predict(adaboost,Cal)$prob[,2]
test$apred<-predict(adaboost,test)$prob[,2]


t1<-adaboost$trees[[1]]
prp(t1)

calcAUC(train_new$apred,train_new[,outcome]) # 0.907329
calcAUC(Cal$apred,Cal[,outcome]) #0.8929935

#Try Gradient Boost
# fit initial model
train_new[,outcome] <- as.character(train_new[,outcome])
predictorSet<-c(selVars, outcome)
fb <- paste(outcome,' ~ ',
            paste(selVars,collapse=' + '),sep='')

gbm1 <-
  gbm(as.formula(fb), # formula
      data=train_new[,predictorSet], # dataset, 
      distribution="bernoulli", # see the help for other choices
      n.trees=1000, # number of trees
      shrinkage=0.1, # shrinkage or learning rate, # 0.001 to 0.1 usually work
      interaction.depth=1, # 1: additive model, 2: two-way interactions, etc.
      bag.fraction = 0.5, # subsampling fraction, 0.5 is probably best
      train.fraction = 0.5, # fraction of data for training,
      n.minobsinnode = 10, # minimum total weight needed in each node
      cv.folds = 5, # do 5-fold cross-validation
      keep.data=FALSE, # keep a copy of the dataset with the object
      verbose=FALSE)



train_new$gpred<-predict(object = gbm1,
                         newdata = train_new,
                         n.trees = 500,
                         type = "response")
Cal$gpred<-predict(object = gbm1,
                   newdata = Cal,
                   n.trees = 500,
                   type = "response")
test$gpred<-predict(object = gbm1,
                    newdata = test,
                    n.trees = 500,
                    type = "response")

calcAUC(train_new$gpred,train_new[,outcome]) #  0.8664815
calcAUC(Cal$gpred,Cal[,outcome]) #0.8622006


### Accuracy using 0.5 as cutoff
## Decision Tree
##accuracyMeasures(Cal$tpred, Cal[,outcome]==pos, name='Decision tree, Cal')
##Naive Bayes
##accuracyMeasures(Cal$nbpred, Cal[,outcome]==pos, name='Naive Bayes, Cal')
##Nearest Neighbor
##accuracyMeasures(Cal$nnpred, Cal[,outcome]==pos, name='Nearest Neighbor, Cal')
##Logistic Regression
##accuracyMeasures(Cal$logpred, Cal[,outcome]==pos, name='Logistic Regression, Cal')
##Random Forest
##accuracyMeasures(Cal$forestpred, Cal[,outcome]==pos, name='Random Forest, Cal')
##Adaboost
##accuracyMeasures(Cal$apred, Cal[,outcome]==pos, name='Adaboost, Cal')
##Gradientboost
##accuracyMeasures(Cal$gpred, Cal[,outcome]==pos, name='Gradient Boost, Cal')


ggplot(train_new, aes(x=logpred, color=Survived, linetype=Survived))+
  geom_density()




### Score classifcations in tables based on each method and optimal cutoff to maximize accuracy of each method

## Predicted Probability Variables
pred_vars<-c('tpred','nbpred','nnpred','logpred','forestpred','apred','gpred')

optimal_cutoff<-function(pred, truth){
  eval<-prediction(pred,truth)
  acc.perf<-performance(eval, measure = "acc")
  ind<-which.max( slot(acc.perf, "y.values")[[1]] )
  acc<-slot(acc.perf, "y.values")[[1]][ind]
  cutoff<-slot(acc.perf, "x.values")[[1]][ind]
}

#### Loop through each probability variable to find cutoff and append it to dataframe, then create formula to classify as 1 or 0


k<-as.integer(dim(Cal)[1])
j<-length(pred_vars)
cutoffdf<-data.frame(matrix(NA, nrow = k, ncol = j))


for (i in 1:j){
  cutoffdf[,i]<-sapply(Cal[,pred_vars[i]],function(x) optimal_cutoff(Cal[,pred_vars[i]],Cal[,outcome]))
  colnames(cutoffdf)[i]<-paste(pred_vars[i],'cut',sep='_')
}

#Append cutoffs to Calibration dataframe
Cal<-cbind(Cal,cutoffdf)

#Append cutoffs to Test dataframe
cutoffs<-Cal[1,paste(pred_vars,'cut',sep='_')]
cutoffs
m<-as.integer(dim(test)[1])
cutoffdf_test<-data.frame(matrix(NA, nrow = m, ncol = j))

for (i in 1:j){
  cutoffdf_test[,i]<-cutoffs[i]
  colnames(cutoffdf_test)[i]<-paste(pred_vars[i],'cut',sep='_')
}


test<-cbind(test,cutoffdf_test)


# Classify calibration cases according to cutoff
classdf<-data.frame(matrix(NA, nrow = k, ncol = j))
Classcols=c()
for (i in 1:j){
  cut_var<-paste(pred_vars[i],'cut',sep='_')
  classdf[,i]<-ifelse(Cal[,pred_vars[i]]>=Cal[,cut_var], 1, 0)
  colnames(classdf)[i]<-paste(pred_vars[i],'class',sep='_')
  Classcols=append(Classcols, paste(pred_vars[i],'class',sep='_'))
}


#Append Classifications to Calibration dataframe
Cal<-cbind(Cal,classdf)

# Classify Test cases according to cutoff
classdf_test<-data.frame(matrix(NA, nrow = m, ncol = j))

for (i in 1:j){
  cut_var<-paste(pred_vars[i],'cut',sep='_')
  classdf_test[,i]<-ifelse(test[,pred_vars[i]]>=test[,cut_var], 1, 0)
  colnames(classdf_test)[i]<-paste(pred_vars[i],'class',sep='_')
}


#Append Classifications to Test dataframe
test<-cbind(test,classdf_test)



#Calculate the average prediction across methods
Cal$avgpred_class<-ifelse(apply(Cal[,Classcols], 1,mean)>=.5,1,0)
test$avgpred_class<-ifelse(apply(test[,Classcols], 1,mean)>=.5,1,0)


#### Print out accuracy of each method
Classcols_plus<-append(Classcols,'avgpred_class')
j<-length(Classcols_plus)

for (i in 1:j){
  cM<-table(truth=Cal[,outcome], prediction=Cal[,Classcols_plus[i]])
  print(sprintf("%s, Accuracy: %g",
                Classcols_plus[i],(cM[1,1]+cM[2,2])/sum(cM)))
}



###############
###############
############### Lets try logistic regression also to aggregate the models

### Split Calibration Set into 2 so we can run the XGB Boost on one and validate on the other
Cal$gp<-runif(dim(Cal)[1])
Cal_1<-subset(Cal, Cal$gp<=.6) #test data
Cal_2<-subset(Cal, Cal$gp>.6) #initial training before split


### Try XGB Boost to aggregate predictions
X<-cbind(Survived=Cal_1[,outcome],Cal_1[,Classcols])
X$Survived<-as.character(X$Survived)
head(X)
xgb <- xgboost(data = data.matrix(X[,-1]), 
               label = X[,outcome], 
               n_estimators= 2000,
               min_child_weight= 2,
               gamma=0.9, 
               eta = 0.1,
               max_depth = 4,
               seed=1234,
               subsample = 0.8,
               colsample_bytree = 0.8,
               objective = "binary:logistic",
               nthread = 2,
               nrounds=500
)


Cal$XGBpred<- predict(xgb, data.matrix(Cal[,Classcols]))
Cal_1$XGBpred<- predict(xgb, data.matrix(Cal_1[,Classcols]))
Cal_2$XGBpred<- predict(xgb, data.matrix(Cal_2[,Classcols]))
test$XGBpred<- predict(xgb, data.matrix(test[,Classcols]))


calcAUC(Cal$XGBpred,Cal[,outcome]) #0.8708594
calcAUC(Cal_1$XGBpred,Cal_1[,outcome])#0.8693205
calcAUC(Cal_2$XGBpred,Cal_2[,outcome])#0.8721461




#### Find Optimal Cutoff for XGB Prediction

k<-as.integer(dim(Cal_2)[1])
cutoffdf_XGB<-data.frame(matrix(NA, nrow = k, ncol = 1))


cutoffdf_XGB[,1]<-sapply(Cal_2$XGBpred,function(x) optimal_cutoff(Cal_2$XGBpred,Cal_2[,outcome]))
colnames(cutoffdf_XGB)<-'XGBpred_cut'
  

#Append cutoffs to Cal_2ibration dataframe
Cal_2<-cbind(Cal_2,cutoffdf_XGB)

#Append cutoffs to Test dataframe
cutoffs<-Cal_2$XGBpred_cut[1]
m<-as.integer(dim(test)[1])
cutoffdf_XGB_test<-data.frame(matrix(NA, nrow = m, ncol = 1))


cutoffdf_XGB_test[,1]<-cutoffs[1]
colnames(cutoffdf_XGB_test)[1]<-'XGBpred_cut'


test<-cbind(test,cutoffdf_XGB_test)


# Classify Cal_2ibration cases according to cutoff
classdf_XGB<-data.frame(matrix(NA, nrow = k, ncol = 1))
classdf_XGB[,1]<-ifelse(Cal_2$XGBpred>=Cal_2$XGBpred_cut, 1, 0)
colnames(classdf_XGB)<-'XGBpred_class'

  

#Append Classifications to Cal_2ibration dataframe
Cal_2<-cbind(Cal_2,classdf_XGB)

# Classify Test cases according to cutoff
classdf_XGB_test<-data.frame(matrix(NA, nrow = m, ncol = 1))


classdf_XGB_test[,1]<-ifelse(test$XGBpred>=test$XGBpred_cut, 1, 0)
colnames(classdf_XGB_test)<-'XGBpred_class'


#Append Classifications to Test dataframe
test<-cbind(test,classdf_XGB_test)




#### Print out accuracy of each method
Classcols_plus_plus<-append(Classcols_plus,'XGBpred_class')
j<-length(Classcols_plus_plus)

for (i in 1:j){
  cM<-table(truth=Cal_2[,outcome], prediction=Cal_2[,Classcols_plus_plus[i]])
  print(sprintf("%s, Accuracy: %g",
                Classcols_plus_plus[i],(cM[1,1]+cM[2,2])/sum(cM)))
}



##### Export the test dataset with predictions
final_submission_df<-test[,c('PassengerId','XGBpred_class')]

colnames(final_submission_df)[2]<-'Survived'

head(final_submission_df)

write.csv(final_submission_df, file = 'McCullough_Titanic.csv', append=FALSE, sep = " ",
          eol = "\n", na = "NA", dec = ".", row.names = FALSE,
          col.names = TRUE, qmethod = c("escape", "double"))

