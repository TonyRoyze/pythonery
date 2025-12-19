library(ISLR)
library(glmnet)
library(pls)
library(leaps)
data("Hitters")
fix(Hitters)
dim(Hitters)
str(Hitters)
sum(is.na(Hitters$Salary)) 
Hitters = na.omit(Hitters) 
dim(Hitters)
sum(is.na(Hitters)) 
################################################################################
#            Best Subset Selection                                             #
################################################################################
regfit.full<-regsubsets(Salary~.,data=Hitters)
summary(regfit.full)
#It gives by default best-subsets up to size 8; 
#lets increase that to 19, i.e. all the variables
regfit.full<-regsubsets(Salary~.,data=Hitters, nvmax=19)
regfull.summary<-summary(regfit.full)
summary(regfit.full)
names(regfull.summary)

plot(regfull.summary$cp,xlab="Number of Variables",ylab="Cp")
which.min(regfull.summary$cp)
points(10,regfull.summary$cp[10],pch=20,col="red")

plot(regfull.summary$adjr2,xlab="Number of Variables",ylab="adjr2")
which.max(regfull.summary$adjr2)
points(11,regfull.summary$adjr2[11],pch=20,col="red")


# There is a plot method for regsubset
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="adjr2")
# coefficents of the selected models
coef(regfit.full,10)
coef(regfit.full,11)

o#####################################################
#                 Forward selection                 #
#####################################################
regfit.fwd<-regsubsets(Salary~.,data=Hitters,nvmax=19,
method="forward")
regfwd.summary<-summary(regfit.fwd)
regfwd.summary
plot(regfit.fwd,scale="Cp")
plot(regfwd.summary$cp)

regfit.bw<-regsubsets(Salary~.,data=Hitters,nvmax=19,
method="backward")
regbw.summary<-summary(regfit.bw)
regbw.summary
plot(regfit.bw,scale="Cp")


which.min(regfull.summary$cp)
which.min(regfwd.summary$cp)
which.min(regbw.summary$cp)

coef(regfit.full,10)
coef(regfit.fwd,10)
coef(regfit.bw,10)
#####################################################
# Model Selection Using a Validation Set             #                                     
#####################################################
# regsubsets does not have a prediction method
# The following is the prediction function we are going to use
predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  mat[,names(coefi)]%*%coefi
}

dim(Hitters)
set.seed(1)
train=sample(seq(263),180,replace=FALSE)
train
regfit.fwd2=regsubsets(Salary~.,data=Hitters[train,],nvmax=19,method="forward")

val.errors=rep(NA,19)
for(i in 1:19){
              pred<-predict.regsubsets(regfit.fwd2,Hitters[-train,],id=i)
              val.errors[i]<-mean((Hitters$Salary[-train]-pred)^2)
}
plot(sqrt(val.errors),ylab="Root MSE",ylim=c(275, 425),pch=19,type="b")
points(sqrt(regfit.fwd2$rss[-1]/180),col="blue",pch=19,type="b")
legend("topright",legend=c("Training","Validation"),
       col=c("blue","black"),pch=19)

which.min(sqrt(val.errors))

set.seed(11)
folds=sample(rep(1:10,length=nrow(Hitters)))
folds
table(folds)
cv.errors=matrix(NA,10,19)
train.errors =matrix(NA,10,19)
for(k in 1:10){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=k,],nvmax=19,method="forward")
  train.errors[k,]=(best.fit$rss[-1]/dim(Hitters[folds!=k,])[1])
  
  for(i in 1:19){
    pred=predict(best.fit,Hitters[folds==k,],id=i)
    cv.errors[k,i]=mean( (Hitters$Salary[folds==k]-pred)^2)
  }
}

rmse.cv=sqrt(apply(cv.errors,2,mean))
rmse.train =sqrt(apply(train.errors,2,mean))

plot(rmse.cv,pch=19,type="b",ylim=c(300, 400),ylab="RMSE")
points(rmse.train,col="blue",pch=19,type="b")
legend("topright",legend=c("Training","10 Fold Cross-Validation"),
       col=c("blue","black"),pch=19)

which.min(rmse.cv) 

regfit.final=regsubsets(Salary~.,data=Hitters ,nvmax=19,method = 'forward')
coef(regfit.final,9)
######################################################################
#                    Ridge Regression                                #
######################################################################
x=model.matrix(Salary~.-1,Hitters)
fix(x)
y=Hitters$Salary
# Fitting ridge regression
fit.ridge=glmnet(x,y,alpha=0)
plot(fit.ridge,xvar="lambda",label=TRUE,lw=2)

# Doing cross validation to select the best lambda
cv.ridge=cv.glmnet(x,y,alpha=0)
plot(cv.ridge)    
bestlam =cv.ridge$lambda.min 
bestlam
# Fitting the ridge regression model under the best lambda
#out=glmnet(x,y,alpha=0)
#predict(out ,type="coefficients",s=bestlam)[1:20,]
#coef(out,s=bestlam)
coef(fit.ridge,s=bestlam)

############################################################
#                     The Lasso                            #
############################################################
fit.lasso<-glmnet(x,y)
plot(fit.lasso,xvar="lambda",label=TRUE,lw=2)
plot(fit.lasso,xvar="dev",label=TRUE,lw=2)

# Doing cross validation to select the best lambda
set.seed(5)
cv.lasso=cv.glmnet(x,y)
plot(cv.lasso)
lam.best=cv.lasso$lambda.min
# coefficient vector under the one std of the best lambda
coef(cv.lasso)
# coefficient vector at the best lambda
coef(cv.lasso,s=lam.best)

# Finding lambda splitting data set into trainning testing
set.seed(1)
train=sample (1: nrow(x), (2/3)*nrow(x))
train
length(train)
lasso.tr<-glmnet(x[train,],y[train])
lasso.tr
pred<-predict(lasso.tr,x[-train,])
dim(pred)
mse<-sqrt(apply((y[-train]-pred)^2,2,mean))
plot(lasso.tr$lambda,mse,type="b",xlab="Log(lambda)",col="red")
lam.best<-lasso.tr$lambda[order(mse)[1]]
coef(lasso.tr,s=lam.best)
#############################################################
#     Principal components Regression(PCR)                  #
#############################################################
library(pls)
set.seed(2)
pcr.fit<-pcr(Salary~., data=Hitters ,scale=TRUE, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit ,val.type="RMSEP")
pcr.fit2<-pcr(Salary~.,data=Hitters,scale=TRUE,ncomp=5) 
summary(pcr.fit2)
coef(pcr.fit2)
############################################################
#               PLSR                                       #
############################################################
set.seed(1) 
pls.fit<-plsr(Salary~., data=Hitters,scale=TRUE, 
             validation="CV")
summary(pls.fit)
validationplot(pls.fit ,val.type="RMSEP")

pls.fit2<-plsr(Salary~., data=Hitters ,scale=TRUE,ncomp=8)
summary(pls.fit2)
coef(pls.fit2)
