X<- dataset
class(X[,1])
sample(X[,1],1)
for(i in 1:ncol(X)){
X[,i]<-ifelse(is.na(X[,i]),sample(X[,i],1),X[,i])
}
View(X)
