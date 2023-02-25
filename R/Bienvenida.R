library(readr)
Rmissing <- read_csv("C:/Users/sebas/OneDrive/Escritorio/Bienvenida/R/Rmissing.csv")
set.seed(30)
X<- Rmissing
X<- as.data.frame(X[,-1])
ind<- which(X$mydata> 100 | X$mydata< -20)
ind<- c(which(is.na(X)),ind)
for(i in 1:length(ind)){
  X[ind[i],]<- rnorm(1,mean(X[-ind,1],na.rm=TRUE),sd(X[-ind,1],na.rm=TRUE))
}
hist(X[,1],xlab='',ylab='Frecuencia Relativa', main=' Histograma', freq=F,col='aquamarine3')
lines(density(X[,1]),lwd=2,lty=2,col='red4')
#####################################################
library(faraway) 
library(caret)
library(ggfortify)
set.seed(123)
Y<- fat
head(Y)
Y<- as.data.frame(cbind(fat$siri,fat$abdom,fat$weight))
colnames(Y)<- c('Siri','Abdom','Weight')
model <- lm(Siri~ Abdom+ Weight, data=Y)
summary(model)
#representación 3D de la regresión con dos predictoras
library(scatterplot3d)
library(plot3D)
library(plotly)
library(scatterplot3d)
library(rgl)
library(plot3Drgl)
z<-Y$Siri
y<-Y$Abdom
x<-Y$Weight
scatter3D(x, y, z, phi = 0, bty = "b",
          pch = 20, cex = 2, ticktype = "detailed",xlab = "Peso (Kg)",
          ylab ="Longitud Abdomen (Cm)", zlab = "% Grasa Corporal")
#La variable Z es la variable a predecir
#Creamos un objeto para realizar las predicciones con elmodelo
objr<-lm(z ~ x+y)
objr
#preparamos el modelado 3d
grid.lines = 42
x.pred <- seq(min(x), max(x), length.out = grid.lines)


y.pred <- seq(min(y), max(y), length.out = grid.lines)
xy <- expand.grid( x = x.pred, y = y.pred)
z.pred <- matrix(predict(objr, newdata = xy), 
                 nrow = grid.lines, ncol = grid.lines)
# Marcamos las líneas de iteracción para que busquen la recta de regresión
fitpoints <- predict(objr)
#ploteamos la gráfica en 3d con recta de regresión
scatter3D(x, y, z, pch = 19, cex = 2, 
          theta = 20, phi = 20, ticktype = "detailed",
          xlab = "Peso (Kg)",
          ylab ="Longitud Abdomen (Cm)", zlab = "% Grasa Corporal",
          surf = list(x = x.pred, y = y.pred, z = z.pred,  
                      facets = NA, fit = fitpoints), main = "")
#Gráfico dinámico
plotrgl()


