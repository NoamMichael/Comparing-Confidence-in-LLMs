##############################################################
#  LEH analysis
#  Don Moore
##############################################################

library(tidyr)
library(dplyr)
library(ggplot2)
stderr <- function(x) sqrt(var(x,na.rm=TRUE)/length(na.omit(x))) #define standard error function
rstudioapi::writeRStudioPreference("data_viewer_max_columns", 1000L) #change the number of columns shown from 50

rm(list=ls())                      # clear environment
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014")                        # clear console

#read in the cleaned code
df <- read.csv("260114LEH.csv")

df$oc <- df$PConf - (df$Score*100) #computer overconfidence at the item level

summary(df$PConf)
sd(df$PConf)
summary(df$Score)
sd(df$Score)

conf <- aggregate(PConf~ResponseId,df,mean)
acc  <- aggregate(Score~ResponseId,df,mean)
orps <- merge(conf,acc,by="ResponseId")

t.test(orps$PConf,orps$Score)

# mixed model ANOVA 
fit <- aov(oc~(Radius*SexMale*MinAge)+Error(ResponseId/(Radius*SexMale*MinAge)),  
           data=df)
summary(fit)

aggregate(oc~Radius,df,mean)
aggregate(PConf~Radius,df,mean)
aggregate(Score~Radius,df,mean)
