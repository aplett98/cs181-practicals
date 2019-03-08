# packages
library(randomForest)
library(tidyverse)
library(dplyr)
library(janitor)

#variables
data <- read_csv("/Users/MeLlamoJohn/cs181/cs181-practicals/p2/foo.csv") %>% clean_names()


xs <- data %>% 
  select(-category)

# model fit
# note that you must turn the ordinal variables into factor or R wont use
# them properly
model <- randomForest(y=as.factor(data$category), x = xs ,ntree=100)

#plot of model accuracy by class
plot(model)

summary(model)
