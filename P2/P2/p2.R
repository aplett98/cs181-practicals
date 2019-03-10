# packages
library(randomForest)
library(tidyverse)
library(dplyr)
library(janitor)

#variables
data <- read_csv("/Users/MeLlamoJohn/cs181/cs181-practicals/p2/foo.csv") %>% clean_names()


xs <- data %>% 
  select(-category,-id, -x0)

train_xs = xs[1:2777,]

evaluate_xs = xs[2778:3086, ]



train_ys = data$category[1:2777]
evaluate_ys = data$category[2778:3086]

# model fit
# note that you must turn the ordinal variables into factor or R wont use
# them properly
model_1000 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=1000, xtest=evaluate_xs)
model_500 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=500, xtest=evaluate_xs)
model_2000 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=2000, xtest=evaluate_xs)
model_250 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=250, xtest=evaluate_xs)
model_50 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=50, xtest=evaluate_xs)
model_10 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=10, xtest=evaluate_xs)
model_100 <- randomForest(y=as.factor(train_ys), x = train_xs ,ntree=100, xtest=evaluate_xs)

sum(model_10$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_50$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_100$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_250$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_500$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_1000$test$predicted == evaluate_ys) / length(evaluate_ys)
sum(model_2000$test$predicted == evaluate_ys) / length(evaluate_ys)

#plot of model accuracy by class
plot(model)

summary(model)




#variables
test_data <- read_csv("/Users/MeLlamoJohn/cs181/cs181-practicals/p2/test_data.csv") %>% clean_names()


test_xs <- test_data %>% 
  select(-id, -x0, -id_hash)

id_hashes <- test_data$id_hash

# model fit
# note that you must turn the ordinal variables into factor or R wont use
# them properly
model <- randomForest(y=as.factor(data$category), x = xs ,ntree=500, xtest=test_xs)

#plot of model accuracy by class
plot(model)

summary(model)

output <- tibble(id_hashes, model$test$predicted)

write.csv(file="predictions.csv", x=output)

plot(model)


library(naivebayes)

counts_calls <- data$x136

train_counts_calls <- counts_calls[1:2777]

evaluate_counts_calls = counts_calls[2778:3086]


train_ys = data$category[1:2777]
evaluate_ys = data$category[2778:3086]

library(naivebayes)

model <- naive_bayes(y=as.factor(train_ys), x = train_counts_calls, prior = c(0.0369, 0.0162,0.0120,0.0103,0.0133,0.0126,0.0172,0.0133,0.5214,0.0068,0.1756,0.0104,0.1218,0.0191,0.0130))

sum(predict(object = model, newdata = as.data.frame(evaluate_counts_calls)) == evaluate_ys)/ length(evaluate_ys)



