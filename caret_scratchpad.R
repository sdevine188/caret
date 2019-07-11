library(tidyverse)
library(caret)
library(ROCR)
library(rsample) # for attrition data


# xgboost (top algorithm for structured data)
# https://analyticsdataexploration.com/xgboost-model-tuning-in-crossvalidation-using-caret-in-r/
# https://xgboost.readthedocs.io/en/latest/tutorials/model.html
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
# https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d

# load attrition data
data(attrition)
attrition %>% glimpse()
# for classification, xgboost expects a two-level factor
# for regression, it expects numeric

# split data into training/testing set
set.seed(123)
training_split <- createDataPartition(y = attrition %>% pull(Attrition), p = 0.7, list = FALSE) %>%
        as.data.frame(.) %>% rename(training_rows = Resample1)

# inspect split
training_split %>% str()
training_split %>% class()
training_split %>% head(10)
training_split %>% glimpse()

# get train/test data
train_data <- attrition %>% filter(row_number() %in% (training_split %>% pull(training_rows)))
train_data %>% glimpse()

test_data <- attrition %>% filter(!(row_number() %in% (training_split %>% pull(training_rows))))
test_data %>% glimpse()


#################


# set control parameters to pass to caret
control_parameters <- trainControl(method = "cv", number = 10, savePredictions = TRUE, classProbs = TRUE)
control_parameters


##################


# set tuning_grid_parameters to pass to xgboost

# nrounds (# Boosting Iterations)
# It is the number of iterations the model runs before it stops.
# With higher value of nrounds model will take more time and vice-versa.
#
# max_depth (Max Tree Depth)
# Higher value of max_depth will create more deeper trees or we can say it will create more complex model.
# Higher value of max_depth may create overfitting and lower value of max_depth may create underfitting.
# All depends on data in hand.Default value is 6.
# range: [1,infinity]
# 
# eta (Shrinkage)
# It is learning rate which is step size shrinkage which actually shrinks the feature weights. 
# With high value of eta,model will run fast and vice versa.With higher eta and lesser nrounds,model will take lesser time to run.With lower eta and higher nrounds model will take more time.
# range: [0,1]
# 
# gamma (Minimum Loss Reduction)
# It is minimum loss reduction required to make a further partition on a leaf node of the tree. 
# The larger value will create more conservative model.
# One can play with this parameter also but mostly other parameters are used for model tuning.
# range: [0,infinity]
# 
# colsample_bytree (Subsample Ratio of Columns)
# Randomly choosing the number of columns out of all columns or variables at a time while tree building process.
# You can think of mtry paramter in random forest to begin understanding more about this.
# Higher value may create overfitting and lower value may create underfitting.One needs to play with this value.
# range: (0,1]
# 
# min_child_weight (Minimum Sum of Instance Weight)
# You can try to begin with thinking of min bucket size in decision tree( rpart).
# It is like number of observations a terminal node.
# If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, 
# then the building process will give up further partitioning. In linear regression mode, 
# this simply corresponds to minimum number of instances needed to be in each node
# range: [0,infinity]

# subsample - subsample ratio of the training instance. 
# Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees 
# and this will prevent overfitting. It makes computation shorter (because less data to analyse). 
# It is advised to use this parameter with eta and increase nrounds. Default: 1.
# analytics vidhaya recommends .5 - .9, with initial set to .8
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

tuning_grid_parameters <- expand.grid(eta = 0.1, colsample_bytree = c(0.5, 0.7), max_depth = c(3, 6),
                                        nrounds = 100, gamma = 1, min_child_weight = 2, subsample = .8)
tuning_grid_parameters


#################


# get xgb model
xgb_model <- train(Attrition ~ ., data = train_data, method = "xgbTree", trControl = control_parameters,
      tuneGrid = tuning_grid_parameters)

# inspect xgb_model
attributes(xgb_model)
xgb_model
summary(xgb_model)
xgb_model$finalModel
xgb_model$results
xgb_model$bestTune
getTrainPerf(xgb_model)
xgb_model$resample

# roc, sensitivity, specificity not automatically available for xgboost
# getTrainPerf(xgb_model)$TrainROC
# getTrainPerf(xgb_model)$TrainSpec
# getTrainPerf(xgb_model)$TrainSens


#####################


# predict on test
test_data_predictions <- predict(object = xgb_model, newdata = test_data, type = "prob")
test_data_predictions %>% glimpse()

# get predicted_class and add actual_class from test_data
test_data %>% glimpse()

test_data_predictions <- test_data_predictions %>% 
        mutate(predicted_class = case_when(Yes >= .5 ~ "Yes", TRUE ~ "No"),
                predicted_class = factor(predicted_class),
               actual_class = test_data %>% pull(Attrition)) %>% as_tibble()
test_data_predictions


################


# create confusion matrix
confusionMatrix(data = test_data_predictions %>% pull(predicted_class), 
                reference = test_data_predictions %>% pull(actual_class), positive = "Yes")


####################


# get roc curve

# confirm levels for actual outcome to pass to performance()
test_data_predictions %>% count(actual_class)
levels(test_data_predictions %>% pull(actual_class))

prediction_obj <- prediction(predictions = test_data_predictions %>% pull(Yes), 
           labels = test_data_predictions %>% pull(actual_class), 
           label.ordering = c("No", "Yes"))

performance_obj <- performance(prediction.obj = prediction_obj, measure = "tpr", x.measure = "fpr")
plot(performance_obj, colorize = TRUE, print.cutoffs.at = c(.1, .5, .7, .8, .9))
abline(a=0, b=1)

# get auc
test_data_predictions_auc <- performance(prediction.obj = prediction_obj, measure = "auc")
test_data_predictions_auc@y.values


#################

