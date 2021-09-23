library(keras)
library(devtools)

devtools::install_github("rstudio/keras")
install_keras()

###########################################
################## DATA ###################
###########################################

# 0 - T-shirt/top

# 1 - Trouser

# 2 - Pullover

# 3 - Dress

# 4 - Coat

# 5 - Sandal

# 6 - Shirt

# 7 - Sneaker

# 8 - Bag

# 9 - Ankle boot


mnist <- dataset_fashion_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x1_train <- array(x_train, dim=list(c(nrow(x_train)), 784)) / 255
x1_test <- array(x_test, dim=list(c(nrow(x_test)), 784)) / 255

y1_train <- to_categorical(y_train, 10)
y1_test <- to_categorical(y_test, 10)

x1_train_cnn <- array(x_train, dim=list(60000, 28, 28, 1)) / 255
x1_test_cnn <- array(x_test, dim=list(10000, 28, 28, 1)) / 255


###########################################
################# MODEL ###################
###########################################


###########################################
############## NEURAL NETWORK #############
###########################################

########## LossHistory is used to get losses values ######
########### for each iteration while training ########### 

LossHistory <- R6::R6Class("LossHistory",
                           inherit = KerasCallback,
                           
                           public = list(
                             
                             losses = NULL,
                             
                             on_batch_end = function(batch, logs = list()) {
                               self$losses <- c(self$losses, logs[["loss"]])
                             }
                           ))

#################### BUILD MODEL #######################

model <- keras_model_sequential() 
model %>%
  layer_dense(units=700, activation='relu', input_shape=784) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units=700, activation='relu') %>%
  layer_dense(units=100, activation='relu') %>%
  layer_dense(units=10, activation='softmax') 


model %>% 
  compile(loss = 'categorical_crossentropy' ,
          optimizer = optimizer_sgd(momentum=0.9, nesterov=T),
          metrics = 'accuracy')

history <- LossHistory$new()

model_output <- model %>% fit(x1_train, y1_train,
                         epochs = 5,
                         batch_size = 128,
                         validation_split = 0.2,
                         callbacks= list(history)
                         )


history$losses # losses for each iteration of GD

#################### TRAIN MODEL #######################

model_output <- model %>% fit(x1_train, y1_train,
                              epochs = 30,
                              batch_size = 250,
                              validation_split = 0.2
)

###########################################
################### CNN  ##################
###########################################

#################### BUILD MODEL #######################

model_cnn  <- keras_model_sequential() 
model_cnn %>%
  layer_conv_2d(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu',
                input_shape = c(28, 28, 1)) %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = 2, strides = 2) %>%
  
  layer_conv_2d(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = 2, strides = 2) %>%
  
  layer_conv_2d(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu') %>%
  layer_batch_normalization() %>%
  
  layer_flatten() %>%
  
  layer_dense(units=100, activation='relu') %>%
  layer_dense(units=10, activation='softmax')

model_cnn %>% 
  compile(loss = 'categorical_crossentropy' ,
          optimizer = optimizer_sgd(momentum=0.9, nesterov=T),
          metrics = 'accuracy')

#################### TRAIN MODEL #######################

#################### TRAIN MODEL #######################

# !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! 
# !!!DON'T RUN the model unless using GPU !!! 
# !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! 

model_output_cnn <- model_cnn %>% fit(x1_train_cnn, y1_train,
                                      epochs = 30,
                                      batch_size = 250,
                                      validation_split = 0.2
)

rm(mnist) # remove data from memory

########################################################
############## PLOT FOR FULL VS MINI BATCH #############
########################################################

plot(1:320, full_batch,type="l",col="navyblue",
     main = 'Full vs Mini Batch',
     xlab ='gradient descent iteration',ylab = "loss function", lwd=2)
lines(1:320, mini_batch, col="indianred1")


legend(150, 2.5, legend=c("Full Batch", "150 Mini Batch"),
       col=c("navyblue", "indianred1"), lty=c(1, 1), lwd=c(2, 1))

history_5000$metrics$val_accuracy[10]
model %>% evaluate(x1_test, y1_test)

########################################################
#################### PLOT FOR SGD ######################
########################################################

plot(1:length(history$losses),history$losses,type='l', 
     col='blue', main='Stochastic Gradient Descent', 
     xlab='Gradient descent iteration', ylab='Loss function')
full_batch = history$losses


########################################################
#################### MODEL SUMMARY #####################
########################################################

summary(model)
summary(model_cnn)

########################################################
#################### PREDICTION ########################
########################################################

model %>% evaluate(x1_test, y1_test)
model_cnn %>% evaluate(x1_test_cnn, y1_test)

########################################################
#################### PREDICTION PER CLASS ##############
########################################################

for (i in 1:10){
  print(model %>% evaluate(x1_test[y1_test[,i]==1,], y1_test[y1_test[,i]==1,]))
}


########################################################
#################### PLOT MODELS #######################
########################################################

plot(model_output)
plot(model_output_cnn)


########################################################
#################### EXTRACT FILTERS ###################
########################################################

# Extract the first convolutional layer (32 filters)
layer <- model_cnn %>% 
  get_layer(index=1) 

# Apply the filter with the first picture (dress)
output <- layer(array(x_train[1,,], dim=list(1, 28, 28, 1)) /255)

# Dress picture
image(x_train[1,,])

# Plot the 16 firs filters
par(mfrow=c(4,4))
for (i in 1:16){
  image(as.matrix(output[1,,i]))
}



