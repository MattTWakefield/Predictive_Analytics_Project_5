# Instruction: You need a computer with good (fast) processor to compile the code.
# Please see the link for details
# https://tensorflow.rstudio.com/tutorials/advanced/images/cnn/
library(tensorflow)
library(keras)

######### Download and prepare the CIFAR10 dataset ###############
##################################################################
#The CIFAR10 dataset contains 60,000 color images in 10 classes, 
#with 6,000 images in each class. The dataset is divided into 50,000 
#training images and 10,000 testing images. The classes are mutually
#exclusive and there is no overlap between them.
##################################################################
cifar <- dataset_cifar10()

class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

index <- 1:30

par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))
cifar$train$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})
###################################################################

################   Create the convolutional base ################
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

summary(model)
#####################################################################
####################################################################
############### Add Dense layers on top #############################
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)
#####################################################################
#####################################################################
############ Compile and train the model ############################
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = cifar$train$x, y = cifar$train$y,
    epochs = 2,
    validation_data = unname(cifar$test),
    verbose = 1
  )

#####################################################################
################ Evaluate the model ################################
####################################################################
plot(history)
evaluate(model, cifar$test$x, cifar$test$y, verbose = 0)
