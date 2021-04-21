#Aim:--------------
#  Build a cnn that classify images
#  Train the CNN
#  Model evalution
#  Save the model for future use

# Load library, prepare and load MNIST datasets----------------
set.seed(1000)
ibrary(keras) #load keras library 
#[MNIST DATASET](http://yann.lecun.com/exdb/mnist/) # use MNIST Database
 
mnist <- dataset_mnist() 
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

# Keras model building using sequential API-----
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

summary(model) # See information about layers. parameters, shape, output etc

# Compile the model using compile function
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )
# Fit the data to the model

model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )
# Make Prediction using predict function
predictions <- predict(model, mnist$test$x) #returns the probability for each class
head(predictions, 2)

# Assess model performance------
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)
#Our model achieved over ~90% accuracy on the test set

#save the model for future use
save_model_tf(object = model, filepath = "P:/msc_statistic_2019_2020/2020_September-December_Sem-4/image-classification/models")
