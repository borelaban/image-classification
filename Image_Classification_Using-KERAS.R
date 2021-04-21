#Aim of MNIST Datsets:--------------
#  Build a cnn that classify images
#  Train the CNN
#  Model evalution
#  Save the model for future use

# Load library, prepare and load MNIST datasets
set.seed(1000)
library(keras) #load keras library 
#[MNIST DATASET](http://yann.lecun.com/exdb/mnist/) # use MNIST Database
 
mnist <- dataset_mnist() 
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

# Keras model building using sequential API
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

# Assess model performance
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)
#Our model achieved over ~90% accuracy on the test set

#save the model for future use
save_model_tf(object = model, filepath = "P:/msc_statistic_2019_2020/2020_September-December_Sem-4/image-classification/models")

#Can be reloaded as 
reloaded_model <- load_model_tf("P:/msc_statistic_2019_2020/2020_September-December_Sem-4/image-classification/models")
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))

# Fashion datasets from MNIST packaged under KERAS ----
# Use Fashion MNIST dataset as a replacement of the classical MNIST dataset
# Fashion MNIST dataset is images of clothings, bags and sneakers etc
# Train the cnn model to classify this images
# Fashion MNIST dataset contains  70,000 grayscale images in 10 categories
# The resolution size is 28 by 28 pixels

## We will use 60,000 images to train the network and 10,000 images to evaluate 
## how accurately the network learned to classify images

fashion_mnist <- dataset_fashion_mnist() # download the dataset
c(train_images, train_labels) %<-% fashion_mnist$train #train_images and train_labels arrays are the training set data the model uses to learn
c(test_images, test_labels) %<-% fashion_mnist$test # the model is tested against this array of test set

# The pixel values ranges between 0 and 255. The labels are arrays of integers, ranging from 0 to 9
# The integers correspond to the class of clothings the image represent
# Each image is mapped to a single label. Since the class names are not included with the dataset, we’ll store them in a vector to use later when plotting the images

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

## Data Exploration 

# what is the format of the dataset
dim(train_images) # Training images is 60000 and 28by28 pixels
dim(train_labels) # 60,000 labels in train set
train_labels[1:20] # labels are integers between 0-9

dim(test_images)
dim(test_labels)
test_labels[1:20]

# Preprocessing the data

# Preprocess the data before training the network
# Let inspect the first image
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# Divide the values by 255 before feeding the NN model to scale it to a value o-1
train_images <- train_images / 255
test_images <- test_images / 255

# Display of first 20 images and class name below each image

par(mfcol=c(4,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:20) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

# Model building
## Configure the layers of the network model: Set-up the layers
#Layers extract representation from the data fed into them

model_fashion <- keras_model_sequential()
model_fashion %>%
  layer_flatten(input_shape = c(28, 28)) %>% #transforms form of the 2-d array(28*28) to 1-d 28*28=784pixels
  layer_dense(units = 128, activation = 'relu') %>% # fully connect layers with 128 neurons/nodes
  layer_dense(units = 10, activation = 'softmax') # full connect softmax node layers with 10 neuronsnodes, it  returns an array of 10 probability scores that sum to 1.
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 digit classes

# Compile the model

## Components
### Loss fxn - measures of how accurate the model is. Aim to minimis this paramater
### Optimizer - use to adjust the model based on the data it sees and the loss function
### Metrics - use to monitor the training and testing steps eg accuracy

model_fashion %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy'
)

# Model training

## Feed the training data to the model (train_images and train_labels)
## Model learns to associate images with the labels

model_fashion %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

# Evaluate the accuracy

score <- model_fashion %>% evaluate(test_images, test_labels, verbose = 0)
cat('Test loss:', score["loss"], "\n")
cat('Test accuracy:', score["accuracy"], "\n")

#the accuracy on the training dataset is higher than on the test dataset
#this is overfitting: ML performs worse on new data than on their training data

# Making prediction

predictions <- model_fashion %>% predict(test_images)
predictions[1, ] # describe the “confidence” of the model t

which.max(predictions[1, ]) #label with highest Conf value

class_pred <- model_fashion %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

# Prediction about a single image

img <- test_images[1, , , drop = FALSE]
dim(img)

#predict the image
predictions <- model_fashion %>% predict(img)
predictions

# subtract 1 as labels are 0-based
prediction <- predictions[1, ] - 1
which.max(prediction)

class_pred <- model_fashion %>% predict_classes(img)
class_pred

# The model predict a label of 9