library(tensorflow)
library(keras)
cifar <- dataset_cifar10()
class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')