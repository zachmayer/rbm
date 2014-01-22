\name{rbm}
\alias{rbm}
\title{Fit a Restricted Boltzmann Machine}
\usage{
rbm(x, num_hidden = 2, max_epochs = 1000, learning_rate = 0.1,
  batch_size = nrow(x), initial_weights_mean = 0,
  initial_weights_sd = 0.1, momentum = 0, dropout = FALSE, retx = FALSE,
  verbose = FALSE, ...)
}
\arguments{
  \item{x}{a sparse matrix}

  \item{num_hidden}{number of neurons in the hidden layer}

  \item{retx}{whether to return the RBM predictions for the
  input data}

  \item{...}{not used}
}
\value{
  a RBM object
}
\description{
  This function fits an RBM to the input dataset.  It
  internally uses sparse matricies for faster matrix
  operations
}
\details{
  This code is (mostly) adapted from edwin chen's python
  code for RBMs, avaiable here:
  https://github.com/echen/restricted-boltzmann-machines.
  Some modifications (e.g. momentum) were adapted from
  Andrew Landgraf's R code for RBMs, available here:
  http://alandgraf.blogspot.com/2013/01/restricted-boltzmann-machines-in-r.html.
}
\examples{
#Setup a dataset
set.seed(10)
print('Data from: https://github.com/echen/restricted-boltzmann-machines')
Alice <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
Bob <- c('Harry_Potter' = 1, Avatar = 0, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #SF/fantasy fan, but doesn't like Avatar.
Carol <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
David <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
Eric <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Oscar winners fan, except for Titanic.
Fred <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
dat <- rbind(Alice, Bob, Carol, David, Eric, Fred)

#Fit a PCA model and an RBM model
PCA <- prcomp(dat)
RBM <- rbm(dat, retx=TRUE)

#Examine the 2 models
round(PCA$rotation, 2) #PCA weights
round(RBM$rotation, 2) #RBM weights

#Predict for new data
George <- as.matrix(t(c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 0, Gladiator = 1, Titanic = 1, Glitter = 0)))
predict(PCA, George)
predict(RBM, George, type='activations')
predict(RBM, George, type='probs')
predict(RBM, George, type='states')
}

