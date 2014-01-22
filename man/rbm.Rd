\name{rbm}
\alias{rbm}
\title{Fit a Restricted Boltzmann Machine}
\usage{
rbm(x, n = 5, retx = FALSE, ...)
}
\arguments{
  \item{x}{a sparse matrix}

  \item{n}{number of neurons in the hidden layer}

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

