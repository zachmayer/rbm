\name{predict.RBM}
\alias{predict.RBM}
\title{Predict from a Restricted Boltzmann Machine}
\usage{
\method{predict}{RBM}(object, newdata, ...)
}
\arguments{
  \item{x}{a RBM object}

  \item{newdata}{a sparse matrix of new data}

  \item{...}{not used}
}
\value{
  a sparse matrix
}
\description{
  This function takes an RBM and a matrix of new data, and
  predicts for the new data with the RBM. Note that RBMs
  are stochastic, so you will get slightly different
  predictions each time you run this function
}

