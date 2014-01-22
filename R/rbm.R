#' Fit a Restricted Boltzmann Machine
#' 
#' This function fits an RBM to the input dataset.  It internally uses sparse matricies for faster matrix operations
#' 
#' @param x a sparse matrix
#' @param n number of neurons in the hidden layer
#' @param retx whether to return the RBM predictions for the input data
#' @param ... not used
#' @export
#' @return a RBM object
rbm <- function (x, n = 5, retx = FALSE, ...) {
  require('Matrix')
  stop('not implemented')
  stopifnot(is.Matrix(x))
}

#' Predict from a Restricted Boltzmann Machine
#' 
#' This function takes an RBM and a matrix of new data, and predicts for the new data with the RBM. Note that RBMs are stochastic, so you will get slightly different predictions each time you run this function
#' 
#' @param x a RBM object
#' @param newdata a sparse matrix of new data
#' @param ... not used
#' @export
#' @return a sparse matrix
predict.RBM <- function (object, newdata, ...) {
  require('Matrix')
  stop('not implemented')
  if (missing(newdata)) {
    if (!is.null(object$x)) 
      return(object$x)
    else stop("no scores are available: refit with 'retx=TRUE'")
  }
  stopifnot(is.Matrix(newdata))
  nm <- rownames(object$rotation)
  if (!is.null(nm)) {
    if (!all(nm %in% colnames(newdata))) 
      stop("'newdata' does not have named columns matching one or more of the original columns")
    newdata <- newdata[, nm, drop = FALSE]
  }
  else {
    if (NCOL(newdata) != NROW(object$rotation)) 
      stop("'newdata' does not have the correct number of columns")
  }
  newdata %*% object$rotation
}
