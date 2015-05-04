#' Fit a Stack of Restricted Boltzmann Machines
#'
#' @param x a sparse matrix
#' @param layers an integer vector of the number of neurons in each RBM
#' @param learning_rate The learning rate
#' @param verbose_stack Print messages while training the stack
#' @param use_gpu use rbm_gpu instead of rbm to train the rbm on a gpu
#' @param ... passed to the rbm function
#' @export
#' @return a stacked_rbm object
#' @importFrom Matrix Matrix cBind drop0
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
#' @references
#' \itemize{
#' \item \url{http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines}
#' }
#' @examples
#' #Setup a dataset
#' set.seed(10)
#' print('Data from: https://github.com/echen/restricted-boltzmann-machines')
#' #Big SF/fantasy fan.
#' Alice <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0)
#' #SF/fantasy fan, but doesn't like Avatar.
#' Bob <- c('Harry_Potter' = 1, Avatar = 0, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0)
#' #Big SF/fantasy fan.
#' Carol <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0)
#' #Big Oscar winners fan.
#' David <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0)
#' #Oscar winners fan, except for Titanic.
#' Eric <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 0, Glitter = 0)
#' #Big Oscar winners fan.
#' Fred <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0)
#' dat <- rbind(Alice, Bob, Carol, David, Eric, Fred)
#'
#' Stacked_RBM <- stacked_rbm(dat, layers = c(3, 3), max_epochs=25)
stacked_rbm <- function (x, layers = c(30, 100, 30), learning_rate=0.1, verbose_stack=TRUE, use_gpu=FALSE, ...) {
  if(use_gpu){
    if(require('gputools')){
      rbm <- rbm_gpu
    } else {
     warning('The gputools package is require to train RBMs on the gpu.  RBMs will be trained on the cpu instead.')
    }
  }

  #Checks
  stopifnot(length(dim(x)) == 2)
  if(length(learning_rate)==1){
    learning_rate <- rep(learning_rate, length(layers))
  }
  stopifnot(length(layers) == length(learning_rate))

  if(any('data.frame' %in% class(x))){
    if(any(!sapply(x, is.finite))){
      stop('x must be all finite.  rbm does not handle NAs, NaNs, Infs or -Infs')
    }
    if(any(!sapply(x, is.numeric))){
      stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
    }
    x = Matrix(as.matrix(x), sparse=TRUE)
  } else if (any('matrix' %in% class(x))){
    x = Matrix(x, sparse=TRUE)
  } else if(length(attr(class(x), 'package')) != 1){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  } else if(attr(class(x), 'package') != 'Matrix'){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  }

  x_range <- range(x)
  scaled <- FALSE
  if (x_range[1] <0 | x_range[2] > 1){
    warning("x is out of bounds, automatically scaling to 0-1, test data will be scaled as well")
    x <- (x - x_range[1]) / (x_range[2] - x_range[1])
    scaled <- TRUE
  }

  if(length(layers) < 2){
    stop('Please use the rbm function to fit a single rbm')
  }

  #Fit first RBM
  if(verbose_stack){print('Fitting RBM 1')}
  rbm_list <- as.list(layers)
  rbm_list[[1]] <- rbm(x, num_hidden=layers[[1]], learning_rate=learning_rate[[1]], retx=TRUE, ...)

  #Fit the rest of the RBMs
  for(i in 2:length(rbm_list)){
    if(verbose_stack){print(paste('Fitting RBM', i))}
    rbm_list[[i]] <- rbm(predict(rbm_list[[i-1]], type='probs', omit_bias=TRUE), num_hidden=layers[[i]], learning_rate=learning_rate[[i]], retx=TRUE, ...)
  }

  #Return result
  out <- list(rbm_list=rbm_list, layers=layers, activation_function=rbm_list[[1]]$activation_function, x_range=x_range, scaled=scaled)
  class(out) <- 'stacked_rbm'
  return(out)
}

#' Predict from a Stacked Restricted Boltzmann Machine
#'
#' This function takes a stacked RBM and a matrix of new data, and predicts for the new data with the RBM.
#'
#' @param object a RBM object
#' @param newdata a sparse matrix of new data
#' @param type a character vector specifying whether to return the hidden unit activations, hidden unit probs, or hidden unit states.  Activations or probabilities are typically the most useful if you wish to use the RBM features as input to another predictive model (or another RBM!).  Note that the hidden states are stochastic, and may be different each time you run the predict function, unless you set random.seed() before making predictions.  Activations and states are non-stochastic, and will be the same each time you run predict.
#' @param omit_bias Don't return the bias column in the prediciton matrix.
#' @param ... not used
#' @export
#' @return a sparse matrix
#' @importFrom Matrix Matrix cBind drop0
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
predict.stacked_rbm <- function (object, newdata, type='probs', omit_bias=TRUE, ...) {

  #If no new data, just return predictions from the final rbm in the stack
  if (missing(newdata)) {
    return(predict(object$rbm_list[[length(object$rbm_list)]], type=type, omit_bias=omit_bias))
  } else {
    if(! type %in% c('probs', 'states')){
      stop('Currently we can only return hidden probabilities or states from a stacked rbm.  Activations are not yet supported')
    }
    hidden_probs <- predict(object$rbm_list[[1]], newdata=newdata, type='probs', omit_bias=TRUE)
    for(i in 2:length(object$rbm_list)){
      hidden_probs <- predict(object$rbm_list[[i]], newdata=hidden_probs, type='probs', omit_bias=TRUE)
    }
  }

  #Scale if scaled during training
  if (!is.null(object$scaled)) {
    if(object$scaled){
      newdata <- (newdata - object$x_range[1]) / (object$x_range[2] - object$x_range[1])
    }
    if(min(newdata) < 0 | max(newdata) > 1){
      stop('newdata outside of the scale of the model training data.  Could not re-scale data to be 0-1')
    }
  }

  if(omit_bias){
    if(type=='probs'){return(hidden_probs)}
    hidden_states <- hidden_probs > Matrix(runif(nrow(hidden_probs)*ncol(hidden_probs)), nrow=nrow(hidden_probs), ncol=ncol(hidden_probs))
    return(hidden_states)
  } else{
    if(type=='probs'){return(hidden_probs)}
    rows <- nrow(object$rotation) #Not sure about this one.  rows wasn't defined, so I'm trying to fix it
    hidden_states <- hidden_probs > Matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states)
  }

}

#' Combine weights from a Stacked Restricted Boltzmann Machine
#'
#' This function takes a stacked RBM and returns the combined weight matrix
#'
#' @param x a RBM object
#' @param layer which RBM to return weights for (usually the final RBM, which will combine all 3 RBMs into a single weight matrix)
#' @param ... not used
#' @export
#' @return a sparse matrix
#' @importFrom Matrix Matrix cBind drop
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
combine_weights.stacked_rbm <- function(x, layer=length(x$rbm_list)){
  x$rbm_list[[1]]$rotation %*% x$rbm_list[[2]]$rotation %*% x$rbm_list[[3]]$rotation
}
