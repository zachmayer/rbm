#' Fit a Stack of Restricted Boltzmann Machines
#' 
#' @param x a sparse matrix
#' @param num_hidden an integer vector of the number of neurons in each RBM
#' @param ... passed to the rbm function
#' @export
#' @return a stacked_rbm object
#' @references
#' \itemize{
#' \item \url{http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines}
#' }
#' @examples
#' #Setup a dataset
#' set.seed(10)
#' print('Data from: https://github.com/echen/restricted-boltzmann-machines')
#' Alice <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
#' Bob <- c('Harry_Potter' = 1, Avatar = 0, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #SF/fantasy fan, but doesn't like Avatar.
#' Carol <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
#' David <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
#' Eric <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 0, Glitter = 0) #Oscar winners fan, except for Titanic.
#' Fred <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
#' dat <- rbind(Alice, Bob, Carol, David, Eric, Fred)
#' 
#' Stacked_RBM <- stacked_rbm(dat)
stacked_rbm <- function (x, num_hidden = c(30, 100, 30), verbose_stack=TRUE, ...) {
  require('Matrix')
  
  #Checks
  stopifnot(length(dim(x)) == 2)
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
  
  if(length(num_hidden) < 2){
    stop('Please use the rbm function to fit a single rbm')
  }
  
  #Fit first RBM
  if(verbose_stack){print('Fitting RBM 1')}
  rbm_list <- as.list(num_hidden)
  rbm_list[[1]] <- rbm(x, num_hidden=num_hidden[[1]], retx=TRUE, ...)

  #Fit the rest of the RBMs
  for(i in 2:length(rbm_list)){
    if(verbose_stack){print(paste('Fitting RBM', i))}
    rbm_list[[i]] <- rbm(predict(rbm_list[[i-1]], type='probs', omit_bias=TRUE), num_hidden=num_hidden[[i]], retx=TRUE, ...)
  }
  
  #Return result
  out <- list(rbm_list=rbm_list, num_hidden=num_hidden, activation_function=rbm_list[[1]]$activation_function)
  class(out) <- 'stacked_rbm'
  return(out)
}

#' Predict from a Stacked Restricted Boltzmann Machine
#' 
#' This function takes a stacked RBM and a matrix of new data, and predicts for the new data with the RBM.
#' 
#' @param x a RBM object
#' @param newdata a sparse matrix of new data
#' @param type a character vector specifying whether to return the hidden unit activations, hidden unit probs, or hidden unit states.  Activations or probabilities are typically the most useful if you wish to use the RBM features as input to another predictive model (or another RBM!).  Note that the hidden states are stochastic, and may be different each time you run the predict function, unless you set random.seed() before making predictions.  Activations and states are non-stochastic, and will be the same each time you run predict.
#' @param ... not used
#' @export
#' @return a sparse matrix
predict.stacked_rbm <- function (object, newdata, type='probs', omit_bias=TRUE, ...) {
  require('Matrix')

  #If no new data, just return predictions from the final rbm in the stack
  if (missing(newdata)) {
    if (!is.null(object$x)) {
      predict(object$rbm_list[[length(object$rbm_list)]], type=type, omit_bias=omit_bias)
    }
  } else {
    if(type %in% c('probs', 'states')){
      stop('Currently we can only return hidden probabilities or states from a stacked rbm.  Activations are not yet supported')
    }
    hidden_probs <- predict(object$rbm_list[[1]], newdata=newdata, type='probs', omit_bias=TRUE)
    for(i in 2:length(object$rbm_list)){
      hidden_probs <- predict(object$rbm_list[[i]], newdata=hidden_probs, type='probs', omit_bias=TRUE)
    }
  }
  
  if(omit_bias){
    if(type=='probs'){return(hidden_probs[,-1])}
    hidden_states <- hidden_probs > Matrix(runif(nrow(hidden_probs)*ncol(hidden_probs)), nrow=nrow(hidden_probs), ncol=ncol(hidden_probs))
    return(hidden_states[,-1])
  } else{
    if(type=='probs'){return(hidden_probs)}
    hidden_states <- hidden_probs > Matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states)
  }
  
}
