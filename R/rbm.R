#' Fit a Restricted Boltzmann Machine
#' 
#' This function fits an RBM to the input dataset.  It internally uses sparse matricies for faster matrix operations
#' 
#' This code is (mostly) adapted from edwin chen's python code for RBMs, avaiable here: https://github.com/echen/restricted-boltzmann-machines.  Some modifications (e.g. momentum) were adapted from Andrew Landgraf's R code for RBMs, available here: http://alandgraf.blogspot.com/2013/01/restricted-boltzmann-machines-in-r.html.
#'
#' @param x a sparse matrix
#' @param num_hidden number of neurons in the hidden layer
#' @param max_epochs 
#' @param learning_rate 
#' @param use_mini_batches 
#' @param batch_size
#' @param initial_weights_mean 
#' @param initial_weights_sd 
#' @param momentum 
#' @param dropout 
#' @param retx whether to return the RBM predictions for the input data
#' @param verbose 
#' @param activation_function function to convert hidden activations (-Inf, Inf) to hidden probabilities [0, 1].  Must be able to operate on sparse "Matrix" objects.
#' @param ... not used
#' @export
#' @return a RBM object
#' @references
#' \itemize{
#' \item \url{http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines}
#' \item \url{https://github.com/echen/restricted-boltzmann-machines}
#' \item \url{http://alandgraf.blogspot.com/2013/01/restricted-boltzmann-machines-in-r.html}
#' \item \url{http://web.info.uvt.ro/~dzaharie/cne2013/proiecte/tehnici/DeepLearning/DL_tutorialSlides.pdf}
#' \item \url{http://deeplearning.net/tutorial/rbm.html}
#' \item \url{http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf}
#' \item \url{http://www.cs.toronto.edu/~nitish/msc_thesis.pdf}
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
#' #Fit a PCA model and an RBM model
#' PCA <- prcomp(dat, retx=TRUE)
#' RBM <- rbm(dat, retx=TRUE)
#' 
#' #Examine the 2 models
#' round(PCA$rotation, 2) #PCA weights
#' round(RBM$rotation, 2) #RBM weights
#' 
#' #Predict for new data
#' George <- as.matrix(t(c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 0, Gladiator = 1, Titanic = 1, Glitter = 0)))
#' predict(PCA, George)
#' predict(RBM, George, type='activations')
#' predict(RBM, George, type='probs')
#' predict(RBM, George, type='states')
#' 
#' #Predict for existing data
#' predict(PCA)
#' predict(RBM, type='probs')
rbm <- function (x, num_hidden = 2, max_epochs = 1000, learning_rate = 0.1, use_mini_batches = FALSE, batch_size = 250, initial_weights_mean = 0, initial_weights_sd = 0.1, momentum = 0, dropout = FALSE, dropout_pct = .50, retx = FALSE, activation_function=NULL, verbose = FALSE, ...) {
  require('Matrix')
  #stop('not implemented')
  
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
  
  stopifnot(is.numeric(momentum))
  stopifnot(momentum >= 0 & momentum <=1)
  if(momentum>0){warning('Momentum > 0 not yet implemented.  Ignoring momentum')}
  
  stopifnot(is.numeric(dropout_pct))
  stopifnot(dropout_pct >= 0 & dropout_pct <1)
  if(dropout){warning('Dropout not yet implemented')}
  
  if(is.null(activation_function)){
    activation_function <- function(x){1.0 / (1 + exp(-x))}
  }
  
  # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
  # a Gaussian distribution with mean 0 and standard deviation 0.1.
  #momentum_speed <- sparseMatrix(1, 1, x=0, dims=c(p, num_hidden))
  weights = matrix(rnorm(num_hidden*ncol(x), mean=initial_weights_mean, sd=initial_weights_sd), nrow=ncol(x), ncol=num_hidden)
  # Insert weights for the bias units into the first row and first column.
  weights = cbind(0, weights)
  weights = rbind(0, weights)
  weights = Matrix(weights, sparse=TRUE)
  
  # Insert bias units of 1 into the first column.
  x <- cBind(Bias_Unit=1, x)
  dimnames(weights) = list(colnames(x), c('Bias_Unit', paste('Hidden', 1:num_hidden, sep='_')))
  
  #Fit the model
  x <- drop0(x)
  for (epoch in 1:max_epochs){
    
    #Sample mini-batch
    if(use_mini_batches){
      train_rows = sample(1:nrow(x), batch_size, replace=TRUE)
      x_sample = x[train_rows,]
    } else {
      x_sample = x
    }

    # Clamp to the data and sample from the hidden units. 
    # (This is the "positive CD phase", aka the reality phase.)
    pos_hidden_activations = x_sample %*% weights
    if(dropout){
      pos_hidden_activations_dropped = pos_hidden_activations
      pos_hidden_activations_dropped@x[runif(length(pos_hidden_activations_dropped@x)) < dropout_pct] = 0
      pos_hidden_activations_dropped[,1] <- pos_hidden_activations[,1]
      pos_hidden_activations <- pos_hidden_activations_dropped
    } 
    pos_hidden_probs = activation_function(pos_hidden_activations)
    pos_hidden_states = pos_hidden_probs > Matrix(runif(nrow(x_sample)*(num_hidden+1)), nrow=nrow(x_sample), ncol=(num_hidden+1))
    
    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
    pos_associations = Matrix:::crossprod(x_sample, pos_hidden_probs)
    
    # Reconstruct the visible units and sample again from the hidden units.
    # (This is the "negative CD phase", aka the daydreaming phase.)
    neg_visible_activations = Matrix:::tcrossprod(pos_hidden_states, weights)
    neg_visible_probs = activation_function(neg_visible_activations)
    neg_visible_probs[,1] = 1 # Fix the bias unit.
    neg_hidden_activations = neg_visible_probs %*% weights
    neg_hidden_probs = activation_function(neg_hidden_activations)
    
    # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
    # themselves.
    neg_associations = Matrix:::crossprod(neg_visible_probs, neg_hidden_probs)
    
    # Update weights
    weights = weights + learning_rate * ((pos_associations - neg_associations) / nrow(x_sample))
    if(verbose){
      error = sum((x - neg_visible_probs) ^ 2)
      print(sprintf("Epoch %s: error is %s", epoch, error))
    }
  }   
  
  #Return output
  if(retx){
    output_x <- x %*% weights
  } else {
    output_x <- NULL
  }
  out <- list(rotation=weights, activation_function=activation_function, x=output_x)
  class(out) <- 'RBM'
  return(out)
}

#' Print method for a Restricted Boltzmann Machine
#' 
#' This function prints the weights for a RBM
#'  
#' @param x a RBM object
#' @param ... not used
#' @export
print.RBM <- function (object, ...) {
  print(object$rotation) 
}

#' Predict from a Restricted Boltzmann Machine
#' 
#' This function takes an RBM and a matrix of new data, and predicts for the new data with the RBM.
#' @param x a RBM object
#' @param newdata a sparse matrix of new data
#' @param type a character vector specifying whether to return the hidden unit activations, hidden unit probs, or hidden unit states.  Activations or probabilities are typically the most useful if you wish to use the RBM features as input to another predictive model (or another RBM!).  Note that the hidden states are stochastic, and may be different each time you run the predict function, unless you set random.seed() before making predictions.  Activations and states are non-stochastic, and will be the same each time you run predict.
#' @param ... not used
#' @export
#' @return a sparse matrix
predict.RBM <- function (object, newdata, type='probs', ...) {
  require('Matrix')
  if (missing(newdata)) {
    if (!is.null(object$x)) {
      hidden_activations <- object$x
      rows <- nrow(object$x)
    }
    else stop("no scores are available: refit with 'retx=TRUE'")
  } else {
    #Checks
    stopifnot(length(dim(newdata)) == 2)
    stopifnot(type %in% c('activations', 'probs', 'states'))
    if(any('data.frame' %in% class(newdata))){
      if(any(!sapply(newdata, is.numeric))){
        stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
      }
      x = Matrix(as.matrix(newdata), sparse=TRUE)
    } else if (any('matrix' %in% class(newdata))){
      x = Matrix(newdata, sparse=TRUE)
    } else if(length(attr(class(newdata), 'package')) != 1){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    } else if(attr(class(newdata), 'package') != 'Matrix'){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    }
    
    # Insert bias units of 1 into the first column.
    newdata <- cBind(Bias_Unit=rep(1, nrow(newdata)), newdata)
    
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
    hidden_activations <- newdata %*% object$rotation
    rows <- nrow(newdata)
  }

  if(type=='activations'){return(hidden_activations)}
  hidden_probs <- object$activation_function(hidden_activations)
  if(type=='probs'){return(hidden_probs)}
  hidden_states <- hidden_probs > Matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
  return(hidden_states)
  
}
