#' Fit a Stack of Restricted Boltzmann Machines
#' 
#' @param x a sparse matrix
#' @param num_hidden an integer vector of the number of neurons in each RBM
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
#' @return a stacked_rbm object
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
stacked_rbm <- function (x, num_hidden = c(30, 100, 30), max_epochs = 1000, learning_rate = 0.1, use_mini_batches = FALSE, batch_size = 250, initial_weights_mean = 0, initial_weights_sd = 0.1, momentum = 0, dropout = FALSE, dropout_pct = .50, retx = FALSE, activation_function=NULL, verbose = FALSE, ...) {
  require('Matrix')
  stop('not implemented')
  
  class(out) <- 'stacked_rbm'
  return(out)
}