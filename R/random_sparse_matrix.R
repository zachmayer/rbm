#' @title Random Sparse Matrix
#' @param nrow number of rows in matrix
#' @param ncol number of cols in matrix
#' @param nnz number of non-zero entries
#' @param rfunc random function to use to generate the non-zero entries
#' @param rand.x random number generator for 'x' slot
#' @param ... optionally further arguments passed to sparseMatrix()
#' @return a sparseMatrix of dimension (nrow, ncol)
#' @author Martin Maechler, Zach Mayer
#' @importFrom Matrix sparseMatrix
#' @references
#' \itemize{
#' \item \link[Matrix]{sparseMatrix}
#' \item \url{http://www.inside-r.org/r-doc/Matrix/sparseMatrix}
#' }
#' @export
#' @examples
#' M1 <- random_sparse_matrix(1000, 20, nnz = 200)
#' summary(M1)
random_sparse_matrix <- function(
  nrow, ncol, nnz,
  rfunc = rnorm,
  rand.x = function(n) round(rfunc(nnz), 2), ...){
  stopifnot(
    (nnz <- as.integer(nnz)) >= 0,
    nrow >= 0,
    ncol >= 0,
    nnz <= nrow * ncol
    )
  sparseMatrix(
    i = sample(nrow, nnz, replace = TRUE),
    j = sample(ncol, nnz, replace = TRUE),
    x = rand.x(nnz), dims = c(nrow, ncol), ...)
}

