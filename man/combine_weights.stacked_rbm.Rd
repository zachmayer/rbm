\name{combine_weights.stacked_rbm}
\alias{combine_weights.stacked_rbm}
\title{Combine weights from a Stacked Restricted Boltzmann Machine}
\usage{
combine_weights.stacked_rbm(x, layer = length(x$rbm_list))
}
\arguments{
  \item{x}{a RBM object}

  \item{layer}{which RBM to return weights for (usually the
  final RBM, which will combine all 3 RBMs into a single
  weight matrix)}

  \item{...}{not used}
}
\value{
a sparse matrix
}
\description{
This function takes a stacked RBM and returns the combined
weight matrix
}

