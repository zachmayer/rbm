\name{stacked_rbm}
\alias{stacked_rbm}
\title{Fit a Stack of Restricted Boltzmann Machines}
\usage{
stacked_rbm(x, num_hidden = c(30, 100, 30), verbose_stack = TRUE, ...)
}
\arguments{
  \item{x}{a sparse matrix}

  \item{num_hidden}{an integer vector of the number of
  neurons in each RBM}

  \item{...}{passed to the rbm function}
}
\value{
  a stacked_rbm object
}
\description{
  Fit a Stack of Restricted Boltzmann Machines
}
\examples{
#Setup a dataset
set.seed(10)
print('Data from: https://github.com/echen/restricted-boltzmann-machines')
Alice <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
Bob <- c('Harry_Potter' = 1, Avatar = 0, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #SF/fantasy fan, but doesn't like Avatar.
Carol <- c('Harry_Potter' = 1, Avatar = 1, 'LOTR3' = 1, Gladiator = 0, Titanic = 0, Glitter = 0) #Big SF/fantasy fan.
David <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
Eric <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 0, Glitter = 0) #Oscar winners fan, except for Titanic.
Fred <- c('Harry_Potter' = 0, Avatar = 0, 'LOTR3' = 1, Gladiator = 1, Titanic = 1, Glitter = 0) #Big Oscar winners fan.
dat <- rbind(Alice, Bob, Carol, David, Eric, Fred)

Stacked_RBM <- stacked_rbm(dat)
}
\references{
  \itemize{ \item
  \url{http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines}
  }
}

