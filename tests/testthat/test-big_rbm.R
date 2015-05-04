test_that("A large, sparse RBM runs efficiently", {

  skip_on_cran()

  set.seed(627)
  x <- random_sparse_matrix(
    nrow=500000,
    ncol=5000,
    nnz=1e+03,
    rfunc=runif
  )
  x@x <- x@x / max(x@x)

  #X cannot be represented by a dense matrix
  #x_mat <- as.matrix(x)

  #Fit a RBM (should be pretty quick)
  t1 <- system.time(model1 <- rbm(x, num_hidden=3, max_epochs=100, use_mini_batches=TRUE, batch_size=50))
  expect_less_than(t1[['elapsed']], 10)

  #Fit a stack (should also be very quick)
  t2 <- system.time(model2 <- stacked_rbm(x, layers=c(5, 5, 5), max_epochs=100, use_mini_batches=TRUE, batch_size=50, verbose_stack=TRUE))
  expect_less_than(t2[['elapsed']], 30)

  #Predict through the stack (should be expremely quick)
  t3 <- system.time(p <- predict(model2, newdata=x))
  expect_less_than(t3[['elapsed']], 5)

})