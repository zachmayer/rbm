test_that("Stacked RBMs work correctly", {
  M1 <- random_sparse_matrix(1000, 20, nnz = 200)
  a <- summary(M1)
  expect_identical(dim(M1), c(1000L, 20L))
})