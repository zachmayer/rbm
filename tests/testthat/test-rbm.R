test_that("The universe still works", {

  data("movie_reviews")
  data("george_reviews")

  for(args in list(
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = FALSE, retx=FALSE, verbose=FALSE),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = FALSE, retx=FALSE, verbose=TRUE),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = FALSE, retx=TRUE, verbose=FALSE),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = FALSE, retx=TRUE, verbose=TRUE),

    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = TRUE, retx=FALSE, verbose=FALSE, batch_size=1),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = TRUE, retx=FALSE, verbose=TRUE, batch_size=1),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = TRUE, retx=TRUE, verbose=FALSE, batch_size=1),
    a = list(x=movie_reviews, num_hidden=3, max_epochs=10, use_mini_batches = TRUE, retx=TRUE, verbose=TRUE, batch_size=1)

  )){
    model <- do.call(rbm, args)

    for(pred_args in list(
      a = list(object=model, newdata=george_reviews, type='activations', omit_bias=TRUE),
      b = list(object=model, newdata=george_reviews, type='probs', omit_bias=TRUE),
      c = list(object=model, newdata=george_reviews, type='states', omit_bias=TRUE),
      d = list(object=model, newdata=george_reviews, type='activations', omit_bias=FALSE),
      e = list(object=model, newdata=george_reviews, type='probs', omit_bias=FALSE),
      f = list(object=model, newdata=george_reviews, type='states', omit_bias=FALSE),
      a = list(object=model, newdata=movie_reviews, type='activations', omit_bias=TRUE),
      b = list(object=model, newdata=movie_reviews, type='probs', omit_bias=TRUE),
      c = list(object=model, newdata=movie_reviews, type='states', omit_bias=TRUE),
      d = list(object=model, newdata=movie_reviews, type='activations', omit_bias=FALSE),
      e = list(object=model, newdata=movie_reviews, type='probs', omit_bias=FALSE),
      f = list(object=model, newdata=movie_reviews, type='states', omit_bias=FALSE)
    )){
      p <- do.call(predict, pred_args)
      if(pred_args$type == 'activations'){
        expect_is(p@x, 'numeric')
      }
      if(pred_args$type == 'probs'){
        expect_is(p@x, 'numeric')
      }
      if(pred_args$type == 'states'){
        expect_is(p@x, 'logical')
      }
    }

    if(pred_args$omit_bias == TRUE){
      expect_equal(dim(p), c(nrow(pred_args$newdata), ncol(pred_args$object$rotation) - 1))
    }
    if(pred_args$omit_bias == FALSE){
      expect_equal(dim(p), c(nrow(pred_args$newdata), ncol(pred_args$object$rotation)))
    }
  }
})