test_that("Stacked RBMs work correctly", {

  data("movie_reviews")
  data("george_reviews")

  for(args in list(
    a = list(x=movie_reviews, layers=c(3, 3), verbose_stack = TRUE, max_epochs=10, use_mini_batches = FALSE),
    a = list(x=movie_reviews, layers=c(3, 3), verbose_stack = TRUE, max_epochs=10, use_mini_batches = TRUE, batch_size=1),
    a = list(x=movie_reviews, layers=c(3, 3), verbose_stack = FALSE, max_epochs=10, use_mini_batches = FALSE),
    a = list(x=movie_reviews, layers=c(3, 3), verbose_stack = FALSE, max_epochs=10, use_mini_batches = TRUE, batch_size=1),
    a = list(x=george_reviews, layers=c(3, 3), verbose_stack = TRUE, max_epochs=10, use_mini_batches = FALSE),
    a = list(x=george_reviews, layers=c(3, 3), verbose_stack = TRUE, max_epochs=10, use_mini_batches = TRUE, batch_size=1),
    a = list(x=george_reviews, layers=c(3, 3), verbose_stack = FALSE, max_epochs=10, use_mini_batches = FALSE),
    a = list(x=george_reviews, layers=c(3, 3), verbose_stack = FALSE, max_epochs=10, use_mini_batches = TRUE, batch_size=1)
  )){
    model <- do.call(stacked_rbm, args)
    rm(args)
    for(pred_args in list(
      a = list(object=model, newdata=george_reviews, type='states', omit_bias=TRUE),
      b = list(object=model, newdata=george_reviews, type='probs', omit_bias=TRUE),
      d = list(object=model, newdata=george_reviews, type='states', omit_bias=FALSE),
      e = list(object=model, newdata=george_reviews, type='probs', omit_bias=FALSE),
      a = list(object=model, newdata=movie_reviews, type='states', omit_bias=TRUE),
      b = list(object=model, newdata=movie_reviews, type='probs', omit_bias=TRUE),
      d = list(object=model, newdata=movie_reviews, type='states', omit_bias=FALSE),
      e = list(object=model, newdata=movie_reviews, type='probs', omit_bias=FALSE)
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

      rbm_stack <- pred_args$object$rbm_list
      final_rbm <- rbm_stack[[length(rbm_stack)]]
      if(pred_args$omit_bias == TRUE){
        expect_identical(as.numeric(dim(p)), as.numeric(c(nrow(pred_args$newdata), ncol(final_rbm$rotation) - 1)))
      }
      if(pred_args$omit_bias == FALSE){
        expect_identical(as.numeric(dim(p)), as.numeric(c(nrow(pred_args$newdata), ncol(final_rbm$rotation))))
      }
      rm(pred_args)
    }
  }
})