#Goto https://www.kaggle.com/c/digit-recognizer/data
#Download https://www.kaggle.com/c/digit-recognizer/download/train.csv
library(data.table)
digits <- fread('~/Downloads/train.csv')
digits <- as.matrix(digits)
devtools::use_data(digits, overwrite=TRUE, compress='xz')
