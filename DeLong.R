rm( list = ls( envir = globalenv() ), envir = globalenv() )

library(dplyr)

ROC2 <- read_csv("y_test_and_y_score.csv")


library(ROCR)
library(pROC)
library(rms)
library(tidyverse)


rocmodel1 <- roc(ROC2$y_test,ROC2$y_score1)
rocmodel2 <- roc(ROC2$y_test,ROC2$y_score2)
rocmodel3 <- roc(ROC2$y_test,ROC2$y_score3)
rocmodel4 <- roc(ROC2$y_test,ROC2$y_score4)
plot(rocmodel3, rocmodel2)

roc.test(roc1 = rocmodel1, roc2 = rocmodel2, method = "delong")

roc.test(roc1 = rocmodel1, roc2 = rocmodel3, method = "delong")

roc.test(roc1 = rocmodel1, roc2 = rocmodel4, method = "delong")

roc.test(roc1 = rocmodel2, roc2 = rocmodel3, method = "delong")

roc.test(roc1 = rocmodel2, roc2 = rocmodel4, method = "delong")

roc.test(roc1 = rocmodel3, roc2 = rocmodel4, method = "delong")

