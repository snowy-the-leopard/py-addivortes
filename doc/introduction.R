## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----message=FALSE, warning=FALSE---------------------------------------------
# Load the package
require(AddiVortes)

# Load the Boston Housing dataset from a URL
Boston <- read.csv(paste0("https://raw.githubusercontent.com/anonymous2738/",
                          "AddiVortesAlgorithm/DataSets/BostonHousing_Data.csv"))


# Separate predictors (X) and the response variable (Y)
X_Boston <- as.matrix(Boston[, 2:14])
Y_Boston <- as.numeric(as.matrix(Boston[, 15]))

# Clean up the environment
rm(Boston)

## -----------------------------------------------------------------------------
n <- length(Y_Boston)

# Set a seed for reproducibility
set.seed(1025)

# Create a training set containing 5/6 of the data
TrainSet <- sort(sample.int(n, 5 * n / 6))

# The remaining data will be our test set
TestSet <- setdiff(1:n, TrainSet)

## -----------------------------------------------------------------------------
# Run the AddiVortes algorithm on the training data
results <- AddiVortes(y = Y_Boston[TrainSet], 
                      x = X_Boston[TrainSet, ],
                      m = 200, 
                      totalMCMCIter = 2000, 
                      mcmcBurnIn = 200, 
                      nu = 6, 
                      q = 0.85, 
                      k = 3, 
                      sd = 0.8, 
                      omega = 3, 
                      lambdaRate = 25,
                      IntialSigma = "Linear",
                      showProgress = FALSE)

## -----------------------------------------------------------------------------
# Generate predictions on the test set
preds <- predict(results,
                 X_Boston[TestSet, ],
                 showProgress = FALSE)

# The 8th element of the results object might contain useful info, like variable importance
# For this example, we just display it
cat("In-Sample RMSE:", results$inSampleRmse, "\n")

# Calculate the Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((Y_Boston[TestSet] - preds)^2))
cat("Test Set RMSE:", rmse, "\n")

## ----fig.width=6, fig.height=6, fig.align='center'----------------------------
# Plot true values vs. predicted values
plot(Y_Boston[TestSet],
     preds,
     xlab = "True Values",
     ylab = "Predicted Values",
     main = "AddiVortes Predictions vs True Values",
     xlim = range(c(Y_Boston[TestSet], preds)),
     ylim = range(c(Y_Boston[TestSet], preds)),
     pch = 19, col = "darkblue"
)

# Add the line of equality (y = x) for reference
abline(a = 0, b = 1, col = "darkred", lwd = 2)

# Get quantile predictions to create error bars/intervals
preds_quantile <- predict(
  results,
  X_Boston[TestSet, ],
  "quantile"
)

# Add error segments for each prediction
for (i in 1:nrow(preds_quantile)) {
  segments(Y_Boston[TestSet][i], preds_quantile[i, 1],
           Y_Boston[TestSet][i], preds_quantile[i, 2],
           col = "darkblue", lwd = 1
  )
}
legend("topleft", legend=c("Prediction", "y=x Line", "95% Interval"), 
       col=c("darkblue", "darkred", "darkblue"),
       lty=1, pch=c(19, NA, NA), lwd=2)


