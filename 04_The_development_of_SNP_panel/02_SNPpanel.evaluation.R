################################################################
## Population assignment test and identification of the best-performing SNP set using R

## ===============================================================
## step1: Load the computing environment.
## ===============================================================
setwd("/example/path/to/your/data")  # Set the working directory to the specified path
library(caret)      # For machine learning and confusion matrix
library(dplyr)      # For data manipulation
library(reshape2)   # For reshaping data (if needed)
library(ggplot2)    # For plotting
library(e1071)      # For SVM classifier and tuning

## ===============================================================
## step2: Load data and perform preprocessing
## ===============================================================
# Read input data and preprocessing
trash <- c("FID", "PAT", "MAT", "SEX", "PHENOTYPE")  # Columns to remove
excludevars <- c("ID", "IID", "num", "group")        # Metadata columns
group <- read.table(file = "buffalo_3col.list", header = FALSE, sep = "")  # Group info
data <- read.table(file = "Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k.raw", header = TRUE, sep = "")  # Genotype data

# Rename group columns
names(group) <- c("IID", "ID", "group")
group$num <- 1:nrow(group)
group$group <- as.factor(group$group)  # Convert group to factor

# Remove unnecessary columns from genotype data
data <- data[, !(names(data) %in% trash)]

# Rename genotype columns to "chr:pos" format
names <- colnames(data)
names1 <- gsub(pattern = "X(\\d+)\\.(\\d+)_.*", replacement = "\\1:\\2", names)
colnames(data) <- names1

# Randomly sample training set indices and save sets
set.seed(24427)
number <- sample(1:26, 19)
number <- append(number, sample(27:46, 15))
number <- append(number, sample(47:66, 15))
number <- append(number, sample(67:91, 18))
number <- append(number, sample(92:115, 18))
number <- append(number, sample(116:137, 16))
number <- append(number, sample(138:162, 18))
number <- append(number, sample(163:187, 18))

# Prepare training and testing groups
breed_training <- group[number, ]
breed_training <- breed_training[order(breed_training$num, decreasing = FALSE), ]
breed_testing <- subset(group, !(num %in% breed_training$num))

# Save the training/testing dataset list
snptest_dataframelist <- list()
snptest_dataframelist[[1]] <- breed_training
snptest_dataframelist[[2]] <- breed_testing
save(snptest_dataframelist, file = "dataframetest_list.RData")

# Load training and testing sets
load("dataframetest_list.RData")
trainings <- snptest_dataframelist[[1]]
testings <- snptest_dataframelist[[2]]

# Load selected SNPs from mRMR overlap list
load("mrmr_SNPoverlap_9.RData")
mrmr_snp <- mrmr_overlap[[1]]  # SNPs selected by mRMR method (100 SNPs as example)

# Filter genotype data based on selected SNPs
tryall <- data[, colnames(data) %in% mrmr_snp]

# Merge with training and testing sample info
traindata <- merge(tryall, trainings, by = "IID")
testdata <- merge(tryall, testings, by = "IID")

# Extract features and labels
traindata_features <- traindata[, !(names(traindata) %in% excludevars)]
traindata_labels <- traindata[["group"]]
testdata_features <- testdata[, !(names(testdata) %in% excludevars)]
testdata_labels <- testdata[["group"]]

## ===============================================================
## step3 SVM classification process
## ===============================================================
# Tune SVM hyperparameters using grid search
set.seed(77)
tunemodel <- tune.svm(x = traindata_features, y = traindata_labels, data = traindata,
                      kernel = "linear", gamma = 10^(-5:5), cost = 2^(1:9))
tunemodel$best.parameters  # Output best gamma and cost

# Train SVM model with optimal parameters
set.seed(77)
svm_try <- svm(x = traindata_features, y = traindata_labels, kernel = "linear",
               gamma = tunemodel$best.parameters$gamma,
               cost = tunemodel$best.parameters$cost,
               probability = TRUE)

# Predict on test data
pred_svm_try <- predict(svm_try, testdata_features, probability = TRUE)

# Generate confusion matrix
cofff <- confusionMatrix(pred_svm_try, testdata$group)
conf_matrix <- as.matrix(cofff$table)  # Confusion matrix counts
conf_matrix_pro <- prop.table(conf_matrix, 2)  # Column-wise proportions
confusiondata <- as.data.frame(conf_matrix)  # Data frame format

# ROC curve analysis
pro <- attr(pred_svm_try, "probabilities")[, 2]  # Get predicted probabilities
library(pROC)
roc <- roc(testdata_labels, pro)  # Compute ROC
plot(roc)  # Plot ROC curve
aucnum <- auc(roc)  # Compute AUC

# Add AUC and Accuracy text to plot
text(x = 0.3, y = 0.15, labels = paste("AUC =", round(aucnum, 2)))
text(x = 0.3, y = 0.25, labels = paste("Accuracy =", sprintf("%.2f", cofff$overall[1])))

## ===============================================================
## step5 Save classification result 
## ===============================================================
save(acc_mrmr_over, file = "acc_mrmr_overlap.RData") # Classification accuracy of the intersected SNP panels, which contains classification accuracy of mRMR classifiers based on SNP lists of varying sizes
