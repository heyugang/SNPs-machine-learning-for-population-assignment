#####################################################################################
## SNP datasets selected using the mRMR method were used for breed assignment through four machine learning classifiers.
## Note:The script displays the execution code for the first training set only, with the remaining four executed in a similar manner.

## ===============================================================
## step1: Load the computing environment.
## ===============================================================
setwd("/example/path/to/your/project")  # Set the working directory to the project path
library(dplyr)  # Load the dplyr library for data manipulation
library(caret)  # Load the caret library for model training and cross-validation
library(class)  # Load the class library for KNN classification
library(e1071)  # Load the e1071 library for Naive Bayes and SVM
library(mRMRe)  # Load the mRMRe library for mutual information-based feature selection
library(CORElearn)  # (Commented) Load the CORElearn library for machine learning models

## ===============================================================
## step2 Load data and perform preprocessing
## ===============================================================
# Prepare some vectors that will be used later
trash <- c("FID", "PAT", "MAT", "SEX", "PHENOTYPE")  # Columns to be excluded from analysis
non_numeric_columns <- c("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")  # Columns that are not numeric
snp_num <- c(100, 250, 500, 800, 1000, 2000, 3000, 5000, 10000)  # Different SNP numbers for analysis
excludevars <- c("ID", "IID", "num", "group")  # Columns to exclude when extracting features
excludevars_all <- c("ID", "IID", "num", "group", "grouplevels")  # More columns to exclude
ctrl <- trainControl(method = "cv", number = 10)  # Cross-validation control for 10-fold cross-validation
grid <- expand.grid(.laplace = seq(0, 1, 0.1), .usekernel = c(TRUE, FALSE), .adjust = 1)  # Grid for tuning Naive Bayes model

# Read input files
data <- read.table(file = 'Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k.raw', header = TRUE, sep = '')  # Read the raw data file
group <- read.table(file = "buffalo_3col.list", header = FALSE, sep = "")  # Read the group list file
names(group) <- c("IID", "ID", "group")  # Set column names for the group data
group$num <- 1:nrow(group)  # Create a numerical index for each row
group$group <- as.factor(group$group)  # Convert the 'group' column to a factor for classification
load("trainset_list.RData")  # Load the training set list from the previously saved RData file

# Initialize lists for storing training and testing sets
trainingsets <- list()
testingsets <- list()

# Loop through the 5 sets of data
for (i in 1:5) {
    # Extract the training set
    train <- number_list[[i]]  # Get the list of IDs for the current training set
    trainingset <- group[group$num %in% train, ]  # Subset the group data for the training set
    trainingset <- trainingset[order(trainingset$num, decreasing = FALSE), ]  # Sort the training set by 'num'

    # Extract the testing set
    testingset <- subset(group, !(num %in% trainingset$num))  # Subset the group data for the testing set

    # Store the training and testing sets in lists
    trainingsets[[i]] <- trainingset
    testingsets[[i]] <- testingset
}

# Data processing
data1 <- data  # Create a copy of the original data
numeric_columns_tmp <- setdiff(names(data), non_numeric_columns)  # Get names of numeric columns excluding non-numeric ones

# Convert these numeric columns from integer to numeric type
data1[numeric_columns_tmp] <- lapply(data1[numeric_columns_tmp], function(x) as.numeric(as.character(x)))
data1 <- data1[, !(names(data1) %in% trash)]  # Remove the 'trash' columns from the data
names <- colnames(data1)  # Get the column names of the processed data
names1 <- gsub(pattern = "X(\\d+)\\.(\\d+)_.*", replacement = "\\1:\\2", names)  # Format column names
colnames(data1) <- names1  # Assign the new formatted column names to the data


## ===============================================================
## step3 Perform population classification using 4 machine learning methods for mrmr
## ===============================================================
# Prepare vectors to store classification accuracy for each method
acc_mrmr1_svm <- numeric(9)  # Store accuracy for SVM
acc_mrmr1_knn <- numeric(9)  # Store accuracy for KNN
acc_mrmr1_nb <- numeric(9)   # Store accuracy for Naive Bayes
acc_mrmr1_rf <- numeric(9)   # Store accuracy for Random Forest
mrmr_all_1list <- list()     # List to store SNPs selected by MRMR

# Loop through 9 different SNP subsets
trainings <- trainingsets[[1]]  # Extract the first training set
testings <- testingsets[[1]]    # Extract the first testing set
trainings$group <- as.factor(trainings$group)  # Convert 'group' column to factor for classification
testings$group <- as.factor(testings$group)    # Convert 'group' column to factor for testing
trainings$grouplevels <- as.factor(trainings$group)  # Add a new 'grouplevels' column
levels(trainings$grouplevels) <- 1:nlevels(trainings$grouplevels)  # Assign levels to 'grouplevels'
trainings$grouplevels <- as.ordered(trainings$grouplevels)  # Convert 'grouplevels' to ordered factor
try <- merge(data1, trainings, by = "IID")  # Merge data with training set by IID
try <- try[, !(names(try) %in% excludevars)]  # Remove excluded columns
try <- try %>% mutate_at(vars(-ncol(try)), ~as.factor(as.numeric(.) + 1))  # Convert all columns except the last to factors
try <- try %>% mutate_at(vars(-ncol(try)), ~ if (is.factor(.)) ordered(.) else .)  # Convert factors to ordered factors
mrmr_data <- mRMR.data(data = try)  # Create mRMR data object

for (n in 1:9) {  # Loop over each SNP subset size
  mrmr.solution <- mRMR.classic(data = mrmr_data, target_indices = 39997, feature_count = snp_num[[n]])  # Run MRMR algorithm
  selected_features <- solutions(mrmr.solution)[[1]]  # Get the selected features from MRMR solution
  selected_try <- try[, selected_features]  # Subset the data with the selected features
  snp <- colnames(selected_try)  # Extract SNP names
  snp <- append("IID", snp)  # Add IID to the SNP list
  tryall <- data1[, colnames(data1) %in% snp]  # Subset data based on selected SNPs
  traindata <- merge(tryall, trainings, by = "IID")  # Merge training data with the selected SNP data
  testdata <- merge(tryall, testings, by = "IID")    # Merge testing data with the selected SNP data
  traindata_features <- traindata[, !(names(traindata) %in% excludevars_all)]  # Extract features for training
  traindata_labels <- traindata[["group"]]  # Extract labels for training
  testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing
  testdata_labels <- testdata[["group"]]   # Extract labels for testing

  # SVM (Support Vector Machine)
  set.seed(77)  # Set seed for reproducibility
  tunemodel <- tune.svm(x = traindata_features, y = traindata_labels, data = traindata, kernel = "linear", gamma = 10^(-5:5), cost = 2^(1:9))  # Tune SVM model
  set.seed(77)  # Set seed for reproducibility
  svm <- svm(x = traindata_features, y = traindata_labels, kernel = "linear", gamma = tunemodel$best.parameters$gamma, cost = tunemodel$best.parameters$cost)  # Train SVM model
  pred_svm <- predict(svm, testdata_features)  # Predict using SVM
  acc_mrmr1_svm[n] <- sum(pred_svm == testdata_labels) / length(testdata_labels)  # Compute SVM accuracy

  # KNN (K-Nearest Neighbors)
  folds <- createFolds(traindata_labels, k = 15, list = TRUE, returnTrain = TRUE)  # Create 15-fold cross-validation
  best_k <- 1  # Initialize best k value
  best_accuracy <- 0  # Initialize best accuracy
  for (k in 1:15) {  # Loop through values of k (1 to 15)
    fold_accuracy <- 0  # Initialize fold accuracy
    for (i in 1:15) {   # Loop through each fold
      train_indices <- folds[[i]]  # Get training indices for the current fold
      train_data <- traindata_features[train_indices, ]  # Extract training data
      train_labels <- traindata_labels[train_indices]  # Extract training labels
      test_data <- traindata_features[-train_indices, ]  # Extract testing data
      test_labels <- traindata_labels[-train_indices]  # Extract testing labels
      predictions <- class::knn(train = train_data, test = test_data, cl = train_labels, k = k)  # KNN prediction
      fold_accuracy <- fold_accuracy + sum(predictions == test_labels) / length(test_labels)  # Compute accuracy for the fold
    }
    fold_accuracy <- fold_accuracy / 10  # Average fold accuracy
    if (fold_accuracy > best_accuracy) {  # Update best k value if accuracy improves
      best_k <- k
      best_accuracy <- fold_accuracy
    }
  }
  k_value <- best_k  # Set the best k value based on cross-validation
  set.seed(77)  # Set seed for reproducibility
  pred_knn <- class::knn(traindata_features, testdata_features, traindata_labels, k = k_value)  # KNN prediction
  acc_mrmr1_knn[n] <- sum(pred_knn == testdata_labels) / length(testdata_labels)  # Compute KNN accuracy

  # Naive Bayes (NB)
  tryall <- tryall %>% mutate_all(~ if (is.numeric(.)) as.factor(.) else .)  # Convert numeric columns to factors for Naive Bayes
  traindata <- merge(tryall, trainings, by = "IID")  # Merge training data
  testdata <- merge(tryall, testings, by = "IID")    # Merge testing data
  traindata_features <- traindata[, !(names(traindata) %in% excludevars_all)]  # Extract features for training
  traindata_labels <- traindata[["group"]]           # Extract labels for training
  testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing
  testdata_labels <- testdata[["group"]]             # Extract labels for testing
  set.seed(77)  # Set seed for reproducibility
  model <- train(x = traindata_features, y = traindata_labels, method = "naive_bayes", trControl = ctrl, tuneGrid = grid)  # Train Naive Bayes model
  set.seed(77)  # Set seed for reproducibility
  nb <- e1071::naiveBayes(x = traindata_features, y = traindata_labels, laplace = model$bestTune$laplace)  # Train Naive Bayes model with best parameters
  pred_nb <- predict(nb, testdata_features)  # Naive Bayes prediction
  acc_mrmr1_nb[n] <- sum(pred_nb == testdata_labels) / length(testdata_labels)  # Compute Naive Bayes accuracy

  # Save the selected SNPs for the current iteration
  mrmr_all_1list[[paste0("snp_mrmr_", n)]] <- snp
  rm(mrmr.solution)  # Clean up the MRMR solution
}

## ===============================================================
## step4: Save the accuracy results for all methods in a data frame
## ===============================================================
acc_mrmr1_9 <- data.frame(snp_num, acc_mrmr1_knn, acc_mrmr1_rf, acc_mrmr1_svm, acc_mrmr1_nb)  # Store accuracy for all methods
save(acc_mrmr1_9, file = "acc_mrmr1_10.RData")  # Classification accuracy files containing the performance of four machine learning classifiers based on SNP lists of varying sizes
save(mrmr_all_1list, file = "snp_mrmr1_list.RData")  # SNP lists selected by the mRMR method, which contains SNP lists with varying sizes
