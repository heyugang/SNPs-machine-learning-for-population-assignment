#####################################################################################
## SNP datasets selected using the Relief method were employed for breed assignment using four machine learning classifiers.
## Note:The script displays the execution code for the first training set only, with the remaining four executed in a similar manner.

## ===============================================================
## step1: Load the computing environment.
## ===============================================================
setwd("/example/path/to/your/project")  # Set the working directory to the project path
library(dplyr)  # Load dplyr for data manipulation
library(caret)  # Load caret for machine learning and cross-validation
library(class)  # Load class for KNN classifier
library(e1071)  # Load e1071 for SVM and Naive Bayes classifiers
library(CORElearn)  # Load CORElearn for machine learning models

## ===============================================================
## step2 Load data and perform preprocessing
## ===============================================================
# Prepare values
ctrl <- trainControl(method = "cv", number = 10)  # Set up cross-validation with 10 folds
grid <- expand.grid(.laplace = seq(0, 1, 0.1), .usekernel = c(TRUE, FALSE), .adjust = 1)  # Set grid for Naive Bayes parameter tuning
trash <- c("FID", "PAT", "MAT", "SEX", "PHENOTYPE")  # Columns to be excluded from analysis
excludevars <- c("ID", "IID", "num", "group")  # Columns to exclude when extracting features
col_del <- c("IID", "ID", "num")  # Columns to delete from the data
snp_num <- c(100, 250, 500, 800, 1000, 2000, 3000, 5000, 10000, 15000)  # SNP numbers for different subsets

# Read the data files
data <- read.table(file = 'Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k.raw', header = TRUE, sep = '')  # Read the raw data file
group <- read.table(file = "buffalo_3col.list", header = FALSE, sep = "")  # Read the group list file
names(group) <- c("IID", "ID", "group")  # Set column names for the group data
group$num <- 1:nrow(group)  # Create a numerical index for each row
group$group <- as.factor(group$group)  # Convert the 'group' column to a factor for classification

# Load previously saved training set list
load("trainset_list.RData")

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
data <- data[, !(names(data) %in% trash)]  # Remove columns that are in the 'trash' list
names <- colnames(data)  # Get column names of the data
names1 <- gsub(pattern = "X(\\d+)\\.(\\d+)_.*", replacement = "\\1:\\2", names)  # Format column names by extracting relevant information
colnames(data) <- names1  # Assign the new formatted column names to the data

## ===============================================================
## step3 Perform population classification using 4 machine learning methods for relf
## ===============================================================
# Read data
acc_relf1_svm <- numeric(10)  # Store SVM accuracy
acc_relf1_knn <- numeric(10)  # Store KNN accuracy
acc_relf1_nb <- numeric(10)   # Store Naive Bayes accuracy
acc_relf1_rf <- numeric(10)   # Store Random Forest accuracy
relf_1list <- list()          # List to store selected features

# Prepare the training and testing sets
trainings <- trainingsets[[1]]  # Extract the first training set
testings <- testingsets[[1]]    # Extract the first testing set
trainings$group <- as.factor(trainings$group)  # Convert 'group' to factor for classification
testings$group <- as.factor(testings$group)    # Convert 'group' to factor for testing

# Apply ReliefF method for feature selection and perform KNN, SVM, and Naive Bayes classification in a loop
try <- merge(data, trainings, by = "IID")  # Merge data with training set by IID
try <- try[, !(names(try) %in% col_del)]  # Remove unnecessary columns
result <- attrEval(group ~ ., data = try, estimator = "ReliefFbestK")  # Apply ReliefF feature selection

for (n in 1:10) {  # Loop through different feature numbers (snp_num)
    N <- snp_num[[n]]  # Get the number of top features to select
    top_features <- names(sort(result, decreasing = TRUE)[1:N])  # Select top N features
    top_features <- append("IID", top_features)  # Add IID to the list of selected features
    tryall <- data[, top_features]  # Subset data with the selected features
    traindata <- merge(tryall, trainings, by = "IID")  # Merge selected features with training set
    testdata <- merge(tryall, testings, by = "IID")    # Merge selected features with testing set
    traindata_features <- traindata[, !(names(traindata) %in% excludevars)]  # Extract features for training
    traindata_labels <- traindata[["group"]]  # Extract labels for training
    testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing
    testdata_labels <- testdata[["group"]]   # Extract labels for testing

    # SVM (Support Vector Machine)
    set.seed(77)  # Set seed for reproducibility
    tunemodel <- tune.svm(x = traindata_features, y = traindata_labels, data = traindata, kernel = "linear", gamma = 10^(-5:5), cost = 2^(1:9))  # Tune SVM model
    set.seed(77)  # Set seed for reproducibility
    svm <- svm(x = traindata_features, y = traindata_labels, kernel = "linear", gamma = tunemodel$best.parameters$gamma, cost = tunemodel$best.parameters$cost)  # Train SVM model
    pred_svm <- predict(svm, testdata_features)  # Predict using SVM
    acc_relf1_svm[n] <- sum(pred_svm == testdata_labels) / length(testdata_labels)  # Compute SVM accuracy

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
    acc_relf1_knn[n] <- sum(pred_knn == testdata_labels) / length(testdata_labels)  # Compute KNN accuracy

    # Naive Bayes (NB)
    tryall <- tryall %>% mutate_all(~ if (is.numeric(.)) as.factor(.) else .)  # Convert numeric columns to factors for Naive Bayes
    traindata <- merge(tryall, trainings, by = "IID")  # Merge training data
    testdata <- merge(tryall, testings, by = "IID")    # Merge testing data
    traindata_features <- traindata[, !(names(traindata) %in% excludevars)]  # Extract features for training
    traindata_labels <- traindata[["group"]]           # Extract labels for training
    testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing
    testdata_labels <- testdata[["group"]]             # Extract labels for testing
    set.seed(77)  # Set seed for reproducibility
    model <- train(x = traindata_features, y = traindata_labels, method = "naive_bayes", trControl = ctrl, tuneGrid = grid)  # Train Naive Bayes model
    set.seed(77)  # Set seed for reproducibility
    nb <- e1071::naiveBayes(x = traindata_features, y = traindata_labels, laplace = model$bestTune$laplace)  # Train Naive Bayes model with best parameters
    pred_nb <- predict(nb, testdata_features)  # Naive Bayes prediction
    acc_relf1_nb[n] <- sum(pred_nb == testdata_labels) / length(testdata_labels)  # Compute Naive Bayes accuracy

    # Save the selected SNPs for the current iteration
    relf_1list[[paste0("snp_relf_", n)]] <- top_features
}

## ===============================================================
## step4: Save the accuracy results for all methods in a data frame
## ===============================================================
acc_relf1_10 <- data.frame(snp_num, acc_relf1_knn, acc_relf1_rf, acc_relf1_svm, acc_relf1_nb)  # Store accuracy for all methods
save(acc_relf1_10, file = "acc_relf1_10.RData")  # Classification accuracy files containing the performance of four machine learning classifiers based on SNP lists of varying sizes
save(relf_1list, file = "snp_relf1_list.RData")  # SNP lists selected by the Relief method, which contains SNP lists with varying sizes
