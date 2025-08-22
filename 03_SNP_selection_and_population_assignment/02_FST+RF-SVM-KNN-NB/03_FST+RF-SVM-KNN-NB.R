#####################################################################################
## SNP datasets selected based on FST values were utilized for breed assignment using four machine learning classifiers.
## Note:The script displays the execution code for the first training set only, with the remaining four executed in a similar manner.

## ===============================================================
## step1: Load the computing environment.
## ===============================================================
setwd("/example/path/to/your/project")  # Set the working directory to the project path
# Load required libraries
library(dplyr)      # For data manipulation
library(caret)      # For machine learning model training
library(class)      # For K-Nearest Neighbors classification
library(e1071)      # For Support Vector Machines and Naive Bayes

## ===============================================================
## step2: Load data and perform preprocessing
## ===============================================================
# Define variables for model training
ctrl <- trainControl(method="cv", number=10)  # Set up 10-fold cross-validation
grid <- expand.grid(.laplace = seq(0, 1, 0.1), .usekernel = c(TRUE, FALSE), .adjust = 1)  # Define parameter grid for Naive Bayes
trash <- c("FID","PAT","MAT","SEX","PHENOTYPE")  # Columns to be removed from the dataset
excludevars <- c("IID","ID","num","group")  # Columns to be excluded from analysis
snp_num <- c(100,250,500,800,1000,2000,3000,5000,10000,15000)  # List of SNP counts to be used for feature selection

# Read genotype data
data <- read.table(file='Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k.raw', header=TRUE, sep='')  # Load genotype data
group <- read.table(file="buffalo_3col.list", header=FALSE, sep="")  # Load group information
names(group) <- c("IID","ID","group")  # Assign column names to group data
group$num <- 1:nrow(group)  # Add a numeric index to each row in group data
load("trainset_list.RData")  # Load predefined training set indices

# Initialize lists to store training and testing datasets
trainingsets <- list()  # List to store training datasets
testingsets <- list()   # List to store testing datasets

for (i in 1:5) {
    # Extract training set based on predefined indices
    train <- number_list[[i]]  # Get training indices for the i-th fold
    trainingset <- group[group$num %in% train, ]  # Subset group data for training
    trainingset <- trainingset[order(trainingset$num, decreasing = FALSE), ]  # Sort training set by numeric index
    # Extract testing set by excluding training indices
    testingset <- subset(group, !(num %in% trainingset$num))  # Subset group data for testing
    # Store training and testing sets in respective lists
    trainingsets[[i]] <- trainingset  # Store the i-th training set
    testingsets[[i]] <- testingset    # Store the i-th testing set
}

# Preprocess data
group$group <- as.factor(group$group)  # Convert group labels to factors
data <- data[, !(names(data) %in% trash)]  # Remove unwanted columns from data
names <- colnames(data)  # Get column names of data
names1 <- gsub(pattern = "X(\\d+)\\.(\\d+)_.*", replacement = "\\1:\\2", names)  # Rename columns to 'chromosome:position' format
colnames(data) <- names1  # Update column names in data

# Read FST values
fst <- read.table(file="Buffalo_187_HC_pass_simplify_num_qc_indep_fix_out.fst", header=TRUE, sep="")  # Load FST values
fst1 <- fst[apply(fst, 1, function(row) any(row %in% names1)), ]  # Filter FST entries that match SNPs in data

# Sort FST values in descending order
fst2 <- fst1[order(fst1$FST, decreasing = TRUE), ]  # Sort FST data by FST value

# Extract top SNPs based on predefined SNP counts
extracted_data_list <- list()  # Initialize list to store extracted SNP data
for (len in snp_num){
  extracted_data <- fst2[1:len, ]  # Select top 'len' SNPs with highest FST values
  extracted_data_list[[length(extracted_data_list) + 1]] <- extracted_data  # Append extracted data to the list
}

## ===============================================================
## step3 Perform population classification using 4 machine learning methods
## ===============================================================
trainings <- trainingsets[[1]]  # Extract the first training set
testings <- testingsets[[1]]    # Extract the first testing set
trainings$group <- as.factor(trainings$group)  # Convert 'group' variable to a factor for classification
testings$group <- as.factor(testings$group)    # Convert 'group' variable to a factor for testing

# Initialize vectors to store accuracy for each method
acc_fst1_svm <- numeric(10)    # Store accuracy for SVM method
acc_fst1_knn <- numeric(10)    # Store accuracy for KNN method
acc_fst1_nb <- numeric(10)     # Store accuracy for Naive Bayes method
acc_fst1_rf <- numeric(10)     # Store accuracy for Random Forest method
fst_1list <- list()            # List to store SNPs for each iteration

# Loop over 10 iterations to compute accuracy using different methods
for (n in 1:10) {
    snp <- as.data.frame(extracted_data_list[[n]])  # Extract SNP data for the current iteration
    list_snp <- snp[, 2]                           # Extract the second column of SNP data
    list_snp <- append("IID", list_snp)            # Append "IID" to the SNP list
    tryall <- data[, colnames(data) %in% list_snp]  # Subset data based on the SNP list
    traindata <- merge(tryall, trainings, by = "IID")  # Merge training data with "tryall" by "IID"
    testdata <- merge(tryall, testings, by = "IID")    # Merge testing data with "tryall" by "IID"
    traindata_features <- traindata[, !(names(traindata) %in% excludevars)]  # Extract features for training set
    traindata_labels <- traindata[["group"]]         # Extract labels for training set
    testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing set
    testdata_labels <- testdata[["group"]]           # Extract labels for testing set

    # SVM (Support Vector Machine)
    set.seed(77)  # Set random seed for reproducibility
    tunemodel <- tune.svm(x = traindata_features, y = traindata_labels, data = traindata, kernel = "linear", gamma = 10^(-5:5), cost = 2^(1:9))  # Tune the SVM model
    set.seed(77)  # Set random seed for reproducibility
    svm <- svm(x = traindata_features, y = traindata_labels, kernel = "linear", gamma = tunemodel$best.parameters$gamma, cost = tunemodel$best.parameters$cost)  # Train SVM model
    pred_svm <- predict(svm, testdata_features)    # Predict using the trained SVM model
    acc_fst1_svm[n] <- sum(pred_svm == testdata_labels) / length(testdata_labels)  # Compute accuracy for SVM

    # KNN (K-Nearest Neighbors)
    folds <- createFolds(traindata_labels, k = 15, list = TRUE, returnTrain = TRUE)  # Create 15-fold cross-validation
    best_k <- 1  # Initialize best_k with 1
    best_accuracy <- 0  # Initialize best_accuracy with 0
    for (k in 1:15) {  # Loop over values of k (1 to 15)
        fold_accuracy <- 0  # Initialize fold_accuracy with 0
        for (i in 1:15) {   # Loop over each fold
            train_indices <- folds[[i]]  # Get indices for the training set
            train_data <- traindata_features[train_indices, ]  # Extract training data for current fold
            train_labels <- traindata_labels[train_indices]  # Extract training labels for current fold
            test_data <- traindata_features[-train_indices, ]  # Extract testing data for current fold
            test_labels <- traindata_labels[-train_indices]  # Extract testing labels for current fold
            predictions <- class::knn(train = train_data, test = test_data, cl = train_labels, k = k)  # Run KNN prediction
            fold_accuracy <- fold_accuracy + sum(predictions == test_labels) / length(test_labels)  # Compute accuracy for current fold
        }
        fold_accuracy <- fold_accuracy / 10  # Average the accuracy across folds
        if (fold_accuracy > best_accuracy) {  # Update best_k if the accuracy improves
            best_k <- k
            best_accuracy <- fold_accuracy
        }
    }
    k_value <- best_k  # Set the best k value based on cross-validation
    set.seed(77)  # Set random seed for reproducibility
    pred_knn <- class::knn(traindata_features, testdata_features, traindata_labels, k = k_value)  # Predict using KNN
    acc_fst1_knn[n] <- sum(pred_knn == testdata_labels) / length(testdata_labels)  # Compute accuracy for KNN

    # Naive Bayes (NB)
    tryall <- tryall %>% mutate_all(~ if (is.numeric(.)) as.factor(.) else .)  # Convert numeric variables to factors for Naive Bayes
    traindata <- merge(tryall, trainings, by = "IID")  # Merge training data with "tryall"
    testdata <- merge(tryall, testings, by = "IID")    # Merge testing data with "tryall"
    traindata_features <- traindata[, !(names(traindata) %in% excludevars)]  # Extract features for training set
    traindata_labels <- traindata[["group"]]           # Extract labels for training set
    testdata_features <- testdata[, !(names(testdata) %in% excludevars)]  # Extract features for testing set
    testdata_labels <- testdata[["group"]]             # Extract labels for testing set
    set.seed(77)  # Set random seed for reproducibility
    model <- train(x = traindata_features, y = traindata_labels, method = "naive_bayes", trControl = ctrl, tuneGrid = grid)  # Train Naive Bayes model
    set.seed(77)  # Set random seed for reproducibility
    nb <- e1071::naiveBayes(x = traindata_features, y = traindata_labels, laplace = model$bestTune$laplace)  # Train Naive Bayes model with best parameters
    pred_nb <- predict(nb, testdata_features)  # Predict using Naive Bayes
    acc_fst1_nb[n] <- sum(pred_nb == testdata_labels) / length(testdata_labels)  # Compute accuracy for Naive Bayes

    # Save the SNP output for the current iteration
    fst_1list[[paste0("snp_fst_", n)]] <- list_snp
}

## ===============================================================
## step4: Save the accuracy results for all methods in a data frame
## ===============================================================
acc_fst1_10 <- data.frame(snp_num, acc_fst1_knn, acc_fst1_rf, acc_fst1_svm, acc_fst1_nb)  # Store accuracy for all methods
save(acc_fst1_10, file = "acc_fst1_10.RData")  # Classification accuracy files containing the performance of four machine learning classifiers based on SNP lists of varying sizes
save(fst_1list, file = "snp_fst1_list.RData")  # SNP lists selected by the FST method, which contains SNP lists with varying sizes