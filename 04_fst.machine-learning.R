########################################################################################
# Perform population classification using 4 machine learning methods

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

# Save the accuracy results for all methods in a data frame
acc_fst1_10 <- data.frame(snp_num, acc_fst1_knn, acc_fst1_rf, acc_fst1_svm, acc_fst1_nb)  # Store accuracy for all methods
save(acc_fst1_10, file = "acc_fst1_10.RData")  # Save the accuracy results
save(fst_1list, file = "snp_fst1_list.RData")  # Save the SNP list for all iterations
