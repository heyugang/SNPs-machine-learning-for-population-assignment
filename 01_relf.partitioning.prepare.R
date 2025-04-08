########################################################################################
## Divide the data into training and testing sets

# Load necessary libraries
library(dplyr)  # Load dplyr for data manipulation
library(caret)  # Load caret for machine learning and cross-validation
library(class)  # Load class for KNN classifier
library(e1071)  # Load e1071 for SVM and Naive Bayes classifiers
library(CORElearn)  # Load CORElearn for machine learning models

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
