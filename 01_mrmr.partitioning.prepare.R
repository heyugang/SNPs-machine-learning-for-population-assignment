###########################################################################
# Divide the data into training and testing sets

# Data storage path and name
library(dplyr)  # Load the dplyr library for data manipulation
library(caret)  # Load the caret library for model training and cross-validation
library(class)  # Load the class library for KNN classification
library(e1071)  # Load the e1071 library for Naive Bayes and SVM
library(mRMRe)  # Load the mRMRe library for mutual information-based feature selection
library(CORElearn)  # (Commented) Load the CORElearn library for machine learning models

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
