##################################################################################
## Merge the intersection of SNPs selected by 5 training sets using R

## ===============================================================
## step1: Load the computing environment.
## ===============================================================
setwd("/example/path/to/your/data")  # Set the working directory to the specified path


## ===============================================================
## step2: Load data and perform preprocessing
## ===============================================================
## Load data(SNP lists for five training sets were obtained by iteratively running the script "mRMR_machine_learning.R".)
load("snp_mrmr1_list.RData")  # Load the first SNP feature list
load("snp_mrmr2_list.RData")  # Load the second SNP feature list
load("snp_mrmr3_list.RData")  # Load the third SNP feature list
load("snp_mrmr4_list.RData")  # Load the fourth SNP feature list
load("snp_mrmr5_list.RData")  # Load the fifth SNP feature list

# Define the SNP count groups to be analyzed
snp_num <- c(100, 250, 500, 800, 1000, 2000, 3000, 5000, 10000)

# Obtain intersection of mRMR-selected SNPs across 5 training sets for SNP count = 100
mrmr12345_100 <- Reduce(intersect, list(mrmr1_list[[1]], mrmr2_list[[1]], mrmr3_list[[1]], mrmr4_list[[1]], mrmr5_list[[1]])) # Note: For other SNP sizes, replace index [[1]] with corresponding index based on snp_num

# Create a list to store overlapping mRMR SNPs for each SNP count group
mrmr_overlap <- list()
for (i in snp_num) {
  vector_name <- paste0("mrmr12345_", i)  # Construct variable name dynamically
  mrmr_overlap[[i]] <- get(vector_name)  # Retrieve variable using its name
}

# Export cross-method SNP intersections to CSV file
list_of_vectors <- list( ec2 = mrmr_overlap[[1]] )  # mRMR-selected overlapping SNPs (100 SNPs)

# Determine the maximum length among the vectors to pad NA values
max_length <- max(sapply(list_of_vectors, length))

# Pad shorter vectors with NA to align them for data frame creation
padded_vectors <- lapply(list_of_vectors, function(x) {
  c(x, rep(NA, max_length - length(x)))
})

# Create data frame from padded vectors
df <- data.frame(padded_vectors)
names(df) <- c("mRMR")  # Rename columns accordingly

# Remove the first row (optional - only if needed)
df <- df[-1,]

## ===============================================================
## step3: save result
## ===============================================================
# Save mRMR overlap list to RData file
save(mrmr_overlap, file = "mrmr_SNPoverlap.RData") ## Intersection of SNP lists selected by the mRMR method from five training sets