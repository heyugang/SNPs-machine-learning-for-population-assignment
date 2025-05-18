#######################################################################
# Divide the data into training and testing sets

number_list <- list()
set.seed(24415)         # Repeat 5 times, using different random seeds for sampling
for (i in 1:5) {
     number <- sample(1:26, 19)                      # Mediterranean buffalo
     number <- append(number, sample(27:46, 15))       # Nili-Ravi buffalo
     number <- append(number, sample(47:66, 15))      # Murrah buffalo
     number <- append(number, sample(67:91, 18))       # Bintangjiang buffalo
     number <- append(number, sample(92:115, 18))      # Southeastern Yunnan buffalo
     number <- append(number, sample(116:137, 16))     # Dehong buffalo
     number <- append(number, sample(138:162, 18))    # Guizhou Bai buffalo
     number <- append(number, sample(163:187, 18))    # Shanghai buffalo    
     # Store the vector generated in each iteration into the list
     number_list[[i]] <- number
 }
save(number_list, file="trainset_list.RData")    # The generated RData file is the training set file for later classification processes

# Extract and generate subsequent training set
train <- number_list[[1]]
breed_training <- group[train,]
breed_training <- breed_training[order(breed_training$num, decreasing=F),]
breed_testing <- subset(group, !(num %in% breed_training$num))

trainingsets <- list()
testingsets <- list()
for (i in 1:5) {
    # Extract training set
    train <- number_list[[i]]
    trainingset <- group[group$num %in% train, ]
    trainingset <- trainingset[order(trainingset$num, decreasing = FALSE), ]
    # Extract testing set
    testingset <- subset(group, !(num %in% trainingset$num))
    # Store the training and testing sets into lists
    trainingsets[[i]] <- trainingset
    testingsets[[i]] <- testingset
}
trainings <- trainingsets[[1]]
testings <- testingsets[[1]]


