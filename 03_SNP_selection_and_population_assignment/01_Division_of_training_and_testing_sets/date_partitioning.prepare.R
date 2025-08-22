#######################################################################
# Divide the data into training and testing sets

## ===============================================================
## A random subset of training samples 
## (IDs 1 to 187, corresponding to their respective cattle breeds) 
## ==============================================================
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

## ===============================================================
## Output the training set list
## ===============================================================
save(number_list, file="trainset_list.RData")    # The output was saved as `trainset_list.RData`, which contains sample ID lists for the five training sets.



