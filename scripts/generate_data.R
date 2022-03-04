# This script generates a data set.
# We provide it on a provisional basis,
# as it is likely to be replaced by a Python
# script with features improving reproducibility
# and data set sharing.
#
# Note:
# Use it as a template to generate the data sets,
# but do not push the changes to Git.
#
#
# Usage:
# Rscript scripts/generate_data.R

library("TreeMHN")
set.seed(2022)
n <- 5 # number of events
N <- 1000 # number of trees

lambda_s <- 1 # sampling event rate
gamma <- 0.1 # penalty parameter
sparsity <- 0.8 # sparsity of the MHN
exclusive_ratio <- 0.5 # proportion of inhibiting edges

# This function will generate a random MHN along with a collection of mutation trees
tree_obj <- generate_trees(n = n, N = N, lambda_s = lambda_s, sparsity = sparsity, exclusive_ratio = exclusive_ratio)
true_Theta <- tree_obj$Theta  # True MHN matrix
tree_df <- tree_obj$tree_df   # Trees
write.csv(tree_df, file = "tree_df.csv", row.names = FALSE)
write.csv(true_Theta, file="mhn.csv", row.names=FALSE)

