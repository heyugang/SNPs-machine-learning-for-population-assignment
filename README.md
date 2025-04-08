# Machine learning based on informational SNPS for population assignment

---

# **1. Introduction**

---

In this study, we aim to develop an informative SNP panel for population assignment using WGS data, by evaluating three SNP selection methods (FST, mRMR, and Relief-F) combined wi**th four** popul**ar ma**chine learning classifiers (KNN, RF, SVM, and NB) . The goal is to determine the most effective approach for population assignment, particularly in cases of low genetic differentiation among populations.

# **2. The main contents**

---

The main content includes the following three parts.

## fst_machine-learning

---

SNP subsets were selected based on FST values, followed by population classification using four different machine learning methods.

## mrmr_machine-learning

---

SNP subsets were selected using the mRMR feature selection method, followed by population classification using four different machine learning algorithms.

## relf_machine-learning

---

SNP subsets were selected using the Relief feature selection method, followed by population classification using four different machine learning algorithm

# 3.Description of the script files.

---

The core scripts for data analysis in this study consist of three main parts.

## Fst

---

- 01_fst.calculate.prepare.sh
    
    In the Linux system, Plink was used to calculate the FST of SNPs from the river-type and swamp-type buffalo datasets based on whole-genome resequencing data, which was subsequently used for SNP locus selection.
    
- 02_fst.filtrate.prepare.sh
    
    In the Linux system, the top 20,000 SNP loci based on FST ranking were selected for use as input files for subsequent division of the training and testing datasets.
    
- 03_fst.partitioning.prepare.R
    
    The dataset was divided into training and testing sets using R software, with the output files serving as input for subsequent population classification.
    
- 04_fst.machine-learning.R
    
    Population classification was performed using four different machine learning methods.
    

## mRMR

---

- 01_mrmr.partitioning.prepare.R
    
    In the R environment, SNP subsets were selected based on the mRMR method, and then divided into training and testing sets for subsequent population classification.
    
- 02_mrmr.machine-learning.R
    
    Population classification was performed using four different machine learning methods.
    

## Relief

---

- 01_relf.partitioning.prepare.R
    
    In the R environment, SNP subsets were selected based on the Relief method, and then divided into training and testing sets for subsequent population classification.
    
- 02_relf.machine-learning.R
    
    Population classification was performed using four different machine learning methods.
    

# 4. **Data availability**

---

The link to access the raw data for this study is provided below.

[https://figshare.com/account/items/28235453/edit](https://figshare.com/account/items/28235453/edit).
