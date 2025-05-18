# Machine learning based on informational SNPS for population assignment


# **1. Introduction**


In this study, we aim to develop an informative SNP panel for population assignment using WGS data, by evaluating three SNP selection methods (FST, mRMR, and Relief-F) combined with four popular machine learning classifiers (KNN, RF, SVM, and NB) . The goal is to determine the most effective approach for population assignment, particularly in cases of low genetic differentiation among populations.

# 2. Workflow


![workflow.png](workflow.png)

# **3. The main contents**


The main components include population genetic analysis, genotype imputation using Beagle, SNP selection and population assignment, as well as the development of a SNP panel for accurate population discrimination.

## **Population genetics analysis**


Principal component analysis (PCA) and pairwise FST value calculation were performed using Plink. ADMIXTURE v1.3 software was used to infer the proportion of ancestry in the tested population with K values ranging from 2 to 6. For phylogenetic analysis, the NJ tree was constructed based on IBD distance and visualized using the online iTOL tool.

## Genotype imputation using Beagle


Missing genotypes were imputed using Beagle software with the parameters “iterations=50” and “window=50”.

## **SNP selection and population assignment**


To identify the most effective SNP selection methods and the highly breed-informative markers, we employed both a genetic approach (the population differentiation index FST) and two machine learning algorithms (Minimum Redundancy Maximum Relevance (mRMR) and Relief-F). For each SNP set obtained from the three selection methods, we applied four widely used machine learning classifiers—Random Forest (RF), Support Vector Machine (SVM), K-Nearest Neighbor (KNN), and Naive Bayes (NB)—to perform population assignment.

### **Fst_machine-learning**


SNP subsets were selected based on FST values, followed by population classification using four different machine learning methods.

### **mRMR_machine-learning**


SNP subsets were selected using the mRMR feature selection method, followed by population classification using four different machine learning algorithms.

### **Relief_machine-learning**


SNP subsets were selected using the Relief feature selection method, followed by population classification using four different machine learning algorithm.

## The development of a SNP panel


We identified the SNP set with the highest classification accuracy and the fewest selected SNPs by combining the optimal SNP selection method with the best-performing machine learning classifier.

# 4.Description of the script files.


## **Population genetics analysis**


**`population_genetic_analysis.sh`**

PCA, Admixture, and Neighbor-Joining (NJ) tree analyses were conducted under the Linux  system.

## Genotype imputation using Beagle


**`Imputation_Beagle.sh`**

Missing genotypes were imputed using Beagle software.

## **SNP selection and population assignment**


### Division of training and testing sets

**`date_partitioning.prepare.R`**

The dataset was divided into training and testing sets using R software, with the output files serving as input for subsequent population classification.

### **Fst_machine-learning**

**`01_fst.calculate.prepare.sh`**

Plink was used to calculate the FST of SNPs from the river-type and swamp-type buffalo datasets based on whole-genome resequencing data, which was subsequently used for SNP locus selection.

**`02_fst.filtrate.prepare.sh`**

The top 20,000 SNP loci based on FST ranking were selected for use as input files for subsequent division of the training and testing datasets.

**`03_fst.machine-learning.R`**

Population classification was performed using four different machine learning methods.

### **mRMR_machine-learning**

**`mRMR.machine-learning.R`**

The SNP datasets selected using the mRMR method were used for breed assignment through four machine learning classifiers.

### **Relief_machine-learning**

**`Relief.machine-learning.R`**

The SNP datasets selected using the Relief method were employed for breed assignment using four machine learning classifiers.

## The development of a SNP panel


**`01_SNPpanel.prepare.R`**

The intersection of SNPs identified by the mRMR5 method across the five training sets was extracted in the R environment.

**`02_SNPpanel.development.R`**

Based on the intersected SNP dataset, breed classification was performed using the Support Vector Machine (SVM) algorithm. Classification accuracy was calculated, and a confusion matrix was generated to evaluate the classification performance of the SNP set.

# 5. **Data availability**


The VCF file derived from whole-genome sequencing for this study can be accessed via the link provided below.

[https://figshare.com/account/items/28235453/edit](https://figshare.com/account/items/28235453/edit).
