#!/bin/bash

###################################################
## Perform genotype imputation for SNP dataset using Beagle

### Define input and output files (example paths and filenames)
DIR_data=/path/to/data_directory
IN_bed=Buffalo_187_HC_pass_simplify_num_qc_indep
VCF=Buffalo_187_HC_pass_simplify_num_qc_indep
VCFdot=Buffalo_187_HC_pass_simplify_num_qc_indep.vcf
OUT=Buffalo_187_HC_pass_simplify_num_qc_indep_fix
OUT_vcf=/path/to/data_directory/Buffalo_187_HC_pass_simplify_num_qc_indep_fix.vcf.gz
OUT_bed=/path/to/data_directory/Buffalo_187_HC_pass_simplify_num_qc_indep_fix

## Move to the working directory
cd ${DIR_data}

## Step 1: Convert the LD-filtered BED file to VCF format using PLINK
plink --bfile $IN_bed --chr-set 24 --recode vcf-iid --allow-extra-chr --out $VCF
sleep 5  # Wait for file writing to complete

## Step 2: Perform genotype imputation using Beagle with 50 iterations and 50-marker window
java -Xmx120g -jar ./bin/beagle.01Mar24.d36.jar gt=$VCFdot out=$OUT iterations=50 window=50
sleep 5  # Pause to ensure output is finalized

## Step 3: Index the imputed VCF file using BCFtools for efficient access
bcftools index -f -t --threads 10 $OUT_vcf
sleep 5  # Ensure indexing completes before next step

## Step 4: Convert the imputed VCF file back to PLINK BED format
plink --vcf $OUT_vcf --chr-set 24 --set-missing-var-ids '@:#' --keep-allele-order --allow-extra-chr \
 --make-bed --out $OUT_bed
