#!/bin/bash
###################################################
#### Select top 40,000 SNPs for river-type and swamp-type buffalo based on FST

## ==========================================
## Define input files
## ==========================================
IN_bed=/path/to/example/input/Buffalo_187_HC_pass_simplify_num_qc_indep_fix
IN_snplist=/path/to/example/list_files/buffalo_snplist.list ## The list consists of a single column representing SNP IDs(Top 40000 SNPs).
IN_list=/path/to/example/list_files/buffalo_3col.list ## The list contains three columns: the first two represent sample IDs, and the third indicates the breed.

## ==========================================
## Define output directory for results
## ==========================================
DIR_data=/path/to/example/data_result
OUT_fst=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_out
OUT=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k
[ ! -d $DIR_data ] && mkdir -p $DIR_data  ## Check if the output directory exists, if not, create it

## ==========================================
## Run the script
## ==========================================
# Step1: Filter top 40,000 FST SNPs from R or Excel, generate bed file.
plink --bfile $IN_bed --chr-set 24 --extract $IN_snplist --keep-allele-order --make-bed --out ${DIR_data}/${OUT}

# Step2: Calculate FST
plink --bfile ${DIR_data}/${OUT} --chr-set 24 --keep $IN_list --fst --within $IN_list --keep-allele-order --out ${DIR_data}/${OUT_fst}

# Step3: Generate raw files and others
plink --bfile ${DIR_data}/${OUT} --chr-set 24 --keep-allele-order --recode --out ${DIR_data}/${OUT}
plink --file ${DIR_data}/${OUT} --chr-set 24 --keep-allele-order --recodeA --out ${DIR_data}/${OUT}
