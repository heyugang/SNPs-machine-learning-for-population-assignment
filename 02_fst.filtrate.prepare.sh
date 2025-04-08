#!/bin/bash
###################################################
#### Select top 20,000 SNPs for river-type and swamp-type buffalo based on FST

## Define input files
IN_bed=/path/to/example/input/Buffalo_187_HC_pass_simplify_num_qc_indep_fix
IN_snplist=/path/to/example/list_files/buffalo_snplist.list
IN_list=/path/to/example/list_files/buffalo_3col.list

## Define output directory for results
DIR_data=/path/to/example/data_result

## Define output file names
OUT_fst=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_out
OUT=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_4k


#### Check if the output directory exists, if not, create it
[ ! -d $DIR_data ] && mkdir -p $DIR_data

###################################################
#### Run the script

# Step 3: Filter top 20,000 FST SNPs from R or Excel, generate bed file.
plink --bfile $IN_bed --chr-set 24 --extract $IN_snplist --keep-allele-order --make-bed --out ${DIR_data}/${OUT}

# Calculate FST
plink --bfile ${DIR_data}/${OUT} --chr-set 24 --keep $IN_list --fst --within $IN_list --keep-allele-order --out ${DIR_data}/${OUT_fst}

# Step 4: Generate raw files and others
plink --bfile ${DIR_data}/${OUT} --chr-set 24 --keep-allele-order --recode --out ${DIR_data}/${OUT}
plink --file ${DIR_data}/${OUT} --chr-set 24 --keep-allele-order --recodeA --out ${DIR_data}/${OUT}
