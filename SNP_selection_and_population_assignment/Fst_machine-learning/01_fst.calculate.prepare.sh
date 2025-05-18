#!/bin/bash
######################################################
#### Calculate FST for SNPs based on whole-genome data

## Define input files (after using beagle for imputation)
IN_bed=/path/to/example/input/Buffalo_187_HC_pass_simplify_num_qc_indep_fix # Plink format
IN_list=/path/to/example/list_files/buffalo_3col.list
IN_list_river=/path/to/example/list_files/buffalo_river.list
IN_list_swamp=/path/to/example/list_files/buffalo_swamp.list

## Define output directory for results
DIR_data=/path/to/example/data_result

## Define output file names
OUT=Buffalo_187_HC_pass_simplify_num_qc_indep_fix
OUT_river=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_river
OUT_swamp=Buffalo_187_HC_pass_simplify_num_qc_indep_fix_swamp


#### Check if the output directory exists, if not, create it
[ ! -d $DIR_data ] && mkdir -p $DIR_data

###################################################
#### Run the script
# Calculate FST for the entire population, river-type, and swamp-type separately
plink --bfile $IN_bed --chr-set 24 --keep $IN_list --fst --within $IN_list --keep-allele-order --out ${DIR_data}/${OUT}
plink --bfile $IN_bed --chr-set 24 --keep $IN_list_river --fst --within $IN_list_river --keep-allele-order --out ${DIR_data}/${OUT_river}
plink --bfile $IN_bed --chr-set 24 --keep $IN_list_swamp --fst --within $IN_list_swamp --keep-allele-order --out ${DIR_data}/${OUT_swamp}
