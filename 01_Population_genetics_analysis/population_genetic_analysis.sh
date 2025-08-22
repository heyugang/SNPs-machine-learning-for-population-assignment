#!/bin/bash
###################################################
#### Population Genetic Analysis Pipeline

## Define input BED file (example path)
IN_bed=/path/to/input/Buffalo_187_HC_pass_simplify_num_qc_indep

## Define input directory (example path)
DIR_in=/path/to/input_directory

## Define output directory (example path)
DIR_data=/path/to/output_directory

## Define output file name prefix
OUT=Buffalo_187_HC_pass_simplify_num_qc_indep

## Create output directory if it doesn't exist
[ ! -d $DIR_data ] && mkdir -p $DIR_data

## PCA Analysis using GCTA

## Step 1: Generate the Genomic Relationship Matrix (GRM) using autosomal chromosomes (1-24)
gcta64 --bfile $IN_bed --autosome-num 24 --make-grm --out ${DIR_data}/${OUT}  --thread-num 24

## Step 2: Perform PCA based on the GRM
gcta64 --grm ${DIR_data}/${OUT} --autosome-num 24 --pca --out ${DIR_data}/${OUT}.gcta --thread-num 24

## Admixture Analysis

## Define the number of threads to use for Admixture
cpus=6

## Ensure output directory exists (again, for safety)
[ ! -d $DIR_data ] && mkdir -p $DIR_data

## Change to the output directory
cd ${DIR_data}

## Loop through different K values (2 to 7) and run Admixture in parallel
for K in 2 3 4 5 6 7; do 
	admixture --cv ${DIR_in}/${OUT}.bed  $K -j$cpus | tee log_${OUT}.${K}.out &
done 
wait

## Construct NJ (Neighbor-Joining) Tree

## Step 1: Compute IBS (Identity-By-State) distance matrix using PLINK
plink --bfile ${IN} --keep-allele-order --chr-set 24 --distance square 1-ibs --out ${DIR_out}/${OUT}

## Step 2: Convert IBS matrix to PHYLIP format for downstream phylogenetic analysis
# Count number of individuals
n_samples=$(cat ${DIR_out}/${OUT}.mdist.id | wc -l)
echo $n_samples

# Combine sample IDs with distance matrix to generate PHYLIP formatted file
awk -v n_samples=$n_samples 'BEGIN{print n_samples } NR==FNR{a[NR]=$0; next} {print $1, a[FNR]}' \
 OFS='\t' ${DIR_out}/${OUT}.mdist ${DIR_out}/${OUT}.mdist.id > ${DIR_out}/${OUT}.phylip.dist
