#!/usr/bin/env bash

dst_root="<dst_root>"
dst_size=256

# CoNIC
src_root="<path_to_CoNIC_src_root>"
dst_prefix="CoNIC"
echo -e "\n################# PROCESS CoNIC #################"
python preprocess_CoNIC.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# fluorescence
src_root="<path_to_fluorescence_src_root>"
dst_prefix="fluorescence"
echo -e "\n################# PROCESS fluorescence #################"
python preprocess_fluorescence.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# histology
src_root="<path_to_histology_src_root>"
dst_prefix="histology"
echo -e "\n################# PROCESS histology #################"
python preprocess_histology.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# thyroid
src_root="<path_to_thyroid_src_root>"
dst_prefix="thyroid"
echo -e "\n################# PROCESS thyroid #################"
python preprocess_thyroid.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# GlandSeg
src_root="<path_to_GlandSeg_src_root>"
dst_prefix="GlandSeg"
echo -e "\n################# PROCESS GlandSeg #################"
python preprocess_GlandSeg.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# DynamicNuclearNet
src_root="<path_to_DynamicNuclearNet_src_root>"
dst_prefix="DynamicNuclearNet"
echo -e "\n################# PROCESS DynamicNuclearNet #################"
python preprocess_DynamicNuclearNet.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# lizard
#src_root="<path_to_lizard_src_root>"
#dst_prefix="lizard"
#echo -e "\n################# PROCESS lizard #################"
#python preprocess_lizard.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# CPM15
src_root="<path_to_CPM15_src_root>"
dst_prefix="CPM15"
echo -e "\n################# PROCESS CPM15 #################"
python preprocess_CPM15.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# CPM17
src_root="<path_to_CPM17_src_root>"
dst_prefix="CPM17"
echo -e "\n################# PROCESS CPM17 #################"
python preprocess_CPM17.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# Kumar
src_root="<path_to_Kumar_src_root>"
dst_prefix="Kumar"
echo -e "\n################# PROCESS Kumar #################"
python preprocess_Kumar.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# TNBC
src_root="<path_to_TNBC_src_root>"
dst_prefix="TNBC"
echo -e "\n################# PROCESS TNBC #################"
python preprocess_TNBC.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# CoNSeP
src_root="<path_to_CoNSeP_src_root>"
dst_prefix="CoNSeP"
echo -e "\n################# PROCESS CoNSeP #################"
python preprocess_CoNSeP.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit

# zebrafish
src_root="<path_to_zebrafish_src_root>"
dst_prefix="zebrafish"
echo -e "\n################# PROCESS zebrafish #################"
python preprocess_zebrafish.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix || exit


# split dataset
echo -e "\n>>>>>>>>>>>>>>>>>>>>> Split Datasets <<<<<<<<<<<<<<<<<<<<<<"
python split_dataset.py --data_root $dst_root --ext "png" --test_size 0.05 --seed 42 || exit
