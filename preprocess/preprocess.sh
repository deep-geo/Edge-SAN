#!/usr/bin/env bash

dst_root="<dst_root>"
dst_size=256

# CoNIC
src_root="<path_to_CoNIC_src_root>"
dst_prefix="CoNIC"
echo "################# PROCESS CoNIC #################"
python preprocess_CoNIC.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# fluorescence
src_root="<path_to_fluorescence_src_root>"
dst_prefix="fluorescence"
echo "################# PROCESS fluorescence #################"
python preprocess_fluorescence.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# histology
src_root="<path_to_histology_src_root>"
dst_prefix="histology"
echo "################# PROCESS histology #################"
python preprocess_histology.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix

# thyroid
src_root="<path_to_thyroid_src_root>"
dst_prefix="thyroid"
echo "################# PROCESS thyroid #################"
python preprocess_thyroid.py --src_root $src_root --dst_root $dst_root --dst_size $dst_size --dst_prefix $dst_prefix
