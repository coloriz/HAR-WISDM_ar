#!/usr/bin/env bash

DATASET_DIR="dataset"

if [[ ! -d ${DATASET_DIR} ]]; then
    mkdir ${DATASET_DIR}
fi

cd ${DATASET_DIR}
curl -O http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
tar -zxvf WISDM_ar_latest.tar.gz
