#!/bin/bash

MODEL_VERSION=$1
DATASET_ID=$2
RUN_ID=$3
RUN_CREATED_AT=$4
PREDICTED_FOR_DOS=$5

poetry run python src/sxope_ml_hcc_prediction/run_initialization.py \
--model_version "$MODEL_VERSION" \
--dataset_id "$DATASET_ID" \
--run_id "$RUN_ID" \
--run_created_at "$RUN_CREATED_AT" \
--predicted_for_dos "$PREDICTED_FOR_DOS"
