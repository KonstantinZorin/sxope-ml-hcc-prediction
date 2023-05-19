#!/bin/bash

MODEL_VERSION=$1
DATASET_ID=$2
RUN_ID=$3
RUN_CREATED_AT=$4
PREDICTED_FOR_DOS=$5
MEMBERS_CHUNK_FULL_PATH=$6

# Copy members list
gcloud storage cp "$MEMBERS_CHUNK_FULL_PATH" /app/external_data/members_id.csv

poetry run python src/sxope_ml_hcc_prediction/inference.py \
--members_path /app/external_data/members_id.csv \
--model_version "$MODEL_VERSION" \
--artifacts_path /app/models \
--dataset_id "$DATASET_ID" \
--run_id "$RUN_ID" \
--run_created_at "$RUN_CREATED_AT" \
--predicted_for_dos "$PREDICTED_FOR_DOS" \
--historical
