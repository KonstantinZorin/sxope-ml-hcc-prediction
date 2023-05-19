#!/bin/bash

poetry run python src/sxope_ml_hcc_prediction/inference.py \
--members_path /app/external_data/"$MEMBERS_FILE_NAME" \
--model_version "$MODEL_VERSION" \
--artifacts_path /app/models
