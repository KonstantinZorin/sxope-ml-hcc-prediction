# sxope-ml-hcc-prediction

**MLFlow setup**

MLFlow UI will work only kubectl installed on/in the gcloud shared-VPC machine/container and configured with following instructions:
- Kubernetes gcloud install: https://phypartners.atlassian.net/wiki/spaces/SEM/pages/4225433642/Kubernetes+installation+and+cluster+access+configuration
- Kubernetes MLFlow install: https://phypartners.atlassian.net/wiki/spaces/SEM/pages/4225925517/Mlflow+Linux
So kubectl port forwarding must be done before app runtime

**Main app interfaces**

- Dataset download: `src/sxope_ml_hcc_prediction/dataset/dataset.py`
- Model train: `src/sxope_ml_hcc_prediction/train.py`
- Model inference: `src/sxope_ml_hcc_prediction/inference.py`

**Pre-commit hooks**

- Run `poetry run pre-commit install` in order to set up hooks that automatically perform code style checks before each commit
- Run `poetry run pre-commit run --all-files` if you want to run checks by yourself
- Run `git commit` with `--no-verify` flag if you want to skip checks (not recommended)

**Docker inference**

- Build conteiner
    ```
    docker build -t sxope-ml-hcc-prediction-staging --build-arg env_type=staging --no-cache .
    ```
    env_type - environment variable (staging/production)

- Run inference 
    ```
    sudo docker run --rm \
    --env MEMBERS_FILE_NAME="members_id.csv" \
    --env MODEL_VERSION="v0.0.2.dev" \
    -v /home/*username*/secrets:/app/secrets \
    -v /home/*username*/external_data:/app/external_data \
    sxope-ml-hcc-prediction-staging:latest
    ```

    - /home/*username*/secrets - path to credentials.json
    - /home/*username*/external_data - path to folder with memberids list