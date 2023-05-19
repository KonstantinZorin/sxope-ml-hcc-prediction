FROM python:3.9.16-buster

ARG env_type
RUN echo "environment: $env_type"

ENV ENV_FILE=".env.$env_type"
ENV PATH="/root/.local/bin:$PATH"

# Install some basic utilities
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
RUN mkdir /app/secrets
RUN mkdir /app/src
RUN mkdir /app/external_data
RUN mkdir /app/models
RUN mkdir /app/config
WORKDIR /app

# Copy APP
COPY /src /app/src
COPY /models/$env_type /app/models
COPY /config /app/config
#COPY /secrets/credentials.json /app/secrets/credentials.json
COPY poetry.lock pyproject.toml README.md /app/
COPY container_tools/predict.sh container_tools/predict_hist.sh container_tools/initialization.sh /app/
COPY .env.gke.$env_type /app/.env

# Add non-root user
ARG USERNAME=oracle
ARG USER_UID=1001
ARG USER_GID=$USER_UID
RUN echo "adding user ${USERNAME} with UID ${USER_UID} and add it user group ${USER_GID}."

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R ${USERNAME}:${USERNAME} /app

USER ${USERNAME}

# All users can use /home/user as their home directory
ENV HOME=/home/${USERNAME}
RUN chmod 777 -R /home/${USERNAME}

# Install google cloud sdk for current user
ENV CLOUDSDK_CORE_DISABLE_PROMPTS=1
ENV PATH $PATH:/home/${USERNAME}/google-cloud-sdk/bin
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Authentication 
#ENV GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/credentials.json
#RUN gcloud auth activate-service-account --key-file=/app/secrets/credentials.json
COPY /secrets/credentials.json $HOME/.config/gcloud/application_default_credentials.json

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version=1.3.2
RUN poetry config installer.max-workers 4
RUN poetry self add "keyrings.google-artifactregistry-auth"
RUN poetry install -vvv

# Set environment
ENV ENV_FILE="/app/.env"
RUN poetry run python src/sxope_ml_hcc_prediction/app_config.py

# Delete secrets
#RUN gcloud auth revoke sa-ml-caregaps-staging@pp-ds-ml-staging.iam.gserviceaccount.com
#RUN rm secrets/credentials.json
RUN rm $HOME/.config/gcloud/application_default_credentials.json

ENTRYPOINT ["/bin/bash"]
