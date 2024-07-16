# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04@sha256:fb6968427928df5d38a88b736f47f45f640eb874277678ea91a8a34649d9792d as base

USER root

# remove the unused conda environments
RUN conda install anaconda-clean  && \
    anaconda-clean -y && \
    rm -rf /opt/miniconda

RUN apt-get update && \
    apt-get install -y graphviz-dev python3-dev python3-pip && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python -c 'import sys; assert sys.version_info[:2] == (3, 10)'

ENV POETRY_CACHE_DIR="/root/.cache/pypoetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VERSION=1.8.2
RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install poetry==$POETRY_VERSION

# Install dependencies separately to allow dependency caching
# Note: Temporarily create dummy content to allow installing the dev dependencies.
WORKDIR /workspaces/causica
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --only main --no-root --no-directory

FROM base as deploy
COPY . /workspaces/causica

FROM base as dev
# Install development shell and utils
COPY .devcontainer/.p10k.zsh /root/
RUN <<EOT
    apt-get update
    apt-get install -y zsh ruby-full moby-cli
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash
    apt-get clean -y
    rm -rf /var/lib/apt/lists/*
    git clone --depth=1 https://github.com/scmbreeze/scm_breeze.git ~/.scm_breeze
    ~/.scm_breeze/install.sh
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k
    mkdir -p ~/command_history
    cat <<'    EOF' >> ~/.zshrc
        # Set history file in mountable location
        export HISTFILE=~/command_history/.zsh_history
        export HISTFILESIZE=10000000
        export HISTSIZE=10000000
        export SAVEHIST=10000000
        export HISTTIMEFORMAT="[%F %T] "
        setopt HIST_IGNORE_ALL_DUPS
        setopt EXTENDED_HISTORY
        setopt INC_APPEND_HISTORY
        setopt APPENDHISTORY

        source ~/powerlevel10k/powerlevel10k.zsh-theme
        [[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
        [ -s "$HOME/.scm_breeze/scm_breeze.sh" ] && source "$HOME/.scm_breeze/scm_breeze.sh"

        # Set up keybindings for word navigation using ctrl + left/right
        # The original key bindings are esc + b/f
        bindkey "^[[1;5C" forward-word
        bindkey "^[[1;5D" backward-word
    EOF
EOT

RUN --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --with dev --no-root --no-directory
