#!/bin/bash
ENV_NAME="ArtTic-LAB"
MINT="\033[38;2;82;183;136m"
WHITE="\033[0m"
RED="\033[38;2;239;68;68m"

log_info() { echo -e "${MINT}[LAUNCHER] >${WHITE} $1"; }
log_err() { echo -e "${RED}[ERROR] >${WHITE} $1"; }

find_conda() {
    if command -v conda &> /dev/null; then
        CONDA_BASE_PATH=$(conda info --base)
        return 0
    fi
    local common_paths=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/miniconda3" "/opt/anaconda3" "/opt/miniforge3")
    for path in "${common_paths[@]}"; do
        if [ -f "$path/bin/conda" ]; then
            CONDA_BASE_PATH="$path"
            return 0
        fi
    done
    return 1
}

if ! find_conda; then
    log_err "Conda not found."
    exit 1
fi

source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || { log_err "Failed to activate environment. Run ./install.sh first."; exit 1; }

export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
export PYTHONWARNINGS="ignore"

while true; do
    python app.py "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 21 ]; then
        log_info "Restarting Backend..."
        sleep 1
        continue
    elif [ $EXIT_CODE -ne 0 ]; then
        log_err "App crashed with code $EXIT_CODE"
        read -p "Press Enter to exit..."
        exit $EXIT_CODE
    else
        break
    fi
done