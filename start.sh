#!/bin/bash
ENV_NAME="ArtTic-LAB"

# Silence SYCL warnings during compilation
export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1

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

echo "[INFO] Preparing to launch ArtTic-LAB..."

if ! find_conda; then
    echo "[ERROR] Conda not found."
    exit 1
fi

source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate '${ENV_NAME}'."
    echo "Run './install.sh' first."
    exit 1
fi

echo "[SUCCESS] Environment activated."
echo
echo "======================================================="
echo "             Launching ArtTic-LAB"
echo "======================================================="
echo

while true; do
    python app.py "$@"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 21 ]; then
        echo "[INFO] Restarting Backend..."
        sleep 1
        continue
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] App crashed with exit code $EXIT_CODE"
        read -p "Press Enter to exit..."
        exit $EXIT_CODE
    else
        break
    fi
done
exit 0