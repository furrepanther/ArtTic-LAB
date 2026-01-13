#!/bin/bash
set -e
ENV_NAME="ArtTic-LAB"
PYTHON_VERSION="3.11"

find_conda() {
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        return 0
    fi
    declare -a conda_paths=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/miniconda3" "/opt/anaconda3" "/opt/miniforge3")
    for path in "${conda_paths[@]}"; do
        if [ -f "$path/bin/conda" ]; then
            eval "$($path/bin/conda shell.bash hook)"
            return 0
        fi
    done
    return 1
}

clear
echo "======================================================="
echo "            ArtTic-LAB Installer (Multi-GPU)"
echo "======================================================="

if ! find_conda; then
    echo "[ERROR] Conda not found."
    exit 1
fi

if conda env list | grep -E "^${ENV_NAME} " &>/dev/null; then
    echo "[INFO] Updating existing environment '${ENV_NAME}'..."
else
    echo "[INFO] Creating Conda environment..."
    conda create --name "${ENV_NAME}" python=${PYTHON_VERSION} -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip --quiet

echo ""
echo "-------------------------------------------------------"
echo "Select your Hardware Accelerator:"
echo "-------------------------------------------------------"
echo "1) Intel ARC (XPU) - Uses PyTorch Nightly (Linux)"
echo "2) NVIDIA (CUDA)   - Uses Stable PyTorch"
echo "3) AMD (ROCm)      - Uses Stable PyTorch"
echo "4) CPU Only        - Slow"
read -p "Enter selection (1-4): " hw_choice

echo "[INFO] Cleaning previous PyTorch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "[INFO] Installing PyTorch..."
case $hw_choice in
    1)
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
        conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0 TORCH_LLM_ALLREDUCE=1
        conda deactivate
        conda activate "${ENV_NAME}"
        ;;
    2)
        pip install torch torchvision torchaudio
        ;;
    3)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
        ;;
    *)
        pip install torch torchvision torchaudio
        ;;
esac

echo "[INFO] Installing Dependencies..."
# Added 'uvicorn[standard]' to fix WebSocket issue
pip install diffusers transformers accelerate safetensors fastapi "uvicorn[standard]" jinja2 toml pyngrok pillow numpy sdnq

echo "======================================================="
echo "[SUCCESS] Installation complete!"
echo "Run ./start.sh to launch."
echo "======================================================="