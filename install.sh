#!/bin/bash
set -e
ENV_NAME="ArtTic-LAB"
PYTHON_VERSION="3.11"
MINT="\033[38;2;82;183;136m"
WHITE="\033[0m"
ORANGE="\033[38;2;249;115;22m"
RED="\033[38;2;239;68;68m"

log_info() { echo -e "${MINT}[INSTALLER] >${WHITE} $1"; }
log_warn() { echo -e "${ORANGE}[WARN] >${WHITE} $1"; }
log_err() { echo -e "${RED}[ERROR] >${WHITE} $1"; }

find_conda() {
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        return 0
    fi
    local common_paths=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/miniconda3" "/opt/anaconda3" "/opt/miniforge3")
    for path in "${common_paths[@]}"; do
        if [ -f "$path/bin/conda" ]; then
            eval "$($path/bin/conda shell.bash hook)"
            return 0
        fi
    done
    return 1
}

detect_gpu() {
    if lspci | grep -i "NVIDIA" &> /dev/null; then echo "2"; return; fi
    if lspci | grep -i "Intel" | grep -i "Display\|VGA\|3D" &> /dev/null; then
        if command -v clinfo &> /dev/null && clinfo | grep -i "Intel.*Arc" &> /dev/null; then echo "1"; return; fi
        echo "1" 
        return
    fi
    if lspci | grep -i "AMD" | grep -i "Display\|VGA\|3D" &> /dev/null; then echo "3"; return; fi
    echo "4"
}

get_gpu_name() {
    case $1 in
        1) echo "Intel Arc (XPU)";;
        2) echo "NVIDIA (CUDA)";;
        3) echo "AMD (ROCm)";;
        4) echo "CPU Only";;
    esac
}

clear
echo -e "${MINT}"
echo "    █████╗ ██████╗ ████████╗████████╗██╗ ██████╗"
echo "   ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██║██╔════╝"
echo "   ███████║██████╔╝   ██║      ██║   ██║██║     "
echo "   ██╔══██║██╔══██╗   ██║      ██║   ██║██║     "
echo "   ██║  ██║██║  ██║   ██║      ██║   ██║╚██████╗"
echo "   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝ ╚═════╝"
echo -e "${WHITE}"
log_info "Initializing ArtTic-LAB Installer..."

if ! find_conda; then
    log_err "Conda not found. Please install Miniconda or Miniforge."
    exit 1
fi

if conda env list | grep -E "^${ENV_NAME} " &>/dev/null; then
    log_info "Updating environment '${ENV_NAME}'..."
else
    log_info "Creating environment '${ENV_NAME}'..."
    conda create --name "${ENV_NAME}" python=${PYTHON_VERSION} -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip --quiet

DETECTED_ID=$(detect_gpu)
DETECTED_NAME=$(get_gpu_name $DETECTED_ID)

echo ""
log_info "Auto-Detected Hardware: ${MINT}${DETECTED_NAME}${WHITE}"
read -p "Is this correct? [Y/n]: " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "1) Intel Arc (XPU)"
    echo "2) NVIDIA (CUDA)"
    echo "3) AMD (ROCm)"
    echo "4) CPU Only"
    read -p "Select Hardware (1-4): " HW_CHOICE
else
    HW_CHOICE=$DETECTED_ID
fi

log_info "Removing conflicting torch versions..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

log_info "Installing PyTorch..."
case $HW_CHOICE in
    1)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0
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

log_info "Installing core dependencies..."
pip install -r requirements.txt

log_info "Setup complete! Run ./start.sh to launch."