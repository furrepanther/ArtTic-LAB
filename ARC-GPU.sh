#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Intel ARC GPU - Native XPU Setup & Repair        ${NC}"
echo -e "${BLUE}=======================================================${NC}"

echo -e "\n${YELLOW}[1/3] Verifying System Runtimes...${NC}"

if command -v pacman &> /dev/null; then
    echo -e "Detected Package Manager: ${GREEN}Pacman (Arch/Manjaro)${NC}"
    sudo pacman -Sy --needed intel-compute-runtime level-zero-loader level-zero-headers \
    intel-media-driver vulkan-intel onevpl-intel-gpu clinfo intel-gpu-tools --noconfirm

elif command -v dnf &> /dev/null; then
    echo -e "Detected Package Manager: ${GREEN}DNF (Fedora/RHEL)${NC}"
    sudo dnf install -y intel-compute-runtime intel-media-driver level-zero \
    intel-gpu-tools clinfo igt-gpu-tools

elif command -v apt &> /dev/null; then
    echo -e "Detected Package Manager: ${GREEN}APT (Ubuntu/Debian)${NC}"
    sudo apt update
    sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    clinfo intel-gpu-tools

elif command -v zypper &> /dev/null; then
    echo -e "Detected Package Manager: ${GREEN}Zypper (OpenSUSE)${NC}"
    sudo zypper refresh
    sudo zypper install -y intel-opencl level-zero intel-media-driver \
    clinfo intel-gpu-tools

else
    echo -e "${RED}Error: Unsupported Package Manager. Please install Intel Compute Runtime & Level Zero manually.${NC}"
    exit 1
fi

ENV_NAME="ArtTic-LAB"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda not found. Please install Miniconda or Anaconda.${NC}"
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo -e "${YELLOW}Environment '$ENV_NAME' not found. Creating it...${NC}"
    conda create -n $ENV_NAME python=3.10 -y
fi

conda activate $ENV_NAME
echo -e "Active Environment: ${GREEN}$CONDA_DEFAULT_ENV${NC}"

echo -e "\n${YELLOW}[2/3] Validating Native XPU Environment...${NC}"


# [2/3] Validating Native XPU Environment
if ! python scripts/validate_xpu.py; then
    echo -e "${YELLOW}Initiating Native XPU Reinstall...${NC}"
    
    pip uninstall -y torch torchvision torchaudio
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
fi

echo -e "\n${YELLOW}[3/3] Final Hardware Check...${NC}"
python scripts/validate_xpu.py --check-hardware