#!/bin/bash

# Color codes for clean output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Intel ARC GPU - Linux Performance Optimizer       ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# 1. System Driver Installation
echo -e "\n${YELLOW}[1/4] Installing System Drivers & Compute Runtimes...${NC}"
sudo pacman -Sy --needed \
    intel-compute-runtime \
    level-zero-loader \
    level-zero-headers \
    intel-media-driver \
    vulkan-intel \
    libvpl-intel-gpu \
    clinfo \
    intel-gpu-tools --noconfirm

# 2. Python Environment Setup
echo -e "\n${YELLOW}[2/4] Configuring Python Environment...${NC}"
ENV_NAME="ArtTic-LAB"

# Find Conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
else
    echo -e "${RED}Error: Conda not found. Please install Miniconda/Miniforge first.${NC}"
    exit 1
fi

echo -e "${GREEN}Activating $ENV_NAME...${NC}"

# 3. Performance Tuning
echo -e "\n${YELLOW}[4/4] Applying Performance Environment Variables...${NC}"

# We will create a local 'env_vars' file that start.sh can source
cat << EOF > intel_gpu_vars.sh
# Intel GPU Performance Tweaks
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_FORCE_ATTENTION_TYPE=SDP
export TORCH_LLM_ALLREDUCE=1
EOF

echo -e "${GREEN}Environment variables generated in 'intel_gpu_vars.sh'${NC}"

# Verification
echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${GREEN}Verification: Testing GPU Detection...${NC}"
python -c "import torch; import intel_extension_for_pytorch as ipex; print(f'Detected GPU: {torch.xpu.get_device_name(0)}'); print(f'Compute Link: {torch.xpu.is_available()}')"

echo -e "\n${YELLOW}To finish the setup, I will now update your start.sh...${NC}"

# Inject the variables into start.sh if not already there
if ! grep -q "intel_gpu_vars.sh" start.sh; then
    sed -i '2i source ./intel_gpu_vars.sh' start.sh
    echo -e "${GREEN}Updated start.sh to auto-load Intel optimizations.${NC}"
fi

echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "You can now run ${YELLOW}./start.sh${NC} for maximum performance."
echo -e "${BLUE}=======================================================${NC}"