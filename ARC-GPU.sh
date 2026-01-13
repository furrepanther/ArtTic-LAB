#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Intel ARC GPU - Setup & Repair Utility            ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# 1. Driver Check
echo -e "\n${YELLOW}[1/4] Checking System Drivers...${NC}"
if sudo pacman -Sy --needed \
    intel-compute-runtime \
    level-zero-loader \
    level-zero-headers \
    intel-media-driver \
    vulkan-intel \
    onevpl-intel-gpu \
    clinfo \
    intel-gpu-tools --noconfirm; then
    echo -e "${GREEN}System drivers OK.${NC}"
else
    echo -e "${RED}Pacman warning ignored (drivers might be up to date).${NC}"
fi

# 2. Environment Configuration
echo -e "\n${YELLOW}[2/4] Configuring Conda Environment...${NC}"
ENV_NAME="ArtTic-LAB"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda not found.${NC}"
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${RED}Environment '$ENV_NAME' not found.${NC}"
    echo -e "Please run './install.sh' first."
    exit 1
fi

conda activate $ENV_NAME

# APPLY VARS DIRECTLY TO CONDA ENV (No external file needed)
echo -e "${YELLOW}Setting persistent environment variables...${NC}"
conda env config vars set ONEAPI_DEVICE_SELECTOR=level_zero:0 TORCH_LLM_ALLREDUCE=1
echo -e "${GREEN}Variables applied. They will load automatically when the environment is activated.${NC}"

# Reload environment to apply changes immediately
conda deactivate
conda activate $ENV_NAME

# 3. PyTorch Version Check & Repair
echo -e "\n${YELLOW}[3/4] Validating PyTorch Installation...${NC}"
python -c "
import sys, subprocess, platform
try:
    import torch
    ver = torch.__version__
    print(f'Current PyTorch: {ver}')
    
    needs_repair = False
    if 'cu' in ver:
        print('${RED}DETECTED NVIDIA CUDA VERSION ON INTEL HOST.${NC}')
        needs_repair = True
    elif '+cpu' in ver:
        print('${RED}DETECTED CPU-ONLY VERSION.${NC}')
        needs_repair = True
    elif not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        print('${YELLOW}XPU not detected. Reinstalling might fix this.${NC}')
        needs_repair = True
    else:
        print('${GREEN}Correct XPU version detected.${NC}')

    if needs_repair:
        print('${YELLOW}Initiating Auto-Repair...${NC}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'])
        print('${YELLOW}Installing PyTorch Nightly (XPU)...${NC}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/nightly/xpu'])
        print('${GREEN}Repair Complete!${NC}')
except Exception as e:
    print(f'${RED}Error: {e}${NC}')
"

# 4. Final Verify
echo -e "\n${YELLOW}[4/4] Final Verification...${NC}"
python -c "
import torch
try:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f'${GREEN}SUCCESS! Detected XPU: {torch.xpu.get_device_name(0)}${NC}')
        print(f'${GREEN}VRAM: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB${NC}')
    else:
        print(f'${RED}FAILURE: XPU still not detected.${NC}')
        exit(1)
except:
    exit(1)
"