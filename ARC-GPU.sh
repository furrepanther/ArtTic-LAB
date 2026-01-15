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
    sudo pacman -Sy --needed intel-compute-runtime level-zero-loader level-zero-headers intel-media-driver vulkan-intel onevpl-intel-gpu clinfo intel-gpu-tools --noconfirm
elif command -v apt &> /dev/null; then
    sudo apt update && sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero level-zero-dev
fi

ENV_NAME="ArtTic-LAB"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda not found.${NC}"
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo -e "\n${YELLOW}[2/3] Validating Native XPU Environment...${NC}"
python -c "
import torch
import sys
try:
    if not hasattr(torch, 'xpu'):
        print('${RED}PyTorch lacks XPU support components.${NC}')
        sys.exit(1)
    if not torch.xpu.is_available():
        print('${YELLOW}XPU detected in software but level-zero driver not found or visible.${NC}')
        sys.exit(1)
    print(f'${GREEN}Native XPU Detected: {torch.xpu.get_device_name(0)}${NC}')
except Exception as e:
    print(f'${RED}Validation Error: {e}${NC}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Initiating Native XPU Reinstall...${NC}"
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
fi

echo -e "\n${YELLOW}[3/3] Final Hardware Check...${NC}"
python -c "
import torch
if torch.xpu.is_available():
    props = torch.xpu.get_device_properties(0)
    print(f'${GREEN}SUCCESS: {props.name}${NC}')
    print(f'${GREEN}VRAM: {props.total_memory / 1024**3:.2f} GB${NC}')
else:
    print('${RED}FAILURE: XPU still not accessible.${NC}')
"