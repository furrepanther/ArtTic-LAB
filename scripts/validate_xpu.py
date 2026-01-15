import torch
import sys
import argparse

def validate_environment():
    try:
        # Check 1: Does the module exist?
        if not hasattr(torch, 'xpu'):
            print('FAIL: PyTorch installed, but "torch.xpu" is missing.', file=sys.stderr)
            return 1
        
        # Check 2: Is the hardware visible?
        if not torch.xpu.is_available():
            print('FAIL: torch.xpu exists, but device is not available (Driver/Kernel issue).', file=sys.stderr)
            return 1
            
        print(f'Native XPU Detected: {torch.xpu.get_device_name(0)}')
        return 0 # Success

    except Exception as e:
        print(f'Validation Error: {e}', file=sys.stderr)
        return 1

def check_hardware():
    try:
        if torch.xpu.is_available():
            props = torch.xpu.get_device_properties(0)
            print(f'SUCCESS: {props.name}')
            print(f'VRAM: {props.total_memory / 1024**3:.2f} GB')
            print(f'Driver: Native PyTorch XPU')
            return 0
        else:
            print('FAILURE: XPU still not accessible after repair.', file=sys.stderr)
            return 1
    except Exception as e:
        print(f'Error during final check: {e}', file=sys.stderr)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-hardware', action='store_true', help='Perform final hardware check')
    args = parser.parse_args()

    if args.check_hardware:
        sys.exit(check_hardware())
    else:
        sys.exit(validate_environment())
