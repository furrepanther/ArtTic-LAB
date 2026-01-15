<p align="center">
  <img src="assets/Banner.png" alt="ArtTic-LAB Banner" width="100%"/>
</p>

<h2 align="center">Universal AI Art Station: Intel XPU • NVIDIA CUDA • AMD ROCm 🎨</h2>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.9%2B-orange.svg?style=for-the-badge&logo=pytorch" alt="PyTorch">
  </a>
</p>

---

**ArtTic-LAB v4.1** is a hardware-agnostic, node-based generative AI platform.

### 🚀 Key Features

- **Universal Hardware Support**:
  - **Intel Arc**: Native XPU (No IPEX required, Pure PyTorch).
  - **NVIDIA**: CUDA native execution.
  - **AMD**: ROCm support via PyTorch.
- **SDNQ (Optional)**: Toggleable quantization for lower VRAM usage.
- **Web UI**: Node-based, responsive canvas (Mobile/Desktop).
- **One-Click Sharing**: Integrated Ngrok tunneling for remote access.
- **Smart Pipeline**: Auto-detects SD1.5, SD2.1, SDXL, SD3, and FLUX models.

### 📥 Installation

1.  **Prerequisite**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2.  **Install**:

    ```bash
    chmod +x install.sh
    ./install.sh
    ```

    _The installer will auto-detect your GPU and install the correct PyTorch version._

3.  **Launch**:
    ```bash
    ./start.sh
    ```

### 🎛️ Usage

- **Load Model**: Select model -> Toggle "Quantization" if on low VRAM -> Click Load.
- **Share**: Click the Share icon in the top right to generate a public link.
