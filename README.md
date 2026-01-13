<p align="center">
  <img src="assets/Banner.png" alt="ArtTic-LAB Banner" width="100%"/>
</p>

<h2 align="center">Your Portal to AI Artistry, Forged for Intel ARC & Beyond 🎨</h2>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.9%2B-orange.svg?style=for-the-badge&logo=pytorch" alt="PyTorch">
  </a>
  <a href="https://github.com/disty0/sdnq">
    <img src="https://img.shields.io/badge/Quantization-SDNQ-green.svg?style=for-the-badge" alt="SDNQ">
  </a>
  <a href="https://github.com/rootLocalGhost/ArtTic-LAB/stargazers">
    <img src="https://img.shields.io/github/stars/rootLocalGhost/ArtTic-LAB?style=for-the-badge&logo=github" alt="Stars">
  </a>
  <a href="https://github.com/rootLocalGhost/ArtTic-LAB/issues">
    <img src="https://img.shields.io/github/issues/rootLocalGhost/ArtTic-LAB?style=for-the-badge" alt="Issues">
  </a>
</p>

---

ArtTic-LAB is a **modern, clean, and powerful** AI image generation suite.

Originally meticulously crafted for the Intel® Arc™ hardware ecosystem, version 4.0 expands horizons with **Native PyTorch XPU support** and **SDNQ integration**, making it compatible with NVIDIA and AMD GPUs as well.

This isn’t just a wrapper — it’s a ground-up application focused on **performance, aesthetics, and a frictionless user experience**.

With full support for models from **Stable Diffusion 1.5 → SDXL → SD3 → FLUX**, ArtTic-LAB is the definitive creative tool. ✨

> **Note to Users:** The primary developer uses an **Intel Arc A770**. While support for NVIDIA and AMD has been implemented using standard PyTorch APIs, it has not been physically tested by the developer. Please report any issues on GitHub!

---

## 🧭 Two Ways to Create

ArtTic-LAB adapts to your preferred workflow — visual or terminal-based.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
  <div style="text-align: center; display: inline;">
    <img src="assets/ArtTic-LAB-GUI-Light.png" style="width: 330px; border-radius: 8px;">
    <img src="assets/ArtTic-LAB-GUI-Dark.png" style="width: 330px; border-radius: 8px;">
    <img src="assets/ArtTic-LAB-CLI.png" style="width: 330px; border-radius: 8px;">
  </div>
</div>

---

## 🔬 Feature Deep Dive

We’ve packed ArtTic-LAB with features designed to maximize performance and streamline your creative process.

<div align="center">

| Feature Group                  | Description                                                                                                                                                                                                                                     |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Engineered for Speed 🏎️**    | **Native XPU:** Uses PyTorch 2.9+ Native XPU support for Intel Arc, removing dependency on legacy IPEX.<br>**Mixed Precision:** All generations run in `bfloat16` for ~2× faster performance and ~50% VRAM savings with minimal quality loss.   |
| **Intelligent Pipeline 🧠**    | **SDNQ Quantization:** Integrated SD.Next Quantization engine for running larger models on smaller VRAM.<br>**Auto Model Detection:** Detects architecture (SD1.5 → SD3 → FLUX) from `.safetensors` and loads the right pipeline automatically. |
| **Total VRAM Control 💧**      | **Proactive OOM Prevention:** Smart resolution limits and one-click model unload.<br>**VAE Tiling & CPU Offloading:** Generate high-res art with minimal VRAM usage.                                                                            |
| **Streamlined for Artists ✨** | **Responsive Async UI:** No freezes while generating.<br>**Unified Image Viewer:** Smooth zoom, drag, and gallery controls.<br>**Full Parameter Control:** Prompt, CFG, LoRA, samplers — all unified in a fluid node-based interface.           |

</div>

---

## 📸 Creations Gallery

<p align="center">
  <em>Explore a glimpse of what ArtTic-LAB can create — all demo images are 512×512 px.</em>
</p>

<div align="center" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
  <img src="assets/demos/1.png" alt="Demo 1" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/2.png" alt="Demo 2" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/3.png" alt="Demo 3" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/4.png" alt="Demo 4" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/5.png" alt="Demo 5" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/6.png" alt="Demo 6" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/7.png" alt="Demo 7" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/9.png" alt="Demo 9" width="256" style="border-radius: 10px; object-fit: cover;"/>
  <img src="assets/demos/10.png" alt="Demo 10" width="256" style="border-radius: 10px; object-fit: cover;"/>
</div>

---

## 🚀 Get Started in Minutes

Launch your personal AI art studio in three simple steps.

### 1️⃣ Prerequisites

- Install **Miniconda** or **Miniforge**.
- After installation, **reopen your terminal** to ensure `conda` is available.

### 2️⃣ Installation

Download and unzip this project, then run the one-time installer:

- **Windows 🪟:** `install.bat`
- **Linux/macOS 🐧:** `chmod +x ./install.sh && ./install.sh`

_During installation, you will be prompted to select your hardware accelerator (Intel Arc, NVIDIA, AMD)._

### 3️⃣ Auto-Repair (Intel Users)

If you have an Intel Arc GPU and accidentally installed the wrong PyTorch version (or are unsure), run the repair utility:

```bash
./ARC-GPU.sh
```

This script will verify your environment, remove any conflicting CUDA/CPU PyTorch versions, and install the correct Native XPU Nightly build.

### 4️⃣ Launch & Create!

Start the server:

- **Windows:** `start.bat`
- **Linux/macOS:** `./start.sh`

Then open the provided local URL (e.g. `http://127.0.0.1:7860`) in your browser.

<details>
<summary><strong>👉 Optional Launch Arguments</strong></summary>

- `--disable-filters` → Enable full logs for debugging.
- `--share` → Create a public ngrok link.
</details>

---

## 📂 Project Structure

```bash
ArtTic-LAB/
├── 📁assets/        # Banners, demos, UI screenshots
├── 📁core/          # Core application logic
├── 📁helpers/       # CLI manager & utilities
├── 📁models/        # Drop your .safetensors models here
├── 📁outputs/       # Generated masterpieces
├── 📁pipelines/     # Core logic for SD model variants
├── 📁web/           # Custom FastAPI web UI
├── 📜app.py         # Main application launcher
├── 📜install.bat    # Windows one-click installer
├── 📜start.bat      # Windows launcher
├── 📜ARC-GPU.sh     # Intel Arc Auto-Repair & Setup Utility
└── 📜...            # Additional project files
```
