# CAPTAIN v.3.0 – Preview

**Conservation Area Prioritization Through Artificial INtelligence**

[![Status](https://img.shields.io/badge/status-Technical%20Preview-orange)](https://github.com/captain-project/captain3preview)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


## What is CAPTAIN?

CAPTAIN is a reinforcement learning system for optimizing conservation and restoration strategies in space and time.

### Key Features in v.3
⚙️ **Engineered for Scale:** Complete rewrite for high performance. Full GPU support and optimized sparse matrix operations allow for analysis at larger scales and finer resolutions.

🌱 **Dynamic Environment Scenarios:** Advanced support for time-evolving scenarios, incorporating climate change projections and dynamic implementation costs.  

🧩 **Modular & Customizable:** A flexible architecture allowing for highly tailored conservation policies and seamless integration of custom spatial data.

🎯 **Multi-objective Optimization:** Full support to quantify synergies and trade-offs between competing conservation and restoration targets.

### ⚠️ Notice
**This repository contains a preview of CAPTAIN v.3.**

* **Status:** This is an active beta. The core simulation engine and training loops are functional, but the API and internal architecture are subject to change.
* **Development:** Primary development occurs in a private repository. This public mirror is provided for demonstration, testing, and early-access feedback.
* **Support:** Please report bugs via GitHub Issues, but note that feature requests may be prioritized based on our internal development roadmap.


## Installation


#### 1. Install `uv` 
`uv` is a high-performance Python package manager that ensures consistent environments across Windows, macOS, and Linux.

* **macOS / Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
* **Windows (PowerShell):**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

#### 2. Set up the Environment
Once `uv` is installed, clone the repository and sync the dependencies. This will automatically create a virtual environment (`.venv`) and install the correct version of PyTorch.

```bash
# Clone and enter the repository
git clone https://github.com/captain-project/captain3preview
cd captain3preview

# Sync dependencies and create virtual environment
uv sync
```

#### 🛠️ Troubleshooting - Windows-Specific Setup


If you are on Windows, you may need to perform two quick steps to ensure uv and Python work correctly:

Enable Long Paths: Windows has a default 260-character limit for file paths. High-performance libraries like PyTorch often exceed this.

Run PowerShell as Administrator and execute:

```PowerShell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Execution Policy: If PowerShell blocks the uv command, run:

```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Examples

Working example scripts are in the [examples folder](https://github.com/captain-project/captain3preview/blob/main/examples)
- [plot\_input\_data.py](https://github.com/captain-project/captain3preview/blob/main/examples/plot_input_data.py) - Visualize spatial and time-varying input data
- [train_policy.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy.py) - Full training loop with real data
- [run_inference.py](https://github.com/captain-project/captain3preview/blob/main/examples/run_inference.py) - Load a trained model and perform optimization

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9). 
A pre-trained model is available [here](https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2). 

<img width="50%" alt="animated_map" src="https://github.com/user-attachments/assets/9ef85156-9a76-4a32-8c29-d5960481a7cb" />

## Project Structure

```
captain3preview/
├── captain/            # Main package
│   ├── agents/         # Policy network, feature extraction, rewards
│   ├── algorithms/     # Evolution strategies trainer, episode runner
│   ├── data/           # SpatialData, ExtinctionRisk classes
│   ├── environment/    # BioEnv simulation engine
│   └── utils/          # Utilities, data loading
└── examples/           # Usage examples
```

## Citation

If you use CAPTAIN v.3, please cite:

```bibtex
@software{captain_3_2026,
  title = {CAPTAIN v.3 beta: Conservation Area Prioritization Through Artificial INtelligence},
  year = {2026},
  url = {https://github.com/captain-project/}
}

@article{silvestro2022improving,
  title={Improving biodiversity protection through artificial intelligence},
  author={Silvestro, Daniele and Goria, Stefano and Sterner, Thomas and Antonelli, Alexandre},
  journal={Nature sustainability},
  volume={5},
  number={5},
  pages={415--424},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{silvestro2025using,
  title={Using artificial intelligence to optimize ecological restoration for climate and biodiversity},
  author={Silvestro, Daniele and Goria, Stefano and Rau, E-ping and Ferreira de Lima, Renato Augusto and Groom, Ben and Jacobsson, Piotr and Sterner, Thomas and Antonelli, Alexandre},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
}
```



## License

This project is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International, see [full license](https://github.com/captain-project/captain2/blob/main/CAPTAIN-License.pdf) for detail.

For commercial licensing inquiries or permission to deviate from these terms, please contact the development team.
