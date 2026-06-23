# Alpha-Synuclein ML Agent: Multi-Environment Deployment Guide

This guide explains how to configure, set up, and run the autonomous ML research agent in different environments: **Local Laptop**, **Google Colab**, **Kaggle Notebooks**, and **Google Cloud Platform (GCP) VMs**. 

---

## ── Architecture Overview ──

The agent features built-in infrastructure to automatically adapt to its runtime environment:
- **`PlatformDetector`**: Inspects the platform, GPU name, available VRAM, and local Ollama availability, recommending the optimal model configuration.
- **`LlamafileManager`**: Automates GGUF model downloading from Hugging Face and serves it locally in environments (Colab/Kaggle) where Ollama cannot be installed.
- **Two-Brain Mode**: Splits workloads, delegating code execution to local GPU models (no rate limits/costs) and reasoning/orchestration to cloud APIs (Gemini/Groq).
- **Rate Limit & Token Budget Managers**: Implements exponential backoff (errors 429, 503) and inter-call delays (RPM caps) to run reliably under free-tier limits.

---

## 1. Local Laptop / Workstation (Windows, macOS, Linux)

Running locally is best for rapid debugging or if you have a local GPU (NVIDIA, Apple Silicon).

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/DhruvalPtl/alpha-synuclein-agent.git
   cd alpha-synuclein-agent
   ```
2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Configure keys**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   GROQ_API_KEY=your_groq_key_here
   MACHINE_ID=laptop
   ```

### Inference Options
* **Option A: Local Ollama (Recommended if GPU VRAM >= 8GB)**:
  1. Install [Ollama](https://ollama.com/).
  2. Run `ollama pull qwen2.5-coder:32b` (or `qwen2.5-coder:14b` / `qwen2.5-coder:7b` for lower VRAM).
  3. Start the Ollama application. The agent will auto-detect it.
* **Option B: Local LM Studio (Alternative)**:
  1. Install [LM Studio](https://lmstudio.ai/).
  2. Download any Qwen2.5-Coder model, load it, and enable the Local Server (port 1234).
  3. Set `MODEL_NAME = "lmstudio-any"` in your notebook launcher.
* **Option C: Cloud APIs (If no GPU is available)**:
  - The agent will fallback to `groq-llama` or `gemini-flash-lite` automatically.

### Running the Agent
Open Jupyter Lab and run `notebooks/run_agent.ipynb`:
```bash
jupyter lab
```

---

## 2. Google Colab (Free or Pro)

Google Colab provides a free NVIDIA T4 GPU (~15 GB VRAM). Since you cannot run system daemons like Ollama easily on Colab, the agent uses **`LlamafileManager`** to download and serve local LLMs for free inference.

### Setup Steps in Colab
Paste the following blocks into separate cells in a Colab notebook.

#### Cell 1: Clone & Install Dependencies
```python
# 1. Clone the repository
!git clone https://github.com/DhruvalPtl/alpha-synuclein-agent.git
%cd alpha-synuclein-agent

# 2. Install requirements & CUDA-optimized PyTorch
!pip install -r requirements.txt
!pip install -e .
!pip install huggingface_hub llama-cpp-python[server]
```

#### Cell 2: Configure API Keys (Securely)
Use Colab's built-in **Secrets manager** (the key icon 🔑 in the left sidebar) to add your keys: `GEMINI_API_KEY` and `GROQ_API_KEY`.
```python
import os
from google.colab import userdata

# Load API keys from Colab Secrets safely (no hardcoded credentials)
try:
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
except Exception:
    print("Warning: GEMINI_API_KEY secret not found")

try:
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except Exception:
    print("Warning: GROQ_API_KEY secret not found")

os.environ["MACHINE_ID"] = "colab"
```

#### Cell 3: Detect Platform & Launch Local Model Server
```python
from agent.core.env_config import setup_environment
from agent.core.platform_detector import PlatformDetector

# Initialize paths and chdir
setup_environment()

# Detect hardware
pd = PlatformDetector().detect()
print(f"Platform: {pd['platform']} | GPU: {pd['gpu_name']} ({pd['gpu_vram_gb']} GB VRAM)")

# If GPU is present, launch Qwen2.5-Coder locally for unlimited coding steps
model_name = pd["recommended_model"]
if pd["gpu_vram_gb"] and pd["gpu_vram_gb"] >= 14:
    from agent.core.llamafile_manager import LlamafileManager
    print("Starting local llamafile server for Qwen2.5-Coder-14B...")
    # Downloads model from HF to ~/.cache/llamafile_models/ and runs local server on port 8080
    mgr = LlamafileManager(model_key="qwen-14b-coder", port=8080)
    base_url = mgr.start()
    model_name = "llamafile-14b"
else:
    print("No GPU or low VRAM. Falling back to Groq API.")
    model_name = "groq-llama"
```

#### Cell 4: Launch the Agent
```python
from agent.core.orchestrator import AgentOrchestrator

agent = AgentOrchestrator(
    model_name=model_name,
    verbosity="concise" # Shows a clean, step-by-step experiment UI
)

# Start research
result = agent.run(max_experiments=30)
```

---

## 3. Kaggle Notebooks / Kernels

Kaggle offers free weekly access to **2× NVIDIA T4 GPUs** (totaling ~30 GB VRAM). The agent's `PlatformDetector` will recommend **`llamafile-32b`** in this environment, offloading the Qwen2.5-Coder-32B model across both GPUs.

### Setup Steps in Kaggle

#### Cell 1: Clone and Bootstrap
```python
import os, subprocess
# Kaggle working directory is /kaggle/working
%cd /kaggle/working

# Clone repository
!git clone https://github.com/DhruvalPtl/alpha-synuclein-agent.git
%cd alpha-synuclein-agent

# Install dependencies
!pip install -q -r requirements.txt
!pip install -q -e .
!pip install -q huggingface_hub llama-cpp-python[server]
```

#### Cell 2: Import Secrets
Add `GEMINI_API_KEY` and `GROQ_API_KEY` to **Kaggle User Secrets** (Add-ons -> Secrets).
```python
import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

try:
    os.environ["GEMINI_API_KEY"] = user_secrets.get_secret("GEMINI_API_KEY")
except Exception:
    pass

try:
    os.environ["GROQ_API_KEY"] = user_secrets.get_secret("GROQ_API_KEY")
except Exception:
    pass

os.environ["MACHINE_ID"] = "kaggle"
```

#### Cell 3: Start Multi-GPU local server and Run
```python
from agent.core.env_config import setup_environment
from agent.core.platform_detector import PlatformDetector
from agent.core.orchestrator import AgentOrchestrator

setup_environment()
pd = PlatformDetector().detect()

model_name = pd["recommended_model"]
if pd["gpu_vram_gb"] and pd["gpu_vram_gb"] >= 28:
    # 2xT4 GPUs detected - launch Qwen2.5-Coder-32B locally
    from agent.core.llamafile_manager import LlamafileManager
    print("Serving Qwen2.5-Coder-32B across multiple GPUs...")
    mgr = LlamafileManager(model_key="qwen-32b-coder", port=8080)
    mgr.start()
    model_name = "llamafile-32b"
elif pd["gpu_vram_gb"] and pd["gpu_vram_gb"] >= 14:
    # 1xT4 GPU detected - launch Qwen-14B locally
    from agent.core.llamafile_manager import LlamafileManager
    mgr = LlamafileManager(model_key="qwen-14b-coder", port=8080)
    mgr.start()
    model_name = "llamafile-14b"

# Start the agent
agent = AgentOrchestrator(model_name=model_name, verbosity="concise")
agent.run(max_experiments=50)
```

---

## 4. Google Cloud Platform (GCP) VM (e.g. `quantkit` Server)

Google Cloud VMs are best for running long, uninterrupted workloads. The VM can run Ollama persistently and run jobs in the background.

### Setup VM via SSH
We provide a setup script (`cloud_setup.py`) to automate package installs, PyTorch-CUDA configurations, directory creations, and `.env` setups.

1. **SSH into your cloud VM**:
   ```bash
   gcloud compute ssh quantkit --zone us-east4-b
   ```
2. **Download and run the bootstrap script**:
   ```bash
   curl -sL https://raw.githubusercontent.com/DhruvalPtl/alpha-synuclein-agent/main/cloud_setup.py | python3
   ```
3. **Configure Environment Keys**:
   ```bash
   cd ~/agent_workspace
   nano .env
   # Insert your GEMINI_API_KEY, GROQ_API_KEY, etc.
   # Verify machine tag is: MACHINE_ID=gcloud
   ```

### Keep VM Agent Running Privately in the Background
To prevent the agent from stopping when your terminal closes, execute it using terminal multiplexers (`screen` or `tmux`):

1. **Start a screen session**:
   ```bash
   screen -S agent-run
   ```
2. **Activate environment and start running**:
   ```bash
   cd ~/agent_workspace
   source .venv/bin/activate
   # Run verification script
   python verify_all.py
   # Run the agent from CLI
   python run_pipeline.py
   ```
3. **Detach from screen**:
   Press `Ctrl + A` followed by `D`.
4. **Reattach later to check logs**:
   ```bash
   screen -r agent-run
   ```

---

## 5. Two-Brain Mode Config (Ultimate Performance Setup)

If you are running on a machine with a GPU (e.g., your VM or Kaggle/Colab), the **Two-Brain mode** represents the ultimate setup:
* **Reasoning Model (e.g. `groq-llama` or `gemini-flash`)**: Handles high-level logic, plans experiments, evaluates results, and decides what to do next.
* **Coding Model (e.g. `local-qwen` or `llamafile-14b`)**: Handles writing actual Python code for the experiment pipelines, where raw token quantity is high but reasoning constraints are narrow.

Configure it in your launcher cell:
```python
agent = AgentOrchestrator(
    two_brain       = True,
    reasoning_model = "groq-llama",   # High RPM, excellent logic
    coding_model    = "local-qwen",   # Local Ollama (no token limit or cost)
    explore_ratio   = 0.60,           # Dedicate 60% of run to exploration phase
)
agent.run(max_experiments=100)
```

---

## Summary Matrix

| Environment | GPU / VRAM | Serving Layer | Recommended Model | Method of Launch |
| :--- | :--- | :--- | :--- | :--- |
| **Local Laptop (No GPU)** | None | Cloud APIs | `groq-llama` | Notebook/CLI |
| **Local Laptop (GPU)** | >=8 GB | Ollama | `local-qwen` (32B or 14B) | Notebook/CLI |
| **Google Colab (Free)** | T4 (~15 GB) | `LlamafileManager` | `llamafile-14b` (Qwen-14B) | Colab Notebook |
| **Kaggle Kernel (Dual T4)** | 2xT4 (~30 GB) | `LlamafileManager` | `llamafile-32b` (Qwen-32B) | Kaggle Notebook |
| **GCP Cloud VM** | V100/A100 | Ollama | `local-qwen` (Two-Brain Mode) | `screen` background process |
