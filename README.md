# Recolorization

A deep learning system for recolorizing images using a specified color palette, built as part of the Neural Networks and Deep Learning module at NUS. The project extends the PaletteNet architecture with attention layers and includes a multi-agent LLM-powered deployment pipeline.

## Problem Statement

Given an image and a color palette, recolorize the image with the palette in a way that is visually harmonious and aesthetically pleasing.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Deployment](#deployment)
  - [Streamlit App](#streamlit-app)
  - [FastAPI + Agent System](#fastapi--agent-system)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [Applications](#applications)

## Model Architecture

We extended the PaletteNet architecture by incorporating **additional attention layers** into both the Encoder and Decoder components. We represent the **target palette as an image**, enabling support for **variable palette sizes** while ensuring illumination adjustments are applied exclusively to the palette colors.

**Forward pass**: Source image -> Encoder (ResNet + self-attention) -> multi-scale features -> Decoder (cross-attention with palette conditioning) -> Recolorized image

- **Input**: Source image (LAB, 256x256), target palette (4x24x3 tensor, 6 colors in LAB), illumination (L channel)
- **Encoder** (`encoder_v3.py`): ResNet-based feature extraction with self-attention, outputs 4 multi-scale feature maps
- **Decoder** (`decoder.py`): Progressive upsampling with cross-attention at each stage, conditioning on the palette embedding
- **Output**: 3-channel LAB image

### Encoder and Decoder

<img src="assets/Screenshot 2024-11-20 at 8.46.11 PM.png" width="600" height="400"> <img src="assets/Screenshot 2024-11-20 at 8.46.41 PM.png" width="600" height="400">

### Training Results

We used Accelerate to train on a **single A100 GPU**. Images were resized to **256x256** to fit GPU memory. Training time was approximately **3 hours**.

<img width="439" height="350" alt="Screenshot 2024-11-22 at 8 46 17 PM" src="https://github.com/user-attachments/assets/7abdb91a-adcd-4c0d-83d3-fae7562a32c2" />

<img width="1105" height="493" alt="Screenshot 2024-11-20 at 11 04 20 PM" src="https://github.com/user-attachments/assets/d3819e15-07f7-4d55-8c73-cc16d51510b6" />

### Sample Results

<img width="793" height="941" alt="Screenshot 2024-11-22 at 10 48 14 PM" src="https://github.com/user-attachments/assets/afed6eed-a0d8-46fc-bf9b-da46b255e289" />

<img width="721" height="1161" alt="Comparisons_drawio_2" src="https://github.com/user-attachments/assets/25fb8a89-ff49-4893-98c9-0d7e2674cd6a" />

## Project Structure

```
Recolorization/
├── src/                              # Training source code
│   ├── custom_model/                 # Main model implementation
│   │   ├── model.py                  # RecolorizerModel (encoder + decoder)
│   │   ├── encoder_v3.py             # Feature encoder with self-attention
│   │   ├── decoder.py                # Decoder with cross-attention
│   │   ├── attention.py              # Self and cross-attention modules
│   │   ├── data.py                   # Dataset class (RecolorizeDataset)
│   │   ├── train_recolor.py          # Trainer class
│   │   ├── run_recolor_training.py   # Training entry point
│   │   ├── train_gpu.sh              # GPU training launch script
│   │   └── requirements.txt
│   └── common_utils/                 # Shared utilities
│       ├── configs/                  # Accelerate configs (GPU/CPU)
│       └── train_utils/              # W&B logging
│
├── src_infer/                        # Standalone inference testing
│   └── custom_model/
│       ├── test_model.py             # Inference test script
│       └── benchmark_cpu_*.json      # CPU benchmarking results
│
├── deployments/
│   ├── inference/                    # FastAPI backend + agent system
│   │   ├── infer.py                  # Core inference utilities
│   │   ├── agents/
│   │   │   ├── api.py                # FastAPI endpoints (/chat, /health)
│   │   │   ├── graph.py              # LangGraph agent graph
│   │   │   ├── state.py              # Shared agent state (RecolorState)
│   │   │   ├── routing.py            # Intent-based routing logic
│   │   │   ├── session.py            # Session management
│   │   │   ├── nodes/                # Agent nodes
│   │   │   │   ├── input_analyzer.py # Deterministic intent detection
│   │   │   │   ├── image_agent.py    # Image validation/processing
│   │   │   │   ├── palette_agent.py  # LLM-based palette generation
│   │   │   │   ├── recolor_agent.py  # Model inference
│   │   │   │   ├── chat_agent.py     # Conversation handling
│   │   │   │   └── respond.py        # Response formatting
│   │   │   └── tools/                # Palette tools
│   │   │       ├── palette_formation.py
│   │   │       ├── palette_utils.py
│   │   │       ├── color_extraction.py
│   │   │       └── colormind.py
│   │   └── requirements.txt
│   └── streamlit_app/                # Interactive Streamlit UI
│       ├── streamlit_app.py
│       └── requirements_deploy.txt
│
├── datasets/                         # Training/test data (DVC-managed)
│   └── processed_palettenet_data_sample_v4/
├── assets/                           # Architecture diagrams
├── Setup.md                          # Detailed setup instructions
└── README.md
```

## Setup

### Prerequisites

- Python 3.12
- [MiniConda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- CUDA 12.x (for GPU training)
- [Ollama](https://ollama.ai/) with `llama3.1:8b` (for the agent system)

### Environment

```bash
conda create -n recolor python=3.12
conda activate recolor
```

### Model Checkpoint

Download the pretrained model checkpoint from [Google Drive](https://drive.google.com/file/d/1dLir8CG_BdsSfxCKlHDgpOPShKpWooRr/view?usp=sharing) and place it in the appropriate directory depending on your use case:

- Training/testing: `src_infer/custom_model/`
- Streamlit app: `deployments/streamlit_app/`
- FastAPI backend: `deployments/inference/checkpoint/checkpoint_epoch_90.pt`

For detailed setup instructions, refer to [Setup.md](Setup.md).

## Training

```bash
# Pull the dataset
dvc pull datasets/processed_palettenet_data_sample_v4

# Install training dependencies
cd src/custom_model
pip install -r requirements.txt

# Visualize the data (optional)
python data.py

# Launch training
./train_gpu.sh
```

**Training configuration** (via `train_gpu.sh`):
- Batch size: 8 (train), 4 (validation)
- Learning rate: 0.0002 (Adam)
- Loss: MSE (L2)
- Epochs: 1000 with checkpointing every 5 epochs
- Hardware: Single A100 GPU with FP16 mixed precision
- Experiment tracking: Weights & Biases

Checkpoints are saved to `src/custom_model/recolor_model_ckpts/`.

### Code Walkthrough

The training entry point is [`train_gpu.sh`](src/custom_model/train_gpu.sh), which launches training via HuggingFace Accelerate. The flow is:

1. [`run_recolor_training.py`](src/custom_model/run_recolor_training.py) - Initializes the trainer and starts training
2. [`train_recolor.py`](src/custom_model/train_recolor.py) - Trainer class with the training loop
3. [`data.py`](src/custom_model/data.py) - Dataset class that loads images, converts to LAB color space, and prepares palette tensors
4. [`model.py`](src/custom_model/model.py) - Main model combining encoder ([`encoder_v3.py`](src/custom_model/encoder_v3.py)) and decoder ([`decoder.py`](src/custom_model/decoder.py))

## Inference

```bash
cd src_infer/custom_model
pip install torch torchvision scikit-image
python test_model.py
```

Results are saved to `src_infer/custom_model/test_results/`.

## Deployment

### Streamlit App

An interactive web UI for uploading images, picking 6 colors, and generating recolorized results.

```bash
pip install -r deployments/streamlit_app/requirements_deploy.txt
pip install watchdog
cd deployments/streamlit_app
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Images are resized to max 350x350 (rounded to nearest 16x16) for inference.

### FastAPI + Agent System

A multi-agent system built with **LangGraph** that provides a conversational interface for recolorization. It uses **Ollama** (Llama 3.1:8b) for natural language understanding and palette generation.

#### Agent Architecture

```
User Message
    |
    v
chat_agent --> input_analyzer --> [routes to one or more agents]
                                      |            |           |
                                      v            v           v
                                image_agent  palette_agent  chat_agent
                                      |            |           |
                                      +-----+------+-----------+
                                            |
                                            v
                                       join_slots (checks if image + palette ready)
                                            |
                                     +------+------+
                                     |             |
                                     v             v
                              recolor_agent    respond
                                     |             |
                                     +------+------+
                                            |
                                            v
                                         respond --> User
```

**Agents:**
- **input_analyzer**: Deterministic intent detection (upload image, set palette, describe palette, extract colors, recolor, etc.)
- **image_agent**: Validates and stores uploaded images (format, size checks)
- **palette_agent**: Generates palettes via LLM tool-calling -- extract from images (ColorThief/Pylette), generate from text descriptions, fetch from Colormind API, parse hex/RGB input, create variations (warmer, cooler, bold, subtle, complementary, etc.)
- **recolor_agent**: Runs the recolorization model inference
- **respond**: Formats the final response with text, palette, and result image

#### Running the API

```bash
cd deployments/inference
pip install -r requirements.txt

# Ensure Ollama is running with llama3.1:8b
ollama pull llama3.1:8b

python -m uvicorn agents.api:app --reload --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000` (Swagger docs at `/docs`).

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message (with optional image) and get a response with palette suggestions or recolorized result |
| GET | `/health` | Health check |

#### Environment Variables (optional)

For LangSmith tracing, create a `.env` file:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your_key>
LANGSMITH_PROJECT="recolor-workflow"
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch 2.5.1, TorchVision |
| Training | HuggingFace Accelerate, Weights & Biases |
| Color Processing | scikit-image (LAB conversion), ColorThief, Pylette |
| Agent Framework | LangGraph, LangChain, LangSmith |
| LLM | Ollama (Llama 3.1:8b) |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data Management | DVC |
| Palette APIs | Colormind |

## Limitations

- **Inference resolution**: Attention layers increase memory usage, limiting CPU inference to ~256x256 images
- **Single GPU training**: Current setup supports only one A100 GPU
- **Palette size**: Fixed at 6 colors for the standard model (variable palette support is experimental)

## Applications

1. **Marketing** - Ensure brand assets follow specific color palettes
2. **Gaming & Animation** - Recolor game assets, characters, and environments for different themes
3. **Education & Research** - Experiment with color theory and simulate artistic effects
4. **Design Tools** - Rapid color iteration in design workflows
