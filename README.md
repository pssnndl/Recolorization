# Recolorization Implementation 
This is a project as part of our Neural Networks and Deep Learning module at the NUS. We have built upon the existing works to build a better model with less data.

## Setup
For setting up and experimenting with this repository, you can refer to [Setup File](Setup.md)

## Problem Statement
Given an image and a palette, we aim to recolorize that image with the palette that is visually harmonious and aesthetically pleasing.

## Implementation

### Model Architecture
We extended the PaletteNet architecture by incorporating `additional attention layers` into both the Encoder and Decoder components. Furthermore, we represented the `target palette as an image`, enabling support for `variable palette sizes` while ensuring that illumination adjustments are applied exclusively to the colors in the palette. 


### Encoder and Decoder 
<img src="assets/Screenshot 2024-11-20 at 8.46.11 PM.png" width="600" height="400"> <img src="assets/Screenshot 2024-11-20 at 8.46.41 PM.png" width="600" height="400">

### Training
We used accelerate to train the model on a `single A100 GPU`. We had to resize our images to `256 x 256` to ensure it doesn't go CUDA OOM since we didn't have access to more than 1 GPU. Our training time was around `3 hours`.

<img width="439" height="350" alt="Screenshot 2024-11-22 at 8 46 17 PM" src="https://github.com/user-attachments/assets/7abdb91a-adcd-4c0d-83d3-fae7562a32c2" />

<img width="1105" height="493" alt="Screenshot 2024-11-20 at 11 04 20 PM" src="https://github.com/user-attachments/assets/d3819e15-07f7-4d55-8c73-cc16d51510b6" />

## Code Introduction
Our model is in `src/custom_model`. The starting point of our code is [train_gpu.sh](src/custom_model/train_gpu.sh). It starts the training using Huggingface accelerator. The config for that is in [common_utils/configs/accelerate_single_gpu_config.yaml](src/common_utils/configs/accelerate_single_gpu_config.yaml). It will start with [run_recolor_training.py](src/custom_model/run_recolor_training.py) which with initialize the `trainer` and call the `train` function. The dataset class is defined in [data.py](src/custom_model/data.py). The `trainer` is defined in [train_recolor.py](src/custom_model/train_recolor.py). Our model is defined in [model.py](src/custom_model/model.py) which includes initializes the encoder and decoder objects from [encoder_v3.py](src/custom_model/encoder_v3.py) and [decoder.py](src/custom_model/decoder.py) respectively. 
Checkpoints will be saved in `src/custom_model/recolor_model_ckpts`.

Once training is done, we can test the model from `src_infer/custom_model`. Instructions are there in [Setup File](Setup.md).

Our Streamlit application is in `deployments/streamlit_app`. 

<img width="793" height="941" alt="Screenshot 2024-11-22 at 10 48 14 PM" src="https://github.com/user-attachments/assets/afed6eed-a0d8-46fc-bf9b-da46b255e289" />

<img width="721" height="1161" alt="Comparisons_drawio_2" src="https://github.com/user-attachments/assets/25fb8a89-ff49-4893-98c9-0d7e2674cd6a" />

## Limitations
Since we added Attention layers, we have to resize our images to a smaller dimension for inference to run on CPU.

## Applications
Possible Applications are as follows
1. Marketing (Ensuring that assets follow some brand colors)
2. Gaming and Animation (Recoloring game assets, characters, or environments to match specific themes or mood settings).
3. Education and Research (Helping students experiment with color theory or simulate artistic effects in visual arts and design)
