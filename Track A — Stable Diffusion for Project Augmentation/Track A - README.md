# Week 7 – Track A : Stable Diffusion + LoRA Fine-Tuning

## Overview
This project fine-tunes **Stable Diffusion v1-5** using a lightweight **LoRA (Low-Rank Adaptation)** model to generate customized chess-board imagery.  
The workflow demonstrates baseline image generation, image editing (img2img / inpaint), and LoRA training for concept personalization.


## How to Reproduce

### 1. Setup environment
```bash
pip install -r requirements.txt
```

### 2. Download Notebooks
```bash
Track A - Stable Diffusion & LoRA.ipynb
Track A - LoRA Fine Tuning Weights.ipynb
```

### 3. Dataset for LoRA weight training
```
Training the LoRA weights must begin first.
Download the lora_data folder use those images & metadata.jsonl for training.
Utilizing GPU is recommended.
To bypass training, upload the lora_weights folder and its single file into

Track A - Stable Diffusion & LoRA.ipynb
```

### 4. Apply LoRA Weights
``` bash
# This line will be applied in the code
pipe.unet.load_attn_procs("lora_weights")
```



 

