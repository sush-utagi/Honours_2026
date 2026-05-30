# Utility of Synthetic Images from Diffusion Models for Data Augmentation

UDNER CONSTRUCTION ...



  --- 

ID to put in .env file to switch between models:

# Option 1: HuggingFace repo — SD 1.5 (downloads & caches on first run)
HF_DIFFUSERS_MODEL_ID=runwayml/stable-diffusion-v1-5

# Option 2: HuggingFace repo — SDXL (downloads & caches on first run)
HF_DIFFUSERS_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0

# Option 3: Local file — SDXL from a downloaded .safetensors (no internet needed)
HF_DIFFUSERS_MODEL_ID=/path/to/sd_xl_base_1.0.safetensors


--- 

change location of huggingfcae cache:

# Hugging Face caches (in .cache)
HF_HOME=./.cache/hf
TRANSFORMERS_CACHE=./.cache/transformers
