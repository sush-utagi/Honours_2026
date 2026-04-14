#!/bin/bash

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    MODEL_ID=$(grep '^HF_DIFFUSERS_MODEL_ID=' "$ENV_FILE" | cut -d '=' -f2 | sed "s/['\"]//g")
    echo "Loaded Model ID from .env: $MODEL_ID"
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

DATA_DIR="data_generation_backend/textual_inversion_data"
OUTPUT_BASE="data_generation_backend/embeddings"

find "$DATA_DIR" -name ".DS_Store" -delete

for class_dir in "$DATA_DIR"/*/; do
    class_name=$(basename "$class_dir")
    placeholder="<coco-$class_name>"
    output_dir="$OUTPUT_BASE/$class_name"
    
    # Check for existing results
    if [[ -f "$output_dir/learned_embeds.bin" || -f "$output_dir/learned_embeds.safetensors" ]]; then
        echo "Skipping $class_name; embedding already exists."
        continue
    fi
    
    echo "------------------------------------------------------"
    echo "Starting training for class: $class_name"
    echo "Placeholder token: $placeholder"
    echo "------------------------------------------------------"

    accelerate launch data_generation_backend/textual_inversion.py \
      --pretrained_model_name_or_path="$MODEL_ID" \
      --train_data_dir="$class_dir" \
      --learnable_property="object" \
      --placeholder_token="$placeholder" \
      --initializer_token="$class_name" \
      --num_vectors=1 \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --max_train_steps=500 \
      --repeats=50 \
      --learning_rate=5.0e-04 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --output_dir="$output_dir" \
      --validation_prompt="A photo of a $placeholder on a grey table" \
      --validation_steps=100 \
      --save_steps=100

    echo "Completed $class_name. Results in $output_dir"
done
