#!/bin/bash

DATA_DIR="dataset_creation/textual_inversion_data"
OUTPUT_BASE="data_generation_backend/embeddings"
MODEL_ID="runwayml/stable-diffusion-v1-5"

for class_dir in "$DATA_DIR"/*/; do
    class_name=$(basename "$class_dir")
    placeholder="<coco-$class_name>"
    output_dir="$OUTPUT_BASE/$class_name"
    embed_bin="$output_dir/learned_embeds.bin"
    embed_safetensors="$output_dir/learned_embeds.safetensors"

    # Skip if an embedding for this class already exists
    if [[ -f "$embed_bin" || -f "$embed_safetensors" ]]; then
        echo "Skipping $class_name; embedding already exists in $output_dir"
        continue
    fi
    
    echo "Starting training for class: $class_name"
    echo "Placeholder token: $placeholder"

    accelerate launch data_generation_backend/textual_inversion.py \
      --pretrained_model_name_or_path="$MODEL_ID" \
      --train_data_dir="$class_dir" \
      --learnable_property="object" \
      --placeholder_token="$placeholder" \
      --initializer_token="$class_name" \
      --resolution=512 \
      --mixed_precision="fp16" \
      --train_batch_size=4 \
      --gradient_accumulation_steps=1 \
      --max_train_steps=2000 \
      --learning_rate=5.0e-04 \
      --scale_lr \
      --lr_scheduler="constant" \
      --output_dir="$output_dir"

    echo "Completed $class_name. Embedding saved to $output_dir"
done


# find dataset_creation/textual_inversion_data -name ".DS_Store" -delete