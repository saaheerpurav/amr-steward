FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install training stack (Unsloth nightly for Qwen3 support)
# Pin torchvision to a version compatible with whatever torch unsloth installs
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "torchvision>=0.25.0" \
    "trl>=0.19" "accelerate>=0.34" "peft>=0.11" "datasets>=2.19" \
    "transformers>=4.51" "huggingface_hub>=0.23" \
    "openenv-core>=0.2.3" fastapi uvicorn 2>/dev/null || \
    pip install --no-cache-dir unsloth "torchvision>=0.19.0" "trl>=0.19" \
    "accelerate>=0.34" "peft>=0.11" "datasets>=2.19" "transformers>=4.51" \
    "huggingface_hub>=0.23" "openenv-core>=0.2.3" fastapi uvicorn

# Copy training entrypoint
COPY train_space.py /app/train_space.py

EXPOSE 7860

CMD ["python", "train_space.py"]
