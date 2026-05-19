#!/bin/bash
# Lisper Environment Setup for Mac/GPU
# Run this to set up conda environment and install dependencies

set -e

echo "=================================="
echo "Lisper Environment Setup"
echo "=================================="

# Create environment
echo "[1/4] Creating conda environment..."
conda create -n lisper python=3.11 -y
conda activate lisper

# Install PyTorch
echo "[2/4] Installing PyTorch..."
pip install torch

# Install Unsloth and deps
echo "[3/4] Installing Unsloth and dependencies..."
pip install unsloth
pip install transformers accelerate trl

# Install audio processing
echo "[4/4] Installing audio libraries..."
pip install scipy soundfile

echo ""
echo "=================================="
echo "✓ Environment ready!"
echo "=================================="
echo ""
echo "To test Gemma 4 E2B:"
echo "  conda activate lisper"
echo "  python src/model/test_base.py"
echo ""
echo "To fine-tune:"
echo "  conda activate lisper"
echo "  curl -L -o data/raw/LibriSpeech/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz"
echo "  tar -xzf data/raw/LibriSpeech/train-clean-100.tar.gz -C data/raw/LibriSpeech"
echo "  python src/model/dataset.py --build-multimodal --profile hackathon"
echo "  python src/model/finetune.py --prepare"
echo "  open notebooks/kaggle_gemma4_audio_unsloth.ipynb"
