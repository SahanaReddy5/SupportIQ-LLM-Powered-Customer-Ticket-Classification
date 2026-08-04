# SupportIQ: Fine-Tuned LLM for Customer Support Ticket Classification

SupportIQ is an LLM fine-tuning project that adapts Microsoft's Phi-2 model for automated customer support ticket classification.

The project uses QLoRA (4-bit quantization + LoRA adapters) to efficiently fine-tune an LLM for intent classification across four support categories:

- Account
- Billing
- General
- Technical

## Approach

- Base Model: Microsoft Phi-2 (2.7B parameters)
- Fine-tuning: LoRA / QLoRA
- GPU: Google Colab Tesla T4
- Training Samples: 680
- Evaluation Samples: 120

## Architecture
Dataset
   ↓
Phi-2 Base Model
   ↓
QLoRA + LoRA Fine-tuning
   ↓
Evaluation Harness
   ↓
Hugging Face Deployment

## Pipeline

1. Generate and prepare instruction dataset
2. Load Phi-2 using 4-bit quantization
3. Apply LoRA adapters
4. Fine-tune using supervised fine-tuning
5. Evaluate using precision, recall, F1-score, and confusion matrix
6. Deploy adapter weights to Hugging Face Hub

## Results

Accuracy: 82%

Macro F1 Score: 0.81

Highlights:
- Billing classification achieved 0.98 F1 score
- Technical classification achieved 0.80 F1 score

## Error Analysis

The model performed well on billing and technical issues but showed confusion between general inquiries and technical requests due to overlapping software-related terminology.

Future improvements:
- Expand training data
- Add more boundary examples
- Improve intent separation

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- TRL
- PEFT
- BitsAndBytes
- LoRA
- QLoRA
- Google Colab
- Hugging Face Hub

## Model

Hugging Face:
https://huggingface.co/SahanaReddy5/phi2-support-ticket-classifier
