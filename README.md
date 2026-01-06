# Visual Question Answering on Mobile UI Screenshots


A comparative study of different deep learning approaches for answering questions about mobile app UI screenshots.

## Overview

This project explores Visual Question Answering (VQA) on the **RICO-ScreenQA-Short** dataset, which contains 86,000+ mobile UI screenshots with associated question-answer pairs. We implement and compare three distinct approaches:

| Approach | Model | Test Accuracy | Parameters |
|----------|-------|---------------|------------|
| Classification | ResNet50 + ViT + BERT | ~15% (Top-1) | ~13M trainable |
| Span Extraction | EfficientNetV2 + BERT + Florence-2 OCR | ~42% | ~15M trainable |
| Generative | Pix2Struct (fine-tuned) | ~37% | ~282M total |

## Dataset

**RICO-ScreenQA-Short** - A large-scale dataset for question answering on mobile UI screenshots.

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 68,980 | Training data |
| Validation | 8,618 | Hyperparameter tuning |
| Test | 8,427 | Final evaluation |
| **Total** | **86,025** | |

### Dataset Features
- `screen_id`: Unique identifier for each screen
- `question`: Natural language question about the UI
- `ground_truth`: List of acceptable answers
- `file_name`: Original RICO dataset filename
- `image`: PIL Image of the mobile screenshot

### Sample Questions
| Question | Answer | Type |
|----------|--------|------|
| "What is the default period length?" | "5" | Numeric |
| "What is the name of the application?" | "Menstrual Cycles" | Text Reading |
| "How many exercises are there?" | "12" | Counting |
| "Which exercise am I doing?" | "LUNGES" | Text Reading |
| "What is the upcoming cycle date?" | "Wednesday, February 8, 2017" | Date |

## Approaches

### 1. CNN + ViT Classification ([CNN_ViT.ipynb](CNN_ViT.ipynb))

A custom Vision-Language Model treating VQA as multi-class classification.

**Architecture:**
```
Image → ResNet50 → 7x7x2048 → Flatten → Patch Embedding → ViT (4 layers) → Visual Features
Question → BERT (Small) → Text Features
Visual + Text → Cross-Attention → Fusion MLP → Classifier (4,999 classes)
```

| Parameter | Value |
|-----------|-------|
| Image Size | 224x224 |
| ViT Embed Dim | 256 |
| ViT Heads/Layers | 8 / 4 |
| Answer Vocabulary | 4,999 |
| Batch Size | 64 |
| Epochs | 5 |

---

### 2. OCR-based Span Extraction ([Final_Span_Extraction.ipynb](Final_Span_Extraction.ipynb))

Extracts text via OCR, then predicts answer span positions. **Best performing approach.**

**Architecture:**
```
Image → Florence-2 OCR → OCR Text (cached for 86K images)
Image → EfficientNetV2-S → 1280-dim features
Question → BERT → 768-dim features
OCR Text → BERT → Token embeddings

[Image + Question] → Context Fusion → 768-dim context
[OCR Tokens + Context] → Start/End Heads → Span Prediction
```

| Parameter | Value |
|-----------|-------|
| Image Size | 384x384 |
| Max OCR/Question Length | 256 / 64 tokens |
| Batch Size | 16 |
| Epochs | 12 (3 frozen + 9 fine-tune) |
| Learning Rate | 1e-4 (heads), 1e-5 (backbone) |

**Key Insight:** ~49% of training answers found directly in OCR text.

---

### 3. Pix2Struct Fine-tuning ([Final_pix2struct.ipynb](Final_pix2struct.ipynb))

Fine-tuning Google's pre-trained document understanding model.

**Architecture:**
```
Image + Question → Pix2Struct Encoder → Decoder → Generated Answer
```

| Parameter | Value |
|-----------|-------|
| Base Model | google/pix2struct-docvqa-base |
| Max Patches | 512 |
| Max Answer Length | 20 tokens |
| Training Samples | 10,000 |
| Gradient Accumulation | 16 |

**Advantage:** Minimal preprocessing, leverages pre-training on document QA.

## Results

### Detailed Performance Comparison

| Metric | Classification | Span Extraction | Pix2Struct |
|--------|---------------|-----------------|------------|
| Test Accuracy | 15.0% | **42.4%** | 37.0% |
| Val Accuracy | 14.8% | 41.2% | 35.0% |
| Training Time | ~45 min | ~90 min | ~60 min |
| Inference Speed | Fast | Medium | Slow |
| VRAM Usage | ~4 GB | ~6 GB | ~8 GB |

### Performance by Question Type (Span Extraction)

| Question Type | Accuracy | Notes |
|---------------|----------|-------|
| Text Reading | High | Direct OCR match |
| Counting | Medium | Requires understanding |
| Yes/No | Medium | Binary classification |
| Color/Location | Low | Requires visual reasoning |

## Requirements

```
torch>=2.0
transformers>=4.30
datasets
tensorflow>=2.15
timm
einops
accelerate
sentencepiece
pillow
tqdm
matplotlib
numpy
```

## Usage

```bash
# 1. Install dependencies
pip install torch transformers datasets tensorflow timm accelerate

# 2. Download dataset (in notebook)
from datasets import load_dataset
dataset = load_dataset("/RICO-ScreenQA-Short")
dataset.save_to_disk("/content/hf_datasets/RICO-ScreenQA-Short")

# 3. Run notebooks in Google Colab (GPU required)
```

### Quick Inference Example (Pix2Struct)
```python
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")

inputs = processor(images=image, text=question, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
answer = processor.decode(outputs[0], skip_special_tokens=True)
```

## Key Findings

1. **OCR is crucial for UI VQA** - Most answers appear as text in screenshots
2. **Span extraction > Classification** - Flexible answer extraction beats fixed vocabulary
3. **Pre-trained models work well** - Pix2Struct achieves 37% with minimal fine-tuning
4. **Answer length matters** - Short factual answers are easier to predict
5. **~50% coverage** - About half of answers can be found directly in OCR text

### Challenges Encountered

- Token alignment between OCR text and ground truth answers
- High VRAM requirements for vision-language models
- Long training times on large dataset
- Handling images with poor OCR quality

## Project Structure

```
Project/
├── CNN_ViT.ipynb              # Classification approach
├── OCR_CNN_ViT.ipynb          # OCR-enhanced classification
├── Final_Span_Extraction.ipynb # Span extraction (best)
├── Final_pix2struct.ipynb     # Pix2Struct fine-tuning
├── Project Report CSE468.docx # Full report
├── README.md                  # This file
└── LEARN.md                   # Learnings & notes
```

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4070 |
| Platform | Google Colab |
| CUDA | 12.x |
| Python | 3.10+ |

## References

- [RICO Dataset](http://interactionmining.org/rico) - Mobile UI Screenshots
- [Pix2Struct Paper](https://arxiv.org/abs/2210.03347) - Screenshot Parsing
- [Florence-2](https://arxiv.org/abs/2311.06242) - Vision Foundation Model
- [ScreenQA Paper](https://arxiv.org/abs/2209.08199) - Screen Question Answering


