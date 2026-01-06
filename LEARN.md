# What I Learned

Key takeaways from building a Visual Question Answering system for mobile UI screenshots.

## Technical Lessons

### Vision-Language Models
- **Cross-attention** is essential for fusing visual and text features
- Pre-trained backbones (ResNet, BERT) significantly reduce training time
- Frozen backbones + trainable heads is an effective transfer learning strategy

### OCR + Span Extraction
- Extracting text from images first, then finding answers in text works well for UI screenshots
- Florence-2 provides high-quality OCR with minimal setup
- Token offset mapping is crucial for accurate span prediction
- ~50% of answers can be found directly in OCR text

### Model Architecture Choices
```
Classification: Simple but limited vocabulary
Span Extraction: Flexible but depends on OCR quality
Generative: Most flexible but harder to train
```

### Training Tricks That Helped
- Gradient accumulation for larger effective batch sizes
- Learning rate warmup + cosine decay
- Freezing early layers, unfreezing later for fine-tuning
- Mixed precision training (FP16) for memory efficiency

## Mistakes Made

1. **Starting with classification** - Answer vocabulary approach doesn't scale
2. **Ignoring OCR initially** - UI screenshots are text-heavy, OCR helps a lot
3. **Training from scratch** - Pre-trained models (Pix2Struct) work better faster

## What Would I Do Differently

- Start with Pix2Struct or similar pre-trained document models
- Use larger training subsets for fine-tuning
- Implement ensemble methods combining span + generative approaches
- Add data augmentation for images


## Code Snippets Worth Remembering

**Span finding with offset mapping:**
```python
encoded = tokenizer(text, return_offsets_mapping=True)
offsets = encoded['offset_mapping'][0]
# Find which token contains character position
for idx, (start, end) in enumerate(offsets):
    if start <= char_pos < end:
        token_idx = idx
```

**Gradient accumulation:**
```python
loss = loss / GRAD_ACCUM_STEPS
loss.backward()
if (step + 1) % GRAD_ACCUM_STEPS == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**Freezing/unfreezing layers:**
```python
# Freeze
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze last N layers
for param in model.backbone.layer[-N:].parameters():
    param.requires_grad = True
```
