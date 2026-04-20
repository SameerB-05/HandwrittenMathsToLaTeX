# Handwritten Maths to LaTeX (Based on TAMER)

This repository focuses on **Handwritten Mathematical Expression Recognition (HMER)** — converting handwritten mathematical expressions into LaTeX code using deep learning.

The project is built upon the original **TAMER** framework (AAAI 2025), with additional personal experiments, modifications, and utilities developed for learning and research purposes.

---

## Overview

Handwritten mathematical expression recognition is a challenging task involving:

- Complex 2D symbol layouts  
- Variable handwriting styles  
- Large mathematical vocabularies  
- Structural ambiguity between symbols  

This repository explores adapting and extending the TAMER model for improved experimentation on custom datasets and workflows.

---

## Base Repository

This work is built upon the following repository:

**TAMER (AAAI 2025)**  
https://github.com/qingzhenduyu/TAMER

Full credit for the original architecture, training framework, and core implementation belongs to the original authors.

---

## My Contributions

This repository includes additional work done independently for learning and experimentation:

### Dataset & Preprocessing

- InkML stroke parsing utilities  
- Stroke-to-image rasterization pipeline  
- Dataset cleaning scripts  
- Image size verification tools  
- Pickle dataset generation utilities  
- Label retokenization tools  

### Vocabulary Expansion

- Extended token dictionary generation  
- Out-of-vocabulary (OOV) checking scripts  
- Caption/token length analysis  
- Filtering unsupported labels  

### Model Adaptation

- Model surgery utilities for checkpoint modification  
- Fine-tuning workflows on custom datasets  
- Config updates for CROHME / HME100K experiments  
- Inference testing on custom images  

### Evaluation & Testing

- Prediction checking scripts  
- Single image inference pipeline  
- Custom testing utilities  

---

## Repository Structure

```text
config/                     # Training / dataset configs
eval/                       # Evaluation scripts
tamer/                      # Core TAMER implementation

Model_finetune.py          # Fine-tuning workflow
Model_surgery.py           # Modify checkpoints / vocab layers
build_extended_dict.py     # Vocabulary expansion
build_pkl.py               # Dataset pickle creation
single_inference.py        # Predict LaTeX from image
check_oov.py               # OOV token analysis
inkmlinkmlimage_v2.py      # InkML to image conversion

## Acknowledgment

This project is built upon the original TAMER repository:

https://github.com/qingzhenduyu/TAMER  
(TAMER, AAAI 2025)

The original authors deserve full credit for the base architecture and core implementation.

This repository contains my personal extensions, experiments, and modifications developed for learning and research purposes, including preprocessing utilities, vocabulary expansion, model adaptation, fine-tuning, and evaluation scripts.