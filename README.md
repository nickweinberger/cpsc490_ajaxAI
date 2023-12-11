# CPSC 490 Final Project: Transformer-Based Models to Automate Legal Billing Software

## Project Overview
This repository contains code samples that I used for my Yale CPSC 490 final project. The aim of the project was to automate legal billing software for my brothers' company. I initially tested three multimodal models (Donut, IDEFICS, and GPT-4) after deciding not to move forward with InstructBLIP and other multimodal models. I then proposed a solution leveraging the DiT model. 

## Folders and Contents

### 1. DiT (Document Image Transformer)
- Folder: `DiT/`
- Contents: code sample from https://github.com/microsoft/unilm/tree/master/dit/object_detection/train_net, instructions for running the DiT for document layout analysis can be found [here](https://github.com/microsoft/unilm/tree/master/dit/object_detection)

### 2. IDEFICS (Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS)
- Folder: `IDEFICS/`
- Contents: code adapted from https://huggingface.co/HuggingFaceM4/idefics-80b-instruct used for preliminary evaluation of IDEFICS for visual question answering (VQA). 

### 3. InstructBLIP
- Folder: `InstructBlip/`
- Contents: code file from https://huggingface.co/docs/transformers/main/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration used to test InstructBLIP before deciding not to move forward with the model.

### 4. SampleImages
- Folder: `SampleImages/`
- Contents: This folder contains 45 JPEG images that were generated and used to evaluate the multimodal models in the first phase of my final project.

## Getting Started
To get started with the project and explore the transformer-based models and evaluation materials, navigate to the respective folders mentioned above or use the links in this README to find more examples and documentation from the original sources.

For any inquiries or further assistance, please contact Nick Weinberger (Email: nick.weinberger@yale.edu)
