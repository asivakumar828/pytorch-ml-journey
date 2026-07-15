# PyTorch ML Journey

A collection of hands-on PyTorch projects covering core deep learning concepts
including CNNs, transfer learning, segmentation, and transformers.

---

## Projects

### 1. CNN on MNIST
Trained a convolutional neural network from scratch on the MNIST handwritten 
digit dataset using PyTorch. Built and understood every component of the 
training loop including forward pass, loss computation, backpropagation, 
and weight updates using the Adam optimizer.

- Architecture: 2 convolutional layers, 2 fully connected layers, dropout
- Loss function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Epochs: 5
- Final test accuracy: 99.2%

---

### 2. Transfer Learning with ResNet18
Fine-tuned a pretrained ResNet18 backbone on CIFAR-10 across two phases to 
demonstrate the difference between feature extraction and full fine-tuning. 
ResNet18 was pretrained on ImageNet and adapted for 10-class image classification.

- Phase 1 (frozen backbone): 64.2% test accuracy
- Phase 2 (unfrozen backbone): 88.9% test accuracy
- Improvement: 24.7% by allowing all layers to update
- Dataset: 50,000 training images, 10 classes
- GPU: Google Colab T4

---

### 3. Road Segmentation with U-Net
Built a semantic segmentation pipeline using a pretrained U-Net with ResNet34 
encoder on the CamVid driving dataset. Every pixel in each frame is classified 
into one of 32 classes including road, car, pedestrian, sky, and building.

- Encoder: ResNet34 pretrained on ImageNet
- Dataset: CamVid, 367 training images, 32 classes
- Road IoU: 0.920
- Sky IoU: 0.782
- Building IoU: 0.767
- Mean IoU: 0.118 (dragged down by rare classes with insufficient training examples)
- Epochs: 5
- GPU: Google Colab T4

---

### 4. BERT Sentiment Classification
Fine-tuned BERT base uncased on the SST-2 Stanford Sentiment Treebank dataset 
for binary sentiment classification. Demonstrates fine-tuning a large pretrained 
language model on a downstream NLP task using HuggingFace Transformers.

- Model: BERT base uncased (110M parameters)
- Dataset: SST-2, 67,349 training examples, 872 validation examples
- Task: Binary sentiment classification (positive/negative)
- Validation accuracy: 92.3% after 1 epoch
- GPU: Google Colab T4

---

## Tech Stack
Python, PyTorch, torchvision, HuggingFace Transformers,
segmentation-models-pytorch, Google Colab T4 GPU
