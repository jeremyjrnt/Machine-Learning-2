# Machine Learning 2

This repository contains four comprehensive machine learning projects completed as part of the Machine Learning 2 course at Technion. Each project explores different aspects of deep learning, from fundamental neural network implementation to advanced topics like adversarial attacks and generative models.

---

## Table of Contents

1. [Project 1: Neural Networks from Scratch &amp; CNN for Dog Emotion Classification](#project-1)
2. [Project 2: Overfitting Analysis &amp; Sentiment Analysis with RNNs](#project-2)
3. [Project 3: Adversarial Attacks &amp; Contrastive Learning](#project-3)
4. [Project 4: Generative Adversarial Networks (GANs)](#project-4)

---

## Project 1: Neural Networks from Scratch & CNN for Dog Emotion Classification

### Overview

This project is divided into theoretical and practical components that cover fundamental concepts in neural networks and convolutional neural networks (CNNs).

### Part 1: Theoretical Foundations

#### 1.1 Softmax Derivative

**Objective:** Derive the gradients of the softmax function and express them solely in terms of the softmax output.

**Theory:**
The softmax function is defined as:

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

**Results:**

- When $i = k$: $\frac{\partial \text{softmax}(x)_j}{\partial x_k} = \text{softmax}(x)_j(1 - \text{softmax}(x)_j)$
- When $i \neq k$: $\frac{\partial \text{softmax}(x)_i}{\partial x_k} = -\text{softmax}(x)_i \cdot \text{softmax}(x)_j$

#### 1.2 Cross-Entropy Gradient

**Objective:** Derive the gradient of cross-entropy loss with respect to the softmax input vector.

**Theory:**
Given $\hat{y} = \text{softmax}(\theta)$ and cross-entropy: $CE(y, \hat{y}) = -\sum_i y_i \log(\hat{y}_i)$

The final result simplifies to: $\frac{\partial CE}{\partial \theta} = \hat{y} - y$

### Part 2: Fully Connected Network on MNIST

#### Objectives

- Implement a fully connected neural network from scratch without using PyTorch's automatic differentiation
- Train on MNIST dataset
- Achieve competitive accuracy

#### Implementation Details

**Architecture:**

- Input layer: 784 neurons (28×28 flattened images)
- Hidden layer: 128 neurons
- Output layer: 10 neurons (digit classes)
- Activation function: Sigmoid

**Constraints:**

- No automatic differentiation (no `backward()`)
- No built-in loss functions
- No built-in activations
- No built-in optimization algorithms
- Manual implementation of forward and backward passes

**Hyperparameters:**

- Learning rate: 0.001
- Batch size: 32
- Epochs: 16
- Seed: 42 (for reproducibility)

**Training Process:**

1. **Forward Pass:** Compute predictions using manual matrix multiplications
2. **Loss Computation:** Cross-entropy loss implementation from scratch
3. **Backward Pass:** Manual gradient computation using chain rule
4. **Weight Updates:** Gradient descent parameter updates

#### Results

- Training accuracy: ~90%
- Test accuracy: ~88%
- The model demonstrates proper convergence with manual backpropagation

#### Learning Rate Analysis

**Experiment:** Tested with 5 different learning rates: [0.0001, 0.001, 0.01, 0.1, 1.0]

**Key Insights:**

- **LR = 0.0001:** Extremely slow convergence, underfitting (accuracy ~11%)
- **LR = 0.001:** Optimal balance, steady convergence to ~60% accuracy
- **LR = 0.01:** Best performance, reaches ~90% accuracy with minimal overfitting
- **LR = 0.1:** Fast convergence but early overfitting signs (~95% train, slight gap)
- **LR = 1.0:** Severe overfitting (100% train accuracy, ~98% test)

**Conclusion:** Learning rate of 0.01 provides the best generalization.

### Part 3: CNN for Dog Emotion Classification

#### Problem Statement

Create a CNN-based classifier to identify dog emotions (angry, happy, relaxed, sad) from images. The model must:

- Achieve at least 70% test accuracy
- Minimize parameter count (deployed on smartphones)
- Not use pre-trained models

#### Dataset

- **Location:** `hw1_data/Dog_Emotion/`
- **Classes:** 4 emotions (angry, happy, relaxed, sad)
- **Splits:** Train, validation, and test sets

#### Exploratory Data Analysis (EDA)

- Analyzed class distribution across splits
- Visualized sample images from each emotion category
- Identified potential class imbalances

#### Data Preprocessing & Augmentation

**Training Augmentation (Extensive):**

- Random resized crop (64×64, scale 0.7-1.0)
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- Random horizontal/vertical flips
- Random affine transformations
- Random grayscale conversion (10%)
- Gaussian blur
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Validation/Test Augmentation (Minimal):**

- Resize to 64×64
- Normalization only

#### Model Architecture: Reduced Efficient CNN

**Design Philosophy:** Use depthwise-separable convolutions to minimize parameters while maintaining representational power.

**Architecture Details:**

```
Initial Convolution:
- Conv2d(3→16, kernel=3, stride=1, padding=1)
- BatchNorm2d(16)
- ReLU

Depthwise-Separable Blocks (6 blocks):
Block 1-2: 16→32→32 (stride=2, then 1)
Block 3-4: 32→64→64 (stride=2, then 1)
Block 5-6: 64→128→128 (stride=2, then 1)

Global Average Pooling: AdaptiveAvgPool2d(1×1)

Classifier:
- Linear(128→128)
- BatchNorm1d(128)
- ReLU
- Dropout(0.5)
- Linear(128→4)
```

**Total Parameters:** ~53,000 (extremely efficient!)

#### Depthwise-Separable Convolution

Splits standard convolution into:

1. **Depthwise:** Applies filters per channel separately
2. **Pointwise:** 1×1 convolution to combine channels
3. Significantly reduces computational cost and parameters

#### Training Details

**Hyperparameters:**

- Batch size: 16 (small for better generalization)
- Learning rate: 0.001
- Optimizer: Adam with weight decay (1e-4)
- Loss function: Label smoothing cross-entropy (smoothing=0.1)
- Scheduler: CosineAnnealingLR (T_max=1000, eta_min=1e-5)
- Epochs: 1000

**Training Strategy:**

- Low batch size + high epochs for gradual learning
- Label smoothing to prevent overconfidence
- Extensive data augmentation to improve generalization
- Cosine learning rate decay for smooth convergence

#### Results

- **Best Test Accuracy:** 75.8%
- **Training Accuracy:** ~82%
- Model successfully passes the 70% threshold
- Excellent parameter efficiency (53k parameters)

#### Key Insights

- **Parameter Reduction:** Successfully reduced from initial 400k to 53k parameters through architectural optimization
- **Data Augmentation Critical:** Extensive augmentation was essential for preventing overfitting on small dataset
- **Label Smoothing:** Helped reduce overconfidence and improved generalization
- **Training Duration:** High epoch count (1000) with small batches allowed thorough learning

**Reference:** Insights from research paper on animal emotion recognition: [Nature Scientific Reports](https://www.nature.com/articles/s41598-023-30442-0)

### Part 4: Analyzing Pre-trained CNN (VGG16)

#### Objectives

- Load and analyze a pre-trained VGG16 model
- Understand filter responses in convolutional layers
- Visualize what the network learns

#### Preprocessing Steps for VGG16

1. **Resize:** Images to 224×224 pixels
2. **Convert to Tensor:** Scale to [0, 1]
3. **Normalize:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
4. **Add Batch Dimension:** For model input

#### Experiments

**1. Bird Image Classification:**

- Loaded images from `hw1_data/birds`
- Applied VGG16 for top-5 predictions
- Analyzed class probabilities from ImageNet classes

**2. Dog Image Classification:**

- Selected dog image from `hw1_data/dogs`
- Obtained top-5 predictions with confidence scores

**3. Filter Response Visualization:**
Visualized first 3 filters in the first convolutional layer:

- **Filter 1:** Detects vertical and diagonal edges
- **Filter 2:** Captures texture and fine details
- **Filter 3:** Identifies larger-scale shapes and structures

#### Insights

- Early layers detect low-level features (edges, textures)
- These features are combined in deeper layers for complex pattern recognition
- Filter visualization reveals what the network "sees" in images

---

## Project 2: Overfitting Analysis & Sentiment Analysis with RNNs

### Overview

This project explores generalization, overfitting, and recurrent neural networks (RNNs) for natural language processing tasks.

### Part 1: Generalization and Overfitting (Random Labels)

#### Objective

Demonstrate that neural networks can overfit to random labels, achieving near-zero training loss while having no meaningful generalization.

#### Experimental Setup

**Dataset:**

- MNIST (first 128 training samples only)
- Random binary labels generated from Bernoulli(0.5)

**Constraints:**

- Batch size: 128 (entire subset in one batch)
- Shuffle: False
- Binary classification (0 or 1)

**Network Architecture:**

- Input: 784 (28×28 flattened)
- Hidden: 128 neurons with ReLU
- Output: 2 classes
- Loss: Cross-entropy

**Hyperparameters:**

- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 100

#### Results

- **Training Loss:** Converges to ~0.0
- **Training Accuracy:** ~100%
- **Test Accuracy:** ~49.58% (random chance)

#### Key Insights

This experiment demonstrates that:

1. Neural networks have sufficient capacity to memorize random labels
2. Perfect training performance doesn't guarantee generalization
3. When labels are random, test accuracy equals random chance (50% for binary)
4. **Overfitting** is the memorization of training data without learning generalizable patterns

**Conclusion:** This illustrates the importance of:

- Proper regularization techniques
- Validation set monitoring
- Not relying solely on training metrics

### Part 2: Sentiment Analysis with RNNs

#### Problem Statement

Build RNN-based models to classify tweet emotions into three categories: happiness, sadness, and neutral.

**Target:** Achieve at least 47% test accuracy

#### Dataset

**Files:**

- `trainEmotions.csv`: Training tweets with emotion labels
- `testEmotions.csv`: Test tweets

**Format:**

```
emotion, tweet_content
happiness, Welcome @doeko ! Really glad to know you here...
sadness, Disappointment really sucks! I'm getting used to it.
neutral, I just want to Sleep.
```

#### Exploratory Data Analysis (EDA)

**Data Statistics:**

- Training samples: ~15,000 tweets
- Test samples: ~3,000 tweets
- Classes: happiness, sadness, neutral

**Class Distribution:**

- Analyzed label distribution across train and test sets
- Visualized with bar plots
- Checked for class imbalance

**Data Cleaning:**

- Removed duplicate tweets
- Stripped special characters and punctuation
- Tokenization for word-level processing

#### Text Preprocessing Pipeline

**1. Data Cleaning:**

```python
- Remove duplicates
- Remove punctuation using regex
- Lowercase conversion
```

**2. Word Embeddings:**

- **Method:** Word2Vec (Google News 300-dimensional)
- **Source:** Pre-trained gensim model
- **Fallback:** Zero vector for out-of-vocabulary words

**3. Sequence Processing:**

- Variable-length sequences padded to max length
- Padding value: 0.0

**4. Label Encoding:**

- LabelEncoder for emotion categories
- Conversion to integer labels: happiness=0, neutral=1, sadness=2

#### Model Architectures

#### 1. Vanilla RNN

**Not implemented separately** - Focus on gated architectures due to vanishing gradient issues

#### 2. GRU (Gated Recurrent Unit)

```python
Architecture:
- Embedding dimension: 300 (from Word2Vec)
- GRU layers: 1
- Hidden dimension: Varied [128, 256]
- Dropout: Varied [0.1, 0.5]
- Output: Fully connected layer (hidden_dim → 3 classes)
```

#### 3. LSTM (Long Short-Term Memory)

```python
Architecture:
- Embedding dimension: 300 (from Word2Vec)
- LSTM layers: 1
- Hidden dimension: Varied [128, 256]
- Dropout: Varied [0.1, 0.5]
- Output: Fully connected layer (hidden_dim → 3 classes)
```

**Key Difference:** LSTM uses cell state to better capture long-term dependencies compared to GRU.

#### Hyperparameter Experiments

**Grid Search Space:**

- Hidden dimensions: [128, 256]
- Dropout rates: [0.1, 0.5]
- Learning rates: [0.001, 0.0001]
- Optimizers: [Adam, SGD with momentum=0.9]
- Batch size: 32
- Epochs: 5 (for quick experimentation)

**Total Configurations:** 2×2×2×2 = 16 configurations per model type

#### Training Details

**Loss Function:** Cross-Entropy Loss

**Data Loaders:**

- Training: Shuffled batches
- Validation: Sequential batches
- Test: Sequential batches

**Training Loop:**

1. Forward pass through RNN
2. Loss computation
3. Backpropagation
4. Optimizer step
5. Validation evaluation each epoch

#### Results

**Best GRU Configuration:**

- Hidden dimension: 256
- Dropout: 0.1
- Learning rate: 0.001
- Optimizer: Adam
- **Test Accuracy:** ~52%

**Best LSTM Configuration:**

- Hidden dimension: 256
- Dropout: 0.1
- Learning rate: 0.001
- Optimizer: Adam
- **Test Accuracy:** ~51%

#### Confusion Matrix Analysis

**Observations:**

- Happiness class: Best recall and precision
- Neutral class: Moderate confusion with sadness
- Sadness class: Some misclassification as neutral
- Diagonal dominance indicates reasonable performance

#### Loss and Accuracy Curves

**Training Dynamics:**

- Steady decrease in training loss over epochs
- Test loss follows similar trend
- Accuracy increases gradually
- Minimal overfitting gap between train and test

#### Comparative Analysis

**1. Effect of Learning Rate:**

- **LR=0.001:** Optimal balance with Adam optimizer
- **LR=0.0001:** Slower convergence, may need more epochs
- **High LR (>0.01):** Risk of instability

**2. Effect of Optimizer:**

- **Adam:** Fast convergence, adaptive learning rates, minimal tuning
- **SGD:** Requires careful learning rate tuning, slower convergence
- **SGD with high dropout (0.5):** Tends to stagnate early

**3. Effect of Dropout:**

- **Dropout=0.1:** Good generalization, minimal underfit
- **Dropout=0.5:** May cause underfitting with limited epochs/data

**4. Effect of Hidden Dimension:**

- **Hidden=128:** Sufficient for this task
- **Hidden=256:** Marginal improvement, more capacity

**5. GRU vs LSTM:**

- **GRU:** Slightly faster training, fewer parameters
- **LSTM:** Better long-term dependencies, marginally better accuracy
- **Difference:** Minimal in this task (both ~50-52% accuracy)

#### Key Insights

**1. Hyperparameter Tuning Challenges:**
The combinatorial explosion of hyperparameters makes exhaustive search impractical:

- Grid search with 4 parameters and 2-4 values each = 16-256 combinations
- Each configuration requires full training cycle
- Resource-intensive and time-consuming

**2. Practical Strategies:**

- **Informed Search:** Use theoretical knowledge (e.g., Adam generally better than SGD)
- **Coarse-to-Fine:** Start with broad ranges, then narrow down
- **Random Search:** Often more efficient than grid search
- **Early Stopping:** Avoid full training for obviously poor configurations

**3. Optimizer Selection:**

- Adam's adaptive learning rates make it more robust
- SGD requires more careful tuning but offers theoretical guarantees
- For quick experimentation, Adam is preferred

**4. Regularization Trade-offs:**

- Too much dropout (0.5) with limited data → underfitting
- Too little dropout (<0.1) → potential overfitting
- Balance depends on dataset size and model capacity

#### Limitations & Future Work

**Current Limitations:**

- Only 5 epochs trained (due to computational constraints)
- Limited hyperparameter search space
- Pre-trained embeddings not fine-tuned
- No attention mechanism

**Potential Improvements:**

- Increase training epochs (20-50)
- Implement attention mechanisms
- Use BERT or other transformer-based embeddings
- Data augmentation (synonym replacement, back-translation)
- Ensemble methods
- Class balancing techniques

#### Conclusion

This project demonstrates:

1. RNNs can effectively model sequential text data
2. Hyperparameter tuning significantly impacts performance
3. Pre-trained embeddings (Word2Vec) provide strong baselines
4. Gated architectures (GRU/LSTM) handle sentiment analysis reasonably well
5. Achieving >47% accuracy target validates the approach

---

## Project 3: Adversarial Attacks & Contrastive Learning

### Overview

This project explores adversarial robustness and self-supervised learning through contrastive methods. It consists of adversarial attacks on CNN classifiers and contrastive learning for image embeddings.

### Part 1: Training a CNN on SVHN

#### Dataset

**SVHN (Street View House Numbers):**

- Real-world digit images from house numbers
- 10 classes (digits 0-9)
- More challenging than MNIST due to natural variation
- Available through `torchvision.datasets`

#### Objective

Train a CNN classifier achieving at least 90% test accuracy.

#### Model Architecture: ConvNet

```python
Convolutional Layers:
Block 1:
  - Conv2d(3→64, kernel=3, padding=1)
  - BatchNorm2d(64)
  - ELU activation
  - Conv2d(64→128, kernel=3, padding=1)
  - BatchNorm2d(128)
  - ELU activation
  - MaxPool2d(2)
  - Dropout(0.2)

Block 2:
  - Conv2d(128→256, kernel=3, padding=1)
  - BatchNorm2d(256)
  - ELU activation
  - Conv2d(256→256, kernel=3, padding=1)
  - BatchNorm2d(256)
  - ELU activation
  - MaxPool2d(2)
  - Dropout(0.3)

Block 3:
  - Conv2d(256→512, kernel=3, padding=1)
  - BatchNorm2d(512)
  - ELU activation
  - MaxPool2d(2)
  - Dropout(0.4)

Fully Connected Layers:
  - Linear(512×4×4 → 1024)
  - ELU activation
  - Dropout(0.5)
  - Linear(1024 → 10)
```

#### Training Configuration

**Hyperparameters:**

- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Scheduler: StepLR (step_size=5, gamma=0.5)
- Epochs: 10
- Mixed precision training: Enabled (AMP)

**Data Preprocessing:**

- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- No additional augmentation initially

**Regularization:**

- Batch normalization after each conv layer
- Increasing dropout rates (0.2 → 0.5) in deeper layers
- L2 regularization implicit in Adam

#### Results

- **Final Test Accuracy:** 91.23%
- Successfully exceeds 90% threshold
- Stable convergence with smooth loss curves

#### Model Analysis

**Confusion Matrix Insights:**

- Most errors occur between visually similar digits (e.g., 3 vs 8, 4 vs 9)
- Diagonal dominance indicates good overall performance
- Some classes (0, 1, 6) have near-perfect classification

**Error Analysis:**

- Misclassified images often have:
  - Partial occlusions
  - Unusual viewpoints
  - Multiple digits in frame
  - Poor lighting conditions
  - Motion blur

### Part 2: Adversarial Attacks (FGSM)

#### Theory: Fast Gradient Sign Method (FGSM)

**Concept:** Generate adversarial examples by adding small perturbations in the direction of the gradient that maximizes loss.

**Formula:**

$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))
$$

Where:

- $x$: Original input image
- $\epsilon$: Perturbation magnitude
- $L$: Loss function
- $\nabla_x L$: Gradient of loss with respect to input

**Goal:** Create imperceptible perturbations that cause misclassification.

#### Implementation

```python
def fgsm_attack(model, images, labels, epsilon):
    # Enable gradient computation for input
    images.requires_grad = True
  
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
  
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
  
    # Generate adversarial perturbation
    perturbation = epsilon * images.grad.sign()
    adversarial_images = images + perturbation
  
    return adversarial_images
```

#### Evaluation Function

```python
def eval_adversarial(model, test_loader, epsilon):
    """
    Evaluate model accuracy on FGSM-perturbed images
    """
    correct = 0
    total = 0
  
    for images, labels in test_loader:
        images.requires_grad = True
    
        # Generate adversarial examples
        adv_images = fgsm_attack(model, images, labels, epsilon)
    
        # Evaluate on perturbed images
        outputs = model(adv_images.detach())
        _, predicted = outputs.max(1)
    
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
  
    return 100 * correct / total
```

#### Results: Attack Success

**Baseline (ε=0):** 91.23% accuracy

**Attack Results:**

- **ε=0.02:** 78.4% accuracy (-12.8%)
- **ε=0.05:** 52.1% accuracy (-39.1%)
- **ε=0.1:** 23.7% accuracy (-67.5%) ✓ (meets <25% requirement)
- **ε=0.2:** 8.9% accuracy (-82.3%)
- **ε=0.3:** 5.2% accuracy (-86.0%)

**Key Finding:** At ε=0.1, the model's accuracy drops from 91% to 24%, demonstrating vulnerability to adversarial attacks.

#### Visualization Analysis

**1. Flipped Predictions:**
Images correctly classified before attack but misclassified after:

- Visual differences are barely perceptible to humans
- Model confidently predicts wrong classes
- Demonstrates adversarial perturbations' effectiveness

**2. Adversarial Confusion Matrix:**
Shows systematic misclassifications:

- Certain class confusions are more common
- Attack causes non-uniform error distribution
- Some digits more vulnerable than others

**3. Perceptibility Analysis:**

**Human Perception Threshold:**

- **ε ≤ 0.05:** Perturbations imperceptible to humans, images look unchanged
- **ε = 0.1:** Slight noise visible but digits still clearly recognizable
- **ε ≥ 0.2:** Obvious visual degradation, salt-and-pepper noise visible
- **ε ≥ 0.3:** Significant noise, harder for humans to classify

**Key Insight:** The "sweet spot" for adversarial attacks is ε≈0.05-0.1, where perturbations fool the model but remain imperceptible to humans.

### Part 3: Adversarial Training

#### Theory

**Adversarial Training** improves model robustness by including adversarial examples in the training process.

**Approach:**
For each mini-batch:

1. Generate adversarial examples using FGSM
2. Concatenate original and adversarial examples
3. Train on combined dataset
4. Model learns to be robust against perturbations

#### Implementation

```python
def train_network_adversarial(num_epochs=10, epsilon=0.1):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Generate adversarial examples
            images.requires_grad = True
            adv_images = fgsm_attack(model, images, labels, epsilon)
        
            # Combine original and adversarial
            combined_images = torch.cat([images, adv_images])
            combined_labels = torch.cat([labels, labels])
        
            # Train on combined dataset
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### Training Configuration

- Epochs: 10
- Epsilon: 0.1 (same as attack)
- Effective batch size: 2× original (includes adversarial examples)
- All other hyperparameters unchanged

#### Results: Adversarial Training

**Clean Test Accuracy:**

- Before adversarial training: 91.23%
- After adversarial training: 88.15% (-3.1%)

**Adversarial Test Accuracy (ε=0.1):**

- Before adversarial training: 23.7%
- After adversarial training: 73.4% (+49.7%) ✓ (exceeds 70% requirement)

**Trade-off:**

- Small decrease in clean accuracy (~3%)
- Massive increase in adversarial robustness (~50%)
- Excellent cost-benefit ratio

#### Confusion Matrix Comparison

**After Adversarial Training:**

- Much stronger diagonal elements
- Fewer systematic confusions
- More balanced error distribution
- Model maintains structure even under attack

**Observations:**

- Certain digit pairs still confusing (3↔8, 4↔9)
- Overall misclassification rate dramatically reduced
- Model generalizes better across perturbation space

#### Key Insights

**1. Vulnerability vs Robustness:**

- Standard training optimizes for clean data only
- Models develop brittle decision boundaries
- Small perturbations can cross boundaries easily

**2. Adversarial Training Benefits:**

- Smooths decision boundaries
- Increases margin between classes
- Model learns invariant features
- Generalizes to perturbed inputs

**3. Trade-offs:**

- Slight decrease in clean accuracy acceptable
- Significant robustness gain justifies cost
- Critical for safety-critical applications

**4. Practical Applications:**

- Security systems (face recognition)
- Autonomous vehicles (sign detection)
- Medical imaging (tumor detection)
- Any system vulnerable to adversarial inputs

### Part 4: Contrastive Learning (SimCLR)

#### Overview

Implement self-supervised contrastive learning to create meaningful image embeddings without labeled data.

#### Dataset

**Tiny ImageNet:**

- Subset of ImageNet
- 200 classes
- 96×96 pixel images
- Training split: ~100,000 images
- Test split: ~10,000 images

#### Theory: SimCLR Framework

**Concept:** Learn representations by maximizing agreement between differently augmented views of the same image.

**Pipeline:**

1. **Data Augmentation:** Create two random views (x_i, x_j) of same image
2. **Encoder:** CNN extracts features (h_i, h_j)
3. **Projection Head:** MLP maps to embedding space (z_i, z_j)
4. **Contrastive Loss:** Maximize similarity of positive pairs, minimize for negative pairs

**Key Idea:** Images are their own labels! Augmented versions should have similar embeddings.

#### Data Augmentation Strategy

**Training Augmentations (Aggressive):**

- Random resized crop (96×96)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale (20% probability)
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

**Test Augmentations (Minimal):**

- Resize to 96×96
- Normalization only

**Rationale:**

- Strong augmentations force model to learn invariant features
- Two views of same image should be similar despite transformations
- Creates rich positive pairs for contrastive learning

#### Model Architecture: ContrastiveResNet18

**Base Encoder:**

- ResNet18 (pre-trained on ImageNet)
- Remove final classification layer
- Use as feature extractor

**Projection Head:**

```python
Fully Connected Layers:
- Linear(512 → 256)
- BatchNorm1d(256)
- ReLU
- Linear(256 → 128)
- BatchNorm1d(128)
- ReLU
- Linear(128 → 64)  # Final embedding dimension
```

**Design Choices:**

- Pre-trained backbone for faster convergence
- Multi-layer projection head for better representations
- BatchNorm for stable training
- Output embedding: 64 dimensions

#### Loss Function: NT-Xent (Normalized Temperature-scaled Cross-Entropy)

**Formula:**

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Where:

- $z_i, z_j$: Embeddings of two augmented views (positive pair)
- $\text{sim}(u, v)$: Cosine similarity
- $\tau$: Temperature parameter (0.5)
- $N$: Batch size

**Implementation:**

```python
def nt_xent_loss(embeddings1, embeddings2, temp=0.5):
    batch_size = embeddings1.size(0)
  
    # Concatenate both views
    combined = torch.cat([embeddings1, embeddings2], dim=0)
  
    # Compute similarity matrix
    similarity = torch.matmul(combined, combined.T) / temp
  
    # Labels: each image's positive pair
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)])
  
    # Cross-entropy loss
    loss = F.cross_entropy(similarity, labels)
    return loss
```

**Key Properties:**

- Pulls positive pairs together
- Pushes negative pairs apart
- Temperature controls softness of distribution
- Symmetric loss (both views treated equally)

#### Training Configuration

**Hyperparameters:**

- Batch size: 256 (large for more negatives)
- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 2 (demonstration)
- Temperature: 0.5

**Why Large Batch Size?**

- More negative samples per batch
- Richer contrastive signal
- Better gradient estimates
- Critical for contrastive learning success

#### Results

- **Final Loss:** <3.0 ✓ (meets requirement)
- Successfully converged in 2 epochs
- Loss decreased from ~6.5 to ~2.8

#### Evaluation: Embedding Visualization

**1. t-SNE Visualization:**

- Projects 64D embeddings to 2D
- Displays test images at their embedding locations
- Reveals learned structure

**Observations:**

- Similar images cluster together
- Clear visual groupings emerge
- Unsupervised learning discovers semantic structure

**2. Nearest Neighbor Analysis:**

For 3 query images, find 5 nearest neighbors based on embedding distance:

- **Measure:** L2 distance in embedding space
- **Expected:** Visually similar images should be close

**Results:**

- Query: Dog → Neighbors: Similar dogs
- Query: Car → Neighbors: Other vehicles
- Query: Building → Neighbors: Similar architectures

**Quality Assessment:**

- Semantic similarity captured reasonably well
- Some neighbors very relevant, others less so
- 2 epochs insufficient for perfect embeddings
- Longer training would improve quality

#### Dry Questions & Answers

**Q1: Large or small batch size for contrastive learning?**
**A:** **Large batch size is strongly preferred** because:

- More negative samples per image
- Richer contrastive signal
- Better gradient estimates
- Improved representation quality
- SimCLR paper uses batch sizes of 4096+

**Q2: Evaluation metrics for unsupervised representation learning?**
**A:** Several approaches:

- **Linear Evaluation:** Train linear classifier on frozen embeddings
- **K-NN Classification:** Use nearest neighbors in embedding space
- **Clustering Metrics:** Silhouette score, NMI
- **Downstream Tasks:** Fine-tune on specific tasks
- **Embedding Quality:** Inter/intra-class distances

**Q3: Creating embeddings for test set?**
**A:** **Only forward pass through encoder:**

- No augmentation (or minimal)
- No contrastive loss computation
- Extract features directly
- Use for downstream tasks

**Q4: Which augmentations for SimCLR?**

| Augmentation      | Use?    | Reason                                     |
| ----------------- | ------- | ------------------------------------------ |
| Random Crop       | ✓ Yes  | Creates diverse views, essential           |
| Enlarge 128×128  | ✗ No   | Changes scale, not semantic invariance     |
| Random Rotation   | ✓ Yes  | Orientation invariance, useful             |
| Gaussian Noise    | ? Maybe | Small amounts okay, too much destroys info |
| Random Dimensions | ✗ No   | Breaks aspect ratio, unnatural             |
| Random Grayscale  | ✓ Yes  | Color invariance, used in SimCLR           |

**Best Practices:**

- Use augmentations that preserve semantic content
- Avoid augmentations that destroy too much information
- Combination of geometric and photometric transforms

#### Limitations & Future Work

**Current Limitations:**

- Only 2 epochs trained
- Small embedding dimension (64)
- No fine-tuning evaluation
- Limited quantitative metrics

**Potential Improvements:**

- Train for 100+ epochs
- Larger batch sizes (512, 1024)
- Larger embedding dimension (128, 256)
- Momentum encoder (MoCo)
- Memory bank for more negatives
- Evaluate on downstream tasks
- Compare with supervised baseline

#### Key Insights

**1. Self-Supervised Learning:**

- No labels required for meaningful representations
- Data augmentation is the key
- Contrastive learning very effective

**2. Representation Quality:**

- Embeddings capture semantic similarity
- Unsupervised structure discovery
- Transferable to downstream tasks

**3. Practical Applications:**

- Pre-training for limited labeled data
- Transfer learning
- Image retrieval
- Clustering and organization

---

## Project 4: Generative Adversarial Networks (GANs)

### Overview

This project implements and analyzes Generative Adversarial Networks (GANs) with different latent space dimensions to generate realistic flower images.

### Theory: GANs

#### Concept

GANs consist of two neural networks competing in a minimax game:

**Generator (G):**

- Input: Random noise vector $z \sim \mathcal{N}(0, 1)$
- Output: Synthetic image $G(z)$
- Goal: Generate realistic images to fool discriminator

**Discriminator (D):**

- Input: Real or fake image
- Output: Probability image is real
- Goal: Distinguish real from fake images

#### Objective Function

$$
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

**Training Dynamics:**

- D tries to maximize classification accuracy
- G tries to minimize D's ability to classify
- Equilibrium: G generates perfect fakes, D cannot distinguish

### Dataset: Oxford Flowers 102

**Specifications:**

- 102 flower categories
- Test split: ~1,000 images
- Resolution: 64×64 pixels (resized)
- Natural images with high variation

**Preprocessing:**

- Resize to 64×64
- Convert to tensor [0, 1]
- Normalize: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- Maps to [-1, 1] range (for Tanh output)

### Model Architectures

#### Generator: DCGAN Style

```python
Architecture (z_dim → 64×64×3 image):

Input: Noise vector (batch, z_dim, 1, 1)

Block 1: ConvTranspose2d(z_dim → g_feat*8, kernel=4, stride=1)
  - BatchNorm2d
  - ReLU
  - Output: (batch, 512, 4, 4)

Block 2: ConvTranspose2d(g_feat*8 → g_feat*4, kernel=4, stride=2)
  - BatchNorm2d
  - ReLU
  - Output: (batch, 256, 8, 8)

Block 3: ConvTranspose2d(g_feat*4 → g_feat*2, kernel=4, stride=2)
  - BatchNorm2d
  - ReLU
  - Output: (batch, 128, 16, 16)

Block 4: ConvTranspose2d(g_feat*2 → g_feat, kernel=4, stride=2)
  - BatchNorm2d
  - ReLU
  - Output: (batch, 64, 32, 32)

Block 5: ConvTranspose2d(g_feat → 3, kernel=4, stride=2)
  - Tanh activation
  - Output: (batch, 3, 64, 64)
```

**Key Features:**

- Uses transposed convolutions for upsampling
- Batch normalization for stable training
- ReLU activations
- Tanh output for [-1, 1] range
- Progressively increases spatial dimensions

#### Discriminator: DCGAN Style

```python
Architecture (64×64×3 image → binary classification):

Input: Image (batch, 3, 64, 64)

Block 1: Conv2d(3 → d_feat, kernel=4, stride=2)
  - LeakyReLU(0.2)
  - Output: (batch, 64, 32, 32)

Block 2: Conv2d(d_feat → d_feat*2, kernel=4, stride=2)
  - BatchNorm2d
  - LeakyReLU(0.2)
  - Output: (batch, 128, 16, 16)

Block 3: Conv2d(d_feat*2 → d_feat*4, kernel=4, stride=2)
  - BatchNorm2d
  - LeakyReLU(0.2)
  - Output: (batch, 256, 8, 8)

Block 4: Conv2d(d_feat*4 → d_feat*8, kernel=4, stride=2)
  - BatchNorm2d
  - LeakyReLU(0.2)
  - Output: (batch, 512, 4, 4)

Block 5: Conv2d(d_feat*8 → 1, kernel=4, stride=1)
  - Output: (batch, 1, 1, 1)
```

**Key Features:**

- Strided convolutions for downsampling (no pooling)
- Batch normalization (except first layer)
- LeakyReLU for better gradients
- No activation on output (BCEWithLogitsLoss includes sigmoid)

### Experimental Setup

#### Objective

Train GANs with three different latent space dimensions (z_dim) and compare their performance:

- **z_dim = 10:** Low-dimensional latent space
- **z_dim = 100:** Standard latent space (DCGAN default)
- **z_dim = 500:** High-dimensional latent space

#### Training Configuration

**Hyperparameters:**

- Batch size: 128
- Learning rate: 2e-4
- Optimizer: Adam (β₁=0.5, β₂=0.999)
- Loss: BCEWithLogitsLoss
- Epochs: 50 (main experiments)
- Labels: real=1.0, fake=0.0
- Random seed: 42 (reproducibility)

**Training Procedure:**

```python
For each epoch:
    For each batch:
        # Train Discriminator
        1. Forward real images → D → loss_real
        2. Generate fake images: z → G → fake
        3. Forward fake images → D → loss_fake
        4. loss_D = loss_real + loss_fake
        5. Backprop and update D
    
        # Train Generator
        6. Generate new fake images: z → G → fake
        7. Forward fake images → D (G's perspective)
        8. loss_G = -log(D(G(z))) [want D to classify as real]
        9. Backprop and update G
```

### Results

#### Experiment 1: z_dim = 10 (Low Dimension)

**Observations:**

- **Loss Behavior:**

  - Discriminator: Stable, moderate values (~1.0-1.5)
  - Generator: Higher variance, struggles initially
- **Generated Images:**

  - Limited diversity
  - Captures basic color schemes (yellows, reds, greens)
  - Lacks fine details and texture
  - Some mode collapse evident
- **Quality:** Poor to fair
- **Diversity:** Low (limited by small latent space)

**Insight:** 10 dimensions insufficient to capture flower image complexity. Latent space too constrained.

#### Experiment 2: z_dim = 100 (Standard Dimension)

**Observations:**

- **Loss Behavior:**

  - Discriminator: Oscillates around 0.8-1.2
  - Generator: More stable than z=10
  - Healthy adversarial dynamics
- **Generated Images:**

  - Recognizable flower structures
  - Good color diversity
  - Reasonable texture details
  - Clear petals and flower centers
  - Better variety across samples
- **Quality:** Good
- **Diversity:** High

**Insight:** 100 dimensions provides excellent balance. Standard choice for DCGAN architecture validated.

#### Experiment 3: z_dim = 500 (High Dimension)

**Observations:**

- **Loss Behavior:**

  - Discriminator: Sometimes collapses to near-zero
  - Generator: High variance, instability
  - Training dynamics less healthy
- **Generated Images:**

  - Highly variable quality
  - Some excellent samples
  - Some mode collapse
  - Occasional artifacts
  - Diversity high but less consistent
- **Quality:** Variable (good to excellent)
- **Diversity:** Very high but inconsistent

**Insight:** 500 dimensions gives G more capacity but training becomes unstable. May require longer training or different hyperparameters.

#### Comparative Loss Curves

**Training Step Level:**

- All three models show characteristic GAN oscillations
- z=100 has smoothest curves
- z=10 shows G struggling more
- z=500 has highest variance

**Epoch Level:**

- z=100 most stable convergence
- z=10 plateaus early
- z=500 continues oscillating

### Latent Space Analysis

#### Latent Space Interpolation

**Method:**

1. Sample two random latent vectors: $z_1, z_2 \sim \mathcal{N}(0,1)$
2. Create interpolations: $z_{\alpha} = (1-\alpha)z_1 + \alpha z_2$ for $\alpha \in [0,1]$
3. Generate images: $G(z_{\alpha})$
4. Visualize smooth transition

**Results:**

**z_dim = 10:**

- Abrupt transitions between images
- Limited smooth variation
- Latent space not well-structured

**z_dim = 100:**

- Smooth, gradual transitions
- Semantic morphing (one flower gradually becomes another)
- Well-structured latent space
- Colors, shapes, sizes change smoothly

**z_dim = 500:**

- Very smooth transitions
- Rich variations
- Sometimes unexpected transformations
- High-dimensional manifold well-explored

**Insight:** Interpolation reveals latent space structure. Smooth transitions indicate continuous, meaningful representations.

#### PCA Visualization of Latent Space

**Method:**

1. Sample 500 latent vectors from prior $z \sim \mathcal{N}(0,1)$
2. Apply PCA to reduce to 2D
3. Scatter plot of latent codes

**Observations (all z_dims):**

- Approximately circular/elliptical distribution
- Standard Gaussian structure preserved
- No obvious clusters (unsupervised generation)
- Uniform coverage of latent space

**Insight:** Latent codes maintain Gaussian properties. No inherent structure in latent space (structure emerges through generator mapping).

#### Latent Code → Generated Image Mapping

**Visualization:**

- Sample 3 random latent vectors
- Reduce to 2D with PCA
- Display corresponding generated images

**Analysis:**

- Nearby latent codes produce similar images
- Latent space has semantic meaning
- Generator learned smooth manifold
- Walking in latent space = walking in image space

### Extended Training: z_dim=100, 100 Epochs

**Purpose:** Investigate if longer training improves quality.

**Results:**

- **Image Quality:** Significantly improved
- **Details:** Finer texture, more realistic petals
- **Consistency:** More samples look realistic
- **Diversity:** Maintained or slightly increased
- **Mode Collapse:** Minimal

**Loss Dynamics:**

- Continued oscillation (healthy GAN behavior)
- No signs of convergence to equilibrium
- Both G and D still improving

**Conclusion:** Extended training beneficial. 50 epochs good, 100 epochs better. Diminishing returns likely beyond this point.

### Model Persistence

**Saved Models:**

```
generator_zdim10.pkl
discriminator_zdim10.pkl
generator_zdim100.pkl
discriminator_zdim100.pkl
generator_zdim500.pkl
discriminator_zdim500.pkl
generator_zdim100_100epochs.pkl
discriminator_zdim100_100epochs.pkl
```

**Usage:**

```python
# Load generator
G = torch.load('generator_zdim100.pkl')
G.eval()

# Generate images
z = torch.randn(16, 100, 1, 1)
fake_images = G(z)
```

### Key Insights & Conclusions

#### 1. Latent Dimension Impact

**z_dim = 10:**

- ❌ Insufficient capacity
- ❌ Limited diversity
- ❌ Poor image quality
- **Use Case:** Proof of concept only

**z_dim = 100:**

- ✓ Optimal balance
- ✓ Stable training
- ✓ Good quality and diversity
- ✓ Standard choice validated
- **Use Case:** Production, research

**z_dim = 500:**

- ✓ High capacity
- ⚠️ Training instability
- ✓ Potential for excellent results
- ⚠️ Requires careful tuning
- **Use Case:** When quality >> stability

#### 2. Training Dynamics

**Healthy GAN Training:**

- Oscillating losses (not converging to zero)
- D slightly ahead but not dominant
- G improving over time
- No mode collapse

**Signs of Issues:**

- D loss → 0 (too strong)
- G loss → ∞ (can't fool D)
- Mode collapse (same images)
- Gradient vanishing

#### 3. Practical Recommendations

**For Best Results:**

- Use z_dim=100 as baseline
- Train for 50-100 epochs minimum
- Monitor generated samples regularly
- Save checkpoints frequently
- Use learning rate scheduling if needed

**Hyperparameter Sensitivity:**

- Learning rate: 2e-4 works well (DCGAN paper)
- Beta1: 0.5 critical (momentum for GANs)
- Batch size: 128 good balance
- Architecture: DCGAN proven effective

#### 4. Evaluation Challenges

**Quantitative Metrics:**

- FID (Fréchet Inception Distance): Measures distribution similarity
- IS (Inception Score): Measures quality and diversity
- Human evaluation: Ultimate test

**Qualitative Assessment:**

- Visual inspection
- Diversity check
- Interpolation smoothness
- Mode collapse detection

#### 5. Applications

**Image Generation:**

- Data augmentation
- Art generation
- Design assistance

**Latent Space:**

- Image editing (latent code manipulation)
- Interpolation for smooth transitions
- Attribute manipulation

**Research:**

- Understanding generative models
- Exploring learned representations
- Benchmark for new GAN variants

### Limitations & Future Work

**Current Limitations:**

- Small dataset (Flowers 102)
- Limited resolution (64×64)
- No conditional generation
- Single architecture tested

**Potential Improvements:**

- Progressive GAN (higher resolution)
- StyleGAN architecture
- Conditional GAN (class-specific)
- Wasserstein GAN (improved training)
- Spectral normalization
- Self-attention layers

**Advanced Techniques:**

- FID/IS metric computation
- Hyperparameter search
- Multiple runs for statistical significance
- Comparison with other generative models (VAE, diffusion)

---

## Technologies & Libraries Used

### Core Frameworks

- **PyTorch 2.0+:** Deep learning framework
- **torchvision:** Computer vision utilities and datasets
- **NumPy:** Numerical computations

### Data Processing

- **Pandas:** Data manipulation (Project 2)
- **PIL (Pillow):** Image processing
- **gensim:** Word embeddings (Project 2)

### Visualization

- **Matplotlib:** Plotting and visualization
- **Seaborn:** Statistical visualizations
- **scikit-learn:** t-SNE, PCA, metrics

### Utilities

- **tqdm:** Progress bars
- **sklearn:** Confusion matrices, label encoding

---

## Installation & Setup

### Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install gensim pillow tqdm
pip install kagglehub  # For Project 3 dataset
```

### Running Projects

**Project 1:**

```bash
cd Project1
jupyter notebook project1.ipynb
# Or run: python project1.py
```

**Project 2:**

```bash
cd Project2
jupyter notebook project2.ipynb
```

**Project 3:**

```bash
cd Project3
jupyter notebook project3.ipynb
```

**Project 4:**

```bash
cd Project4
jupyter notebook project4.ipynb
```

---

## Key Takeaways

### Technical Skills Developed

1. **Neural Network Fundamentals:** Manual backpropagation, gradient computation
2. **CNN Architectures:** Efficient design, depthwise-separable convolutions
3. **RNN/LSTM/GRU:** Sequence modeling, sentiment analysis
4. **Adversarial Machine Learning:** FGSM attacks, adversarial training
5. **Self-Supervised Learning:** Contrastive learning, SimCLR
6. **Generative Models:** GANs, latent space analysis

### Theoretical Understanding

1. **Optimization:** Learning rates, optimizers, convergence
2. **Regularization:** Dropout, batch normalization, data augmentation
3. **Generalization:** Overfitting, train-test gap, robustness
4. **Loss Functions:** Cross-entropy, contrastive loss, adversarial loss
5. **Evaluation:** Accuracy, confusion matrices, qualitative analysis

### Best Practices Learned

1. **Data preprocessing is critical**
2. **Start simple, iterate complexity**
3. **Monitor training curves religiously**
4. **Use proper validation strategies**
5. **Hyperparameter tuning is an art and science**
6. **Visualize**
7. **Save checkpoints frequently**
8. **Document experiments thoroughly**

---

## Acknowledgments

These projects were completed as part of the **Machine Learning 2** course at **Technion**. Special thanks to the course instructors for designing comprehensive assignments that cover both theoretical foundations and practical applications of modern deep learning techniques.

---

## License

These projects are for educational purposes as part of academic coursework.
