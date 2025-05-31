# FashionMNIST-Linear-vs-Non-Linear-vs-CNN-Model
# 🧠 Comparing Linear vs Non-Linear vs CNN Models on FashionMNIST

This project investigates and compares three different types of deep learning models trained on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist), aiming to understand their relative performance in terms of **accuracy**, **training time**, and **architectural complexity**:

- 🔹 **Model 0**: Linear Model (`FashionMNISTModelV0`)
- 🔹 **Model 1**: Non-Linear Model with ReLU Activation (`FashionMNISTModelV1`)
- 🔹 **Model 2**: Convolutional Neural Network (CNN - `FashionMNISTModelV2`, a TinyVGG)



## 🧩 Key Concepts in Play

- **Linear Model**: No hidden activation functions — essentially a multinomial logistic regression.
- **Non-Linear MLP**: Adds non-linearity via **ReLU**, increasing model capacity to capture complex patterns.
- **CNN**: Specialized in extracting **spatial hierarchies** using **convolutional layers**, **weight sharing**, and **local connectivity**.

---

## 🧪 Dataset Setup

All models are trained on the FashionMNIST dataset using:

- `torchvision.datasets.FashionMNIST`
- Transformations: `ToTensor()`
- Batch Size: 32
- Optimizer: `torch.optim.SGD`
- Loss: `CrossEntropyLoss`
- Epochs: 3
- Device: CUDA if available

---

## 🧮 Architecture Comparison

### 🔸 Model V0 - Linear Model

```python
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.Linear(hidden_units, output_shape)
        )
```
- Pros:

  - Extremely fast training (only 131s).

  - Simple and interpretable.

- Cons:

  - Limited learning capacity — can't capture complex spatial patterns.

  - No activation functions, hence strictly linear.

Suitable for baseline tasks or real-time constraints where training speed trumps accuracy.

🔸 Model V1 - Non-Linear (MLP) with ReLU

```python
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        )

```
- Pros:

    - Incorporates non-linearity → can learn more complex functions.

    - Improved accuracy (~0.35% boost).

- Cons:

     - Huge increase in training time (~4200s), mostly due to inefficient use of deep layers without spatial awareness.

     - Ideal when data isn't image-heavy or spatial patterns aren't vital — but less efficient for image classification.

🔸 Model V2 - CNN (TinyVGG)

```python
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

```
- Pros:
  - Best accuracy: 85.97%
  - Exploits spatial hierarchies with convolutions and pooling.
  - Great for image data due to parameter efficiency (compared to fully connected layers).

- Cons:
  - Highest training time (7663s) due to complex forward/backward computations.
  - Best-suited for any vision-related tasks — accuracy gain justifies training cost for most real-world applications.

---

## 📊 Performance Summary

| Model | Architecture Type | Accuracy (%) | Loss | Training Time (s) |
|-------|-------------------|--------------|------|-------------------|
| **V0** | Linear            | 82.96        | 0.492 | 131.91            |
| **V1** | Non-Linear (ReLU) | 83.31        | 0.464 | 4208.89           |
| **V2** | CNN (TinyVGG)     | **85.97**    | **0.359** | **7663.90**      |

---

<p align="center">
  <img src="https://github.com/Ahnuf-Karim-Chowdhury/FashionMNIST-Linear-vs-Non-Linear-vs-CNN-Model/blob/main/Images/02-Linear_VS_Non-Linear_Vs_CNN.png?raw=true" style="max-width: 100%; height: auto;">
</p>

--- 

### 📈 Training Curves 

<p align="center">
  <img src="https://github.com/Ahnuf-Karim-Chowdhury/FashionMNIST-Linear-vs-Non-Linear-vs-CNN-Model/blob/main/Images/03-Linear_VS_Non-Linear_Vs_CNN.png?raw=true" style="max-width: 100%; height: auto;">
</p>


---
## 🧪 CNN Evaluation on FashionMNIST (The Model With the most Accuracy)

The image below showcases **sample predictions** made by a Convolutional Neural Network (CNN) trained on the FashionMNIST dataset:

<p align="center">
  <img src="https://github.com/Ahnuf-Karim-Chowdhury/FashionMNIST-Linear-vs-Non-Linear-vs-CNN-Model/blob/main/Images/model_2_prediction.png?raw=true" style="max-width: 100%; height: auto;">
</p>

---

### 🖼️ Grid Layout
- The 3×3 grid displays **9 test images**.
- Each image is accompanied by:
  - `Pred:` → Model's predicted class
  - `Truth:` → Actual ground-truth label

---

### 🎯 Prediction Color Coding

- ✅ **Correct Predictions**: Label is shown in **green**.
- ❌ **Incorrect Predictions**: Label is shown in **red** for visibility.

---

### ✅ Correct Predictions

The CNN correctly classified 8 out of 9 examples:

- **Sandal**
- **Trouser**
- **Sneaker**
- **Coat**
- **Dress**
- **Coat** (repeated)
- **Sneaker**
- **Trouser**

These results show the CNN's ability to generalize well on various clothing types with distinct shapes and textures.

---

### ❌ Incorrect Prediction

- 🔴 `Pred: Shirt | Truth: T-shirt/top`
  - **Analysis**: This is a reasonable confusion, as **shirts and T-shirts** share similar structural features in grayscale low-resolution images.
  - The CNN may struggle to capture subtle differences without color or fine-grained patterns.

---

### 📌 Insights

- **CNN Strengths**:
  - High accuracy on visually distinct items (e.g., Sneaker vs. Trouser).
  - Robust pattern recognition despite grayscale and low resolution.

- **Limitations**:
  - Ambiguities between similar-looking classes (Shirt vs T-shirt/top).
  - Possible improvements using:
    - Data augmentation
    - Deeper architectures
    - Attention mechanisms

---

### 📉 Performance Summary

| Metric              | Value       |
|---------------------|-------------|
| Correct Predictions | 8 / 9       |
| Accuracy (sample)   | ~88.89%     |
| Error Cases         | Shirt ↔ T-shirt/top |

---



### 💡 Why Use CNNs for Images?

CNNs shine on image data for several reasons:

- **Translation Invariance**: Learned via shared weights and pooling.
- **Sparse Interactions**: Each neuron processes only a small receptive field.
- **Parameter Efficiency**: Fewer weights compared to fully connected networks.

> Despite the high training cost, CNNs generalize better on image-based tasks and are more robust to distortions and noise.


### ⏱️ Efficiency vs Accuracy Tradeoff

| Model             | Training Time | Accuracy     | Use Case                        |
|------------------|---------------|--------------|---------------------------------|
| Linear            | 🟢 Fast       | 🔴 Low       | Prototyping, baseline           |
| Non-Linear (MLP)  | 🟡 Moderate   | 🟡 Slightly Better | Non-image structured data |
| CNN               | 🔴 Slow       | 🟢 Best      | Image recognition, production  |


### 📦 Dependencies

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- Torchvision  
- tqdm  
- matplotlib *(optional for plotting)*

**Install with:**

```bash
pip install torch torchvision tqdm matplotlib

```


---


### 📌 Conclusion

This comparison highlights the classic trade-off between speed and accuracy in deep learning:

- 🧱 Use **Linear Models** for fast prototyping or low-resource environments.
- 🧠 Use **Non-Linear Models** to capture deeper patterns in tabular/structured data.
- 🖼️ Use **CNNs** for images — they are domain-optimized and superior for vision tasks.

Ultimately, model selection depends on your problem domain, performance requirements, and hardware constraints.

---

### 🔗 References

- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- *Deep Learning with PyTorch* (Official Docs)
- [CS231n CNN Notes](https://cs231n.github.io/convolutional-networks/)



