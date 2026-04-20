The Self-Pruning Neural Network

Tredence Analytics AI Engineering Internship - Case Study

This repository contains an implementation of a dynamic, self-pruning neural network designed to optimize its architecture during training. Instead of a post-training compression step, this network identifies and removes unnecessary connections on the fly using a learnable gating mechanism and L1-based sparsity regularization.

## Key Features & Design Decisions

* **Modular Architecture:** The project is organized into distinct modules (`models`, `utils`, `main.py`) to ensure maintainability and scalability, adhering to "Clean Architecture" principles.
* **Custom PrunableLinear Layer:** Implemented a drop-in replacement for `nn.Linear` that utilizes a differentiable gating tensor.
* **Lambda Warm-up Schedule:** To prevent early accuracy collapse, the sparsity penalty ($\lambda$) is linearly ramped up over the initial 3 epochs. This allows the model to learn foundational features before the pruning pressure becomes significant.
* **Negative Gate Initialization:** Gate scores are initialized to $-1.0$, resulting in an initial sigmoid output of $\approx 0.27$. This "partially closed" state facilitates more efficient pruning than a standard $0.5$ initialization.
* **Blackwell Architecture Ready:** The pipeline is device-agnostic, featuring a fallback to CPU to handle driver/library mismatches common with new hardware like the NVIDIA RTX 5050 (Blackwell/sm_120).

---

## Mathematical Formulation

The objective is to minimize a composite loss function that balances classification performance with network sparsity:

$$TotalLoss = L_{CrossEntropy} + \lambda \times \sum_{i,j} |\sigma(s_{i,j})|$$

**Where:**
* $s_{i,j}$ represents the learnable `gate_scores` associated with each weight.
* $\sigma$ is the Sigmoid function, squashing scores into a $(0,1)$ range to act as soft masks.
* The **L1 penalty** (sum of absolute values) is used because its constant gradient ($\pm1$) effectively drives less significant gate values toward zero, inducing sparsity.

---

## Results Summary

The model was evaluated on the CIFAR-10 dataset across three different $\lambda$ coefficients.

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level (%) | Observation |
| :--- | :--- | :--- | :--- |
| $1 \times 10^{-4}$ | 50.79% | 98.80% | **Optimal Balance**: High sparsity with maintained performance. |
| $1 \times 10^{-3}$ | 46.83% | 99.93% | Aggressive pruning with moderate accuracy trade-off. |
| $1 \times 10^{-2}$ | 42.31% | 100.00% | **Extreme compression**: Model uses minimal active weights. |

Visualizations
gate_distribution.png: Located in the /plots folder, this histogram shows a clear spike at $0$, confirming that the network has successfully "self-pruned" redundant connections.
sparsity_tradeoff.png: Visualizes the relationship between the regularization strength and model accuracy.

Installation & Usage
1. Requirements:
Python 3.10+,
PyTorch,
Torchvision,
Matplotlib,
tqdm


Install dependencies:
pip install -r requirements.txt

3. Running the Experiment
Ensure you are in the root directory and have created a plots/ folder:
mkdir plots
python main.py

## Repository Structure

```text
├── data/               # CIFAR-10 raw data (git-ignored)
├── models/
│   └── prunable_net.py # PrunableLinear & Network Definition
├── utils/
│   └── trainer.py      # Training loop with L1 regularization
├── plots/              # Generated results and histograms
├── main.py             # Entry point for experiment execution
└── requirements.txt    # Project dependencies
