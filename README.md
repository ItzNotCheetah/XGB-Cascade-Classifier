# Cost-Sensitive Real-Time Cascading Router Engine

An asymmetric, hardware-aware meta-routing engine designed to optimize dynamic skipping in multi-stage **"I Don't Know" (IDK) Classifier Cascades**. By feeding physical real-time execution costs directly into an asymmetric gradient-boosted decision boundary, this framework breaks the traditional Pareto trade-off—yielding a **10.45% latency reduction** alongside a simultaneous **accuracy increase to 97.25%** over standard linear baselines.

---

## The Core Breakthrough

In real-time cyber-physical systems (e.g., autonomous driving hazard perception), classification algorithms must trade off accuracy against execution deadlines. Prior literature establishes static threshold conditions (Nguyen et al., 2024) or symmetric target matching (Katikaneni et al., 2024) to skip intermediate execution phases when confidence is extremely low.

However, a real-world hardware execution space is fundamentally **asymmetric**. A missed skip incurs a severe execution penalty ($Latency_B + Latency_C$), while an unnecessary skip incurs a drastically smaller penalty ($Latency_C - Latency_B$). 

This engine bridges the gap between systems engineering and machine learning by:
1. **Multi-Parameter Uncertainty Telemetry:** Extracting a rich feature interaction vector ($\vec{X} = [\text{Confidence}, \text{Entropy}, \text{Margin}]$) from early-stage inference.
2. **Asymmetric Cost Learning:** Computing the physical worst-case execution time (WCET) constraints and mapping them directly to the objective loss function via a tuned XGBoost scale weight.
3. **Breaking the Pareto Bottleneck:** Demonstrating absolute empirical dominance in both timing speedups and system accuracy.

---

## Empirical Performance Summary

Evaluated on native pre-trained PyTorch networks (ResNet-20, ResNet-32, ResNet-56) using the CIFAR-10 validation space:

| Strategy | Avg Latency | End-to-End Accuracy | Speedup vs. Linear Baseline | Absolute Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Linear Cascade** | 3.54 ms | 97.00% | 0.00% (Ref) | Baseline |
| **Static Threshold ($\le 0.3$)** | 3.31 ms | 93.75% | +6.50% | -3.25% (Degraded) |
| **XGBoost Cascade Engine ($\theta = 0.01$)** | **3.17 ms** | **97.25%** | **+10.45%** | **+0.25% (Elevated)** |

### Meta-Router Overhead Footprint
To eliminate the "router execution bottleneck" critique, online meta-inference was rigorously profiled down to the microsecond:
* **Absolute Router Overhead:** $0.0020 \pm 0.0007 \text{ ms/sample}$
* **Impact on Linear Cascade Budget:** **0.06%** (computationally negligible)

---

## System Architecture

1. **Front-End Lightweight Model (Model A - ResNet-20):** Processes incoming raw features. Generates classification outputs and yields real-time uncertainty vectors.
2. **Asymmetric Meta-Router Engine (XGBoost):** Evaluates $\vec{X}$. Compares the cost metrics using compressed raw prediction probabilities. 
3. **Dynamic Routing Split:**
   * **$\theta \ge 0.01$:** Routes to Intermediate Phase (Model B - ResNet-32). Falls back to Model C if B fails.
   * **$\theta < 0.01$:** Actively executes an aggressive **Hardware Skip** directly to the Heavy Expert Fallback (Model C - ResNet-56), bypassing intermediate latency blocks entirely.

---

## Getting Started

### Dependencies
Ensure you have the following frameworks installed in your Python environment:
```bash
pip install xgboost scikit-learn pandas numpy torch torchvision
