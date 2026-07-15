# Real-Time Cost-Sensitive Cascading Router Engine (Delta Model Configuration)

An asymmetric, hardware-aware meta-routing engine designed to optimize dynamic skipping in multi-stage **"I Don't Know" (IDK) Classifier Cascades**. Implementing the specific designation of a machine learning cascade model configuration defined as the **Delta Model**, this framework breaks traditional execution barriers—yielding system acceleration alongside absolute empirical dominance in both timing speedups and end-to-end classification accuracy.

---

## The Core Breakthrough

In real-time systems engineering, classification algorithms must balance accuracy against execution deadlines. Prior literature establishes static threshold conditions or symmetric target matching to handle intermediate execution phases when confidence is extremely low.

However, a real-world hardware execution space is fundamentally **asymmetric**. This engine bridges the gap between systems engineering and machine learning by:
1. **Multi-Parameter Uncertainty Telemetry:** Extracting a rich feature interaction vector ($\vec{X} = [\text{Confidence}, \text{Entropy}, \text{Margin}]$) from early-stage inference.
2. **Asymmetric Cost Learning:** Mapping worst-case execution time constraints directly to the objective loss function via a tuned scale weight. The asymmetric weight multiplier ($\omega$) is derived dynamically based on physically measured execution latencies ($\tau$):
   $$\omega = \frac{\text{Cost of Missed Fallback}}{\text{Cost of Unnecessary Fallback}} \propto \frac{1.0}{\max(\tau_C, 10^{-6})}$$
   This explicitly penalizes the meta-router for missing critical expert corrections, while adjusting for the performance penalty of unnecessary complex executions.
3. **Breaking the Pareto Bottleneck:** Demonstrating absolute empirical dominance in both timing speedups and system accuracy.

---

## Key Telemetry Benchmarks

| Metric | Metric Value | Evaluation Context |
| :--- | :---: | :--- |
| **Meta-Router Micro-Overhead** | **0.0011 ms / sample** | Isolated inference footprint of the XGBoost engine |
| **Systemic Resource Penalty** | **0.30%** | Overhead expressed as a % of the lightweight Front-End Model |
| **Optimal End-to-End Accuracy** | **73.25%** | Achieved at threshold theta = 0.30 (Exceeds Pure Expert Baseline) |
| **Pure Expert Accuracy Baseline**| **72.80%** | Baseline performance when always invoking the heaviest model |
| **Statistical Skip Recall Robustness**| **98.46% + 1.42%** | Verification across 5 independent computational seeds |

---

## System Architecture

1. **Front-End Lightweight Model (Model A - ResNet-20):** Processes incoming raw features. Generates classification outputs, applies an immediate early-exit evaluation loop if the softmax confidence crosses the high threshold barrier ($E_{th} = 0.90$), and yields real-time uncertainty vectors.
2. **Asymmetric Meta-Router Engine (XGBoost):** Evaluates the multi-parameter uncertainty vectors under cost-sensitive scale weights to decide optimal operational execution tracks.
3. **Dynamic Routing Split:**
   * **Intermediate Routing Pathway:** Routes standard failures through the standard network graph down to the intermediate stage (Model B - ResNet-32).
   * **Hardware Skip Pathway:** Actively executes an aggressive **Hardware Skip** directly to the late-stage Heavy Expert Fallback (Model C - ResNet-56), completely bypassing intermediate latency blocks entirely when boundary decisions are highly ambiguous.

---

## Getting Started

### Dependencies
Ensure you have the following frameworks installed in your Python environment:
```bash
pip install xgboost scikit-learn pandas numpy torch torchvision matplotlib
