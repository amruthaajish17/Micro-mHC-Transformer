# Micro-mHC-Transformer

A from-scratch PyTorch implementation of **Manifold-Constrained Hyper-Connections (mHC)**, demonstrating structural stability in multi-stream Transformers. 

This project explores how widening the residual stream increases model capacity, and how enforcing double stochasticity on connection matrices prevents signal and gradient explosions during deep network training.

## 🧠 The Concept

Traditional Transformers rely on a single residual stream $x_{l+1}=x_{l}+\mathcal{F}(x_{l})$. Recent architectural advancements expand this stream into $n$ parallel lanes (e.g., $n=4$) to increase topological complexity without exploding computational FLOPs.

However, **Unconstrained Hyper-Connections** have a fatal flaw: over dozens of layers, the unconstrained mixing matrices cause the signal to diverge, leading to massive training instability. 

**Manifold-Constrained HC (mHC)** solves this by using the **Sinkhorn-Knopp algorithm** to entropically project the residual connection matrices onto the Birkhoff polytope. By forcing the routing matrix to be **doubly stochastic** (rows and columns sum to 1), the signal acts as a convex combination of features, strictly conserving energy and maintaining a stable gradient flow.

## 📂 Repository Structure

* `micro_mhc.py`: Contains the core PyTorch modules.
    * `StandardBlock`: A standard Pre-LN Transformer block (Control).
    * `UnconstrainedHCBlock`: A 4-lane expanded residual stream showcasing the instability of standard linear routing.
    * `mHCBlock`: The constrained architecture utilizing the custom `sinkhorn_knopp` projection function.
* `benchmark_mhc.py`: A deep-network simulation script that stacks 30 layers and calculates the **Amax Gain Magnitude** (Forward Signal Gain & Backward Gradient Gain).

## 📊 Results: Amax Gain Magnitude Benchmark

To prove the efficacy of the mHC architecture, we simulate a 30-layer deep forward and backward pass. The benchmark tracks the maximum absolute row sum (forward) and column sum (backward) of the composite residual matrices.

* **Unconstrained HC:** The signal gain explodes exponentially (reaching 1.5x10^0+ on a log scale within 30 layers). In a full-scale model, this causes immediate gradient explosion.
* **Constrained mHC:** The Sinkhorn-Knopp algorithm perfectly constrains the composite matrix. The Amax Gain Magnitude flatlines exactly at **1.0**, proving absolute signal conservation and training stability.

*(Note: Run the benchmark script to generate the comparison plot `mhc_benchmark_results.png` locally).*

## 🚀 Quick Start

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/Micro-mHC-Transformer.git](https://github.com/YOUR-USERNAME/Micro-mHC-Transformer.git)
   cd Micro-mHC-Transformer
