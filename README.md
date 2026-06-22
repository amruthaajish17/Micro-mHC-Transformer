# Micro-mHC-Transformer

A from-scratch PyTorch implementation of **Manifold-Constrained Hyper-Connections (mHC)**, demonstrating structural stability in multi-stream Transformers. 

This project explores how widening the residual stream increases model capacity, and how enforcing double stochasticity on connection matrices prevents signal and gradient explosions during deep network training. Built with technical accuracy in mind, this foundation is actively designed to be extensible into advanced visual architectures like **YOLO Nano** for lightweight object detection and securely integrated with **Microsoft Presidio** for privacy-preserving AI pipelines.

## 🧠 The Concept

Traditional Transformers rely on a single residual stream ($x_{l+1}=x_{l}+\mathcal{F}(x_{l})$). Recent architectural advancements expand this stream into $n$ parallel lanes (e.g., $n=4$) to increase topological complexity without exploding computational FLOPs.

However, **Unconstrained Hyper-Connections** have a fatal flaw: over dozens of layers, the unconstrained mixing matrices cause the signal to diverge, leading to massive training instability. 

**Manifold-Constrained HC (mHC)** solves this by using the **Sinkhorn-Knopp algorithm** to entropically project the residual connection matrices onto the Birkhoff polytope. By forcing the routing matrix to be **doubly stochastic** (rows and columns sum to 1), the signal acts as a convex combination of features, strictly conserving energy and maintaining a stable gradient flow.

## 📂 Repository Structure

* `micro_mhc.py`: Contains the core PyTorch modules.
    * `StandardBlock`: A standard Pre-LN Transformer block (Control).
    * `UnconstrainedHCBlock`: A 4-lane expanded residual stream showcasing the instability of standard linear routing.
    * `mHCBlock`: The constrained architecture utilizing the custom `sinkhorn_knopp` projection function.
* `benchmark_mhc.py`: A deep-network simulation script that stacks 30 layers and calculates the **Amax Gain Magnitude** (Forward Signal Gain & Backward Gradient Gain).
* `memory_profiler.py`: Analyzes the "Memory Wall" by calculating the theoretical I/O elements and data footprint scaling for the expanded multi-lane stream.
* `stress_test.py`: A deep-network training loop (24 layers) on dummy data to visualize training loss dynamics and gradient explosions under aggressive learning rates.

## 📊 Results

### 1. Amax Gain Magnitude Benchmark
To prove the efficacy of the mHC architecture, we simulate a 30-layer deep forward and backward pass. The benchmark tracks the maximum absolute row sum (forward) and column sum (backward) of the composite residual matrices.
* **Unconstrained HC:** The signal gain explodes exponentially (reaching $1.5 \times 10^{10}$ on a log scale within 30 layers). In a full-scale model, this causes immediate gradient explosion.
* **Constrained mHC:** The Sinkhorn-Knopp algorithm perfectly constrains the composite matrix. The Amax Gain Magnitude flatlines exactly at **1.0**, proving absolute signal conservation.

*(Note: Run `benchmark_mhc.py` to generate the comparison plot `mhc_benchmark_results.png`).*

### 2. 24-Layer Training Stress Test
By stacking the blocks 24 layers deep and applying an aggressive learning rate, we force the architectures to their breaking point to observe empirical loss dynamics:
* **Standard Baseline:** Experiences early gradient collapse and fails to maintain stable optimization.
* **Unconstrained HC:** Loss diverges and oscillates wildly without ever converging, completely locked out by exploding signals across unconstrained paths.
* **Constrained mHC:** By projecting the residual connections directly onto the Birkhoff polytope, the network flawlessly conserves the signal mean and regularizes norms. It handles the aggressive learning rate effortlessly, plunging smoothly into deep convergence.

*(Note: Run `stress_test.py` to generate the training dynamics plot `stress_test_results.png`).*

## 🚀 Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Micro-mHC-Transformer.git
   cd Micro-mHC-Transformer
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install torch matplotlib
   ```

3. Run the core layer tests (verifies tensor shapes and the Birkhoff polytope projection):
   ```bash
   python micro_mhc.py
   ```

4. Run the deep-layer simulation to generate the Amax stability plots:
   ```bash
   python benchmark_mhc.py
   ```

5. Run the theoretical memory profiler to observe the hardware I/O scaling:
   ```bash
   python memory_profiler.py
   ```

6. Run the full 24-layer training stress test to visualize loss convergence:
   ```bash
   python stress_test.py
   ```
