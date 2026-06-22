import torch
import matplotlib.pyplot as plt
from micro_mhc import UnconstrainedHCBlock, mHCBlock

def calculate_composite_gains(model_class, num_layers=30, d_model=256, n=4):
    """
    Simulates a deep network to calculate the composite Amax Gain Magnitude.
    """
    # Create dummy input: 1 Batch, 1 Token (for simplicity), n lanes, C dimension
    x = torch.randn(1, 1, n, d_model)
    
    forward_gains = []
    backward_gains = []
    
    # Initialize the composite matrix as an Identity matrix [n, n]
    composite_H = torch.eye(n)
    
    # Simulate passing through num_layers
    for layer_idx in range(num_layers):
        layer = model_class(d_model=d_model, n=n)
        
        # Forward pass to get the H_res matrix for this layer
        _, H_res = layer(x)
        
        # Extract the matrix for our single batch/token: Shape [n, n]
        H_res_matrix = H_res[0, 0].detach()
        
        # Update the composite mapping (multiplying the matrices together)
        # We take the absolute value to track maximum theoretical magnitude
        composite_H = torch.matmul(torch.abs(H_res_matrix), composite_H)
        
        # Calculate Amax Gain Magnitudes
        max_row_sum = composite_H.sum(dim=1).max().item() # Forward Signal Gain
        max_col_sum = composite_H.sum(dim=0).max().item() # Backward Gradient Gain
        
        forward_gains.append(max_row_sum)
        backward_gains.append(max_col_sum)
        
    return forward_gains, backward_gains

if __name__ == "__main__":
    print("Running deep network simulation (30 layers)...")
    
    # Get gains for both models
    hc_fwd, hc_bwd = calculate_composite_gains(UnconstrainedHCBlock)
    mhc_fwd, mhc_bwd = calculate_composite_gains(mHCBlock)
    
    # --- Plotting the Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Unconstrained HC (Notice the Y-axis is logarithmic!)
    ax1.plot(hc_fwd, label="Forward Signal Gain", color="blue")
    ax1.plot(hc_bwd, label="Backward Gradient Gain", color="grey")
    ax1.set_title("Unconstrained HC: Composite Gain (Log Scale)")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Amax Gain Magnitude")
    ax1.set_yscale("log") # Log scale because it explodes!
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Constrained mHC
    ax2.plot(mhc_fwd, label="Forward Signal Gain", color="blue")
    ax2.plot(mhc_bwd, label="Backward Gradient Gain", color="grey")
    ax2.set_title("mHC: Composite Gain (Linear Scale)")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Amax Gain Magnitude")
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label="Target (1.0)")
    ax2.set_ylim(0.5, 2.0) # Tightly bounded near 1.0
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("mhc_benchmark_results.png")
    print("Done! Open 'mhc_benchmark_results.png' to see the difference.")