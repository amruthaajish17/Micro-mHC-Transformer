import torch
from micro_mhc import StandardBlock, UnconstrainedHCBlock, mHCBlock

def profile_theoretical_memory(model_name, n, C, seq_len, batch_size, is_expanded=False):
    """
    Calculates the theoretical memory elements required for the residual stream 
    maintenance per batch, tracking the 'Memory Wall' scaling factor.
    """
    total_tokens = batch_size * seq_len
    
    if not is_expanded:
        # Standard Residual: Reads 2C, Writes C elements per token
        read_elements = 2 * C * total_tokens
        write_elements = C * total_tokens
    else:
        # Hyper-Connections (HC/mHC) total I/O scales proportional to n
        # Read elements: (5n + 1)C + n^2 + 2n
        # Write elements: (3n + 1)C + n^2 + 2n
        read_elements = ((5 * n + 1) * C + (n**2) + (2 * n)) * total_tokens
        write_elements = ((3 * n + 1) * C + (n**2) + (2 * n)) * total_tokens
        
    total_io = read_elements + write_elements
    # Assuming float32 (4 bytes per element)
    memory_mb = (total_io * 4) / (1024 ** 2)
    
    print(f"=== {model_name} ===")
    print(f"Theoretical Stream Elements: {total_io:,}")
    print(f"Estimated I/O Data Footprint: {memory_mb:.4f} MB\n")

if __name__ == "__main__":
    # Hyperparameters from your setup
    C = 256
    n = 4
    batch_size = 16
    seq_len = 128
    
    print("--- Theoretical Memory & I/O Profiling ---\n")
    
    profile_theoretical_memory("Standard Block Baseline", n, C, seq_len, batch_size, is_expanded=False)
    profile_theoretical_memory("Expanded Hyper-Connections (HC/mHC)", n, C, seq_len, batch_size, is_expanded=True)