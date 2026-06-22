import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from micro_mhc import StandardBlock, UnconstrainedHCBlock, mHCBlock

class DeepNetwork(nn.Module):
    """Stacks our blocks to create a deep, stress-testable network."""
    def __init__(self, block_class, d_model=256, n=4, num_layers=24):
        super().__init__()
        
        # Only pass 'n' if the block is an HC architecture
        if block_class == StandardBlock:
            self.layers = nn.ModuleList([block_class(d_model=d_model) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([block_class(d_model=d_model, n=n) for _ in range(num_layers)])
            
        self.output_layer = nn.Linear(d_model, 10) # Dummy 10-class output

    def forward(self, x):
        for layer in self.layers:
            # HC blocks return a tuple (output, H_res matrix), standard blocks just return output
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
            
        # If it's an expanded HC block, mean-pool across the 'n' lanes to get back to C dimension
        if x.dim() == 4: 
            x = x.mean(dim=2) 
            
        # Mean-pool over the sequence length and pass to the classifier
        return self.output_layer(x.mean(dim=1)) 

def run_stress_test(model_name, block_class, iterations=100, lr=0.05):
    """Trains the model on dummy data and returns the loss history."""
    print(f"Training {model_name}...")
    torch.manual_seed(42)
    
    model = DeepNetwork(block_class)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Generate dummy data
    n = 4 if block_class != StandardBlock else 1
    x = torch.randn(8, 32, n, 256) if n > 1 else torch.randn(8, 32, 256)
    targets = torch.randint(0, 10, (8,))
    
    loss_history = []
    
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        # We clip gradients to prevent immediate "NaN" math crashes, 
        # allowing us to actually plot the explosive spikes in the HC model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
    return loss_history

if __name__ == "__main__":
    print("--- 24-Layer Training Stress Test ---\n")
    
    # 1. Run the training loops for all three architectures
    baseline_loss = run_stress_test("Standard Baseline", StandardBlock)
    hc_loss = run_stress_test("Unconstrained HC", UnconstrainedHCBlock)
    mhc_loss = run_stress_test("Constrained mHC", mHCBlock)
    
    # 2. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_loss, label="Standard Baseline", alpha=0.8)
    plt.plot(hc_loss, label="Unconstrained HC", color="red", alpha=0.8)
    plt.plot(mhc_loss, label="Constrained mHC", color="green", linewidth=2)
    
    plt.title("24-Layer Stress Test: Training Loss Dynamics")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.yscale("log") # Log scale helps visualize the divergence
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("stress_test_results.png")
    print("\nDone! Check 'stress_test_results.png' to see the architecture breakdown.")