# pruning_methods.py
import torch
import torch.nn as nn
from torch.nn.utils import prune
import numpy as np

def count_zero_params(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    return total_params, zero_params

def apply_structured_pruning(model, amount=0.5):
    print("Applying structured pruning...")
    
    total_before, zero_before = count_zero_params(model)
    print(f"Before pruning - Total: {total_before:,}, Zeros: {zero_before:,}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"\nPruning {name}...")
            weights = module.weight.data.cpu().numpy()
            out_channels = weights.shape[0]
            n_to_prune = int(out_channels * amount)
            
            norm_per_channel = np.linalg.norm(weights.reshape(out_channels, -1), axis=1)
            channels_to_prune = np.argsort(norm_per_channel)[:n_to_prune]
            
            mask = torch.ones_like(module.weight.data)
            mask[channels_to_prune] = 0
            module.weight.data *= mask
            
            print(f"Pruned {n_to_prune}/{out_channels} channels")
    
    total_after, zero_after = count_zero_params(model)
    sparsity = (zero_after / total_after * 100) if total_after > 0 else 0
    print(f"\nAfter pruning - Total: {total_after:,}, Zeros: {zero_after:,}")
    print(f"Overall sparsity: {sparsity:.2f}%")
    
    return model

def apply_unstructured_pruning(model, amount=0.5):
    print("Applying unstructured pruning...")
    
    total_before, zero_before = count_zero_params(model)
    print(f"Before pruning - Total: {total_before:,}, Zeros: {zero_before:,}")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"\nPruning {name}...")
            weights = module.weight.data
            
            threshold = torch.kthvalue(torch.abs(weights.data.view(-1)), 
                                    int(weights.numel() * amount))[0]
            
            mask = torch.abs(weights.data) > threshold
            module.weight.data *= mask
            
            zero_count = (module.weight.data == 0).sum().item()
            total_count = module.weight.data.numel()
            print(f"Pruned {zero_count}/{total_count} weights")
    
    total_after, zero_after = count_zero_params(model)
    sparsity = (zero_after / total_after * 100) if total_after > 0 else 0
    print(f"\nAfter pruning - Total: {total_after:,}, Zeros: {zero_after:,}")
    print(f"Overall sparsity: {sparsity:.2f}%")
    
    return model