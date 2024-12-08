# model_utils.py
import torch
import torch.nn as nn
import numpy as np
import time

def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def count_parameters(model):
    total_params = 0
    nonzero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                total_params += module.weight.numel()
                nonzero_params += torch.count_nonzero(module.weight).item()
            
            if hasattr(module, 'bias') and module.bias is not None:
                total_params += module.bias.numel()
                nonzero_params += torch.count_nonzero(module.bias).item()
    
    return total_params, nonzero_params

def measure_inference_time(model, testloader, device, num_runs=100):
    model.eval()
    times = []
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            
            for _ in range(10):
                _ = model(inputs)
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            break
    
    return np.mean(times), np.std(times)