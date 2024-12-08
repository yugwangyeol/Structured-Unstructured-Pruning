# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torchvision.models import resnet18

from data_loader import load_cifar10
from model_utils import train_model, evaluate_model, count_parameters, measure_inference_time
from pruning_methods import apply_structured_pruning, apply_unstructured_pruning
from config import Config

def main():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = load_cifar10(Config.BATCH_SIZE, Config.NUM_WORKERS)

    base_model = resnet18(pretrained=True)
    base_model.fc = nn.Linear(512, 10)
    structured_model = copy.deepcopy(base_model)
    unstructured_model = copy.deepcopy(base_model)
    
    models = {
        'Base': base_model,
        'Structured': structured_model,
        'Unstructured': unstructured_model
    }
    
    for model in models.values():
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizers = {
        name: optim.SGD(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
        for name, model in models.items()
    }

    print("Training all models initially...")
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        train_model(model, trainloader, criterion, optimizers[name], device, Config.EPOCHS)

    print("\nApplying pruning...")
    models['Structured'] = apply_structured_pruning(models['Structured'], Config.PRUNING_AMOUNT)
    models['Unstructured'] = apply_unstructured_pruning(models['Unstructured'], Config.PRUNING_AMOUNT)

    print("\nRetraining pruned models...")
    for name, model in models.items():
        if name != 'Base':
            print(f"\nRetraining {name} pruned model...")
            train_model(model, trainloader, criterion, optimizers[name], device, Config.EPOCHS)

    results = {name: {} for name in models.keys()}

    print("\n=== Results ===")
    print("\nAccuracy:")
    for name, model in models.items():
        acc = evaluate_model(model, testloader, device)
        results[name]['accuracy'] = acc
        print(f"{name} Model: {acc:.2f}%")

    print("\nParameters:")
    for name, model in models.items():
        total_params, zero_params = count_parameters(model)
        non_zero_params = total_params - zero_params
        sparsity = (zero_params / total_params) * 100
        results[name]['params'] = {
            'total': total_params,
            'non_zero': non_zero_params,
            'sparsity': sparsity
        }
        print(f"{name} Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {non_zero_params:,}")
        print(f"  Sparsity: {sparsity:.1f}%")

    print("\nInference Time (ms):")
    for name, model in models.items():
        mean_time, std_time = measure_inference_time(model, testloader, device, Config.INFERENCE_RUNS)
        results[name]['inference'] = {
            'mean': mean_time,
            'std': std_time
        }
        print(f"{name} Model: {mean_time*1000:.2f} ± {std_time*1000:.2f}")
        if name != 'Base':
            speedup = 100 * (results['Base']['inference']['mean'] - mean_time) / results['Base']['inference']['mean']
            print(f"  Speed improvement: {speedup:.1f}%")

if __name__ == "__main__":
    main()