# config.py
class Config:
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    
    EPOCHS = 20
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    PRUNING_AMOUNT = 0.5
    
    INFERENCE_RUNS = 100