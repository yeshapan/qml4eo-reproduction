import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_eurosat_dataloaders(
    data_dir: str = "./data", # Path to store the raw/processed dataset
    batch_size: int = 32,   # Number of images per batch
    img_size: int = 64,     # Target size for resizing images
    val_split: float = 0.2, # Fraction of data to use for validation
    seed: int = 42,         # Random seed for reproducible splits
    num_workers: int = 2,   # Dynamically accept num_workers from notebook
    pin_memory: bool = True # Dynamically accept pin_memory from notebook
):
    
    # Ensure the data directory exists exactly as defined in the repo structure
    os.makedirs(data_dir, exist_ok=True)
    
    # Added spatial augmentation (flip + rotate images) to the training pipeline
    # This helps prevent overfitting and model learns concepts regardless of orientation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),      
        transforms.RandomVerticalFlip(p=0.5),        
        transforms.RandomRotation(degrees=15),       
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # NOTE: Validation data MUST NOT be augmented. 
    # We need a static + consistent ground truth to accurately evaluate model performance
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Downloading + Loading EuroSAT dataset into {data_dir}..")

    # Load two separate instances of the dataset so we can apply the different transformation pipelines for train and
    full_train_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=train_transform)
    full_val_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=val_transform)
    
    class_names = full_train_dataset.classes
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Use a deterministic seed to split the dataset
    '''
    We set the seed to '42' to establish a deterministic foundation → 
    Guarantee the exact same images go into train_subset and val_subset →
    Prevent data leak between train and val + ensure fair comparison across runs for all experiments
    '''
    generator = torch.Generator().manual_seed(seed)
    train_subset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(full_val_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    print(f"Dataset loaded successfully! Total: {total_size} | Train: {train_size} | Val: {val_size}")
    
    return train_loader, val_loader, class_names