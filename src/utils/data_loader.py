import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_eurosat_dataloaders(
    data_dir: str = "./data", 
    batch_size: int = 32,
    img_size: int = 64, 
    val_split: float = 0.2,
    seed: int = 42
):
    """
    Download and prepare the EuroSAT dataset for both classical and hybrid quantum-classical training:
    
    Args:
        data_dir (str): Path to store the raw/processed dataset
        batch_size (int): Number of images per batch
        img_size (int): Target size for resizing images 
                        (Note: QML models often require aggressive downscaling, e.g., 16 or 32)
        val_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducible splits
        
    Returns:
        train_loader (DataLoader), val_loader (DataLoader), class_names (list)
    """
    
    #ensure the data directory exists exactly as defined in the repo structure
    os.makedirs(data_dir, exist_ok=True)
    
    #define standard transformations
    #we'll resize, convert to PyTorch tensors and normalize pixel values [0, 1] to roughly [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Downloading/Loading EuroSAT dataset into {data_dir}...")
    
    #load the dataset (torchvision handles the download automatically if not present)
    eurosat_dataset = datasets.EuroSAT(
        root=data_dir, 
        download=True, 
        transform=transform
    )
    
    class_names = eurosat_dataset.classes
    total_size = len(eurosat_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create reproducible train/val splits
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        eurosat_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    #initialize DataLoaders
    #num_workers=2 is generally safe for Colab environments
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Total images: {total_size} | Training: {train_size} | Validation: {val_size}")
    print(f"Classes: {class_names}")
    
    return train_loader, val_loader, class_names

#quick test block to ensure it works when run directly
if __name__ == "__main__":
    train_loader, val_loader, classes = get_eurosat_dataloaders(data_dir="../../data")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape} (Batch Size, Channels, Height, Width)")