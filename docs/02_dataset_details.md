### **Dataset Specifications: EuroSAT**

This project utilizes the **EuroSAT** dataset (a standard benchmark in Earth Observation (EO) research) to compare classical and quantum ML models.

#### **Overview**
EuroSAT is based on Sentinel-2 satellite images covering 13 spectral bands. For the scope of this reproduction (which focuses on quantum circuit trainability rather than multispectral analysis); we utilize the **RGB version** of the dataset.

* **Task:** Land Cover Classification
* **Total Images:** 27,000
* **Classes:** 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
* **Original Resolution:** 64x64 pixels

#### **Preprocessing Pipeline**
The data loading utility (`src/utils/data_loader.py`) applies following transformations:
1.  **Resizing:** While classical models process the native 64x64 resolution, quantum models require aggressive downscaling (e.g., 16x16 or 32x32) to fit within current NISQ qubit constraints.
2.  **Normalization:** Pixel values are converted to PyTorch tensors and normalized using standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` to ensure stable gradient descent.
3.  **Splitting:** A deterministic random seed (Seed: 42) is used to create an 80/20 train-validation split (21,600 training images, 5,400 validation images).