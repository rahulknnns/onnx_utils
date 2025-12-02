import numpy as np
import os

def generate_data():
    # 1. Create directories
    cpu_dir = "cpu_tensors"
    hwa_dir = "hwa_tensors"
    
    os.makedirs(cpu_dir, exist_ok=True)
    os.makedirs(hwa_dir, exist_ok=True)

    print(f"Generating dummy files in '{cpu_dir}' and '{hwa_dir}'...")

    # 2. Generate 10 sample tensors
    for i in range(10):
        filename = f"layer_{i}_output.npy"
        
        # Create random data (e.g., shape 100x100)
        # Using float32 which is standard for DL models
        shape = (1000,) 
        
        # CPU Data: Standard Normal Distribution
        cpu_data = np.random.randn(*shape).astype(np.float32) * (i + 1)
        
        # HWA Data: Add some simulated quantization noise
        # (e.g., rounding or slight drift)
        noise = np.random.normal(0, 0.2, size=shape).astype(np.float32)
        hwa_data = cpu_data + noise
        
        # Occasionally clip HWA data to simulate saturation (makes histograms interesting)
        if i % 3 == 0:
            hwa_data = np.clip(hwa_data, -2.0, 2.0)

        # 3. Save files
        np.save(os.path.join(cpu_dir, filename), cpu_data)
        np.save(os.path.join(hwa_dir, filename), hwa_data)
        
        print(f"Created {filename}")

    print("\nDone! You can now run the streamlit app.")

if __name__ == "__main__":
    generate_data()