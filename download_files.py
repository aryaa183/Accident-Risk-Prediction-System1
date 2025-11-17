import os
import gdown

# Dictionary of files and their Google Drive file IDs
files = {
    "accident_model_optimized.pkl": "1NFOHMDAPFWL3FM7nhZOvk7TnMBXlixln",
    "risk_model_classifier.pkl": "1hX-W7VXCgVD0VewUiFMgEM0Q2nUM2iUh",
    "risk_model_advanced.pkl": "1e09Hw0txQer4TULVJZqXOkjiui7a1s68"
}

for filename, file_id in files.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists locally.")
