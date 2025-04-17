# download_artifacts.py
from docling.utils.download import download_artifacts
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download Docling artifacts')
    parser.add_argument('--path', type=str, required=True, 
                       help='Path to store artifacts')
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    os.makedirs(args.path, exist_ok=True)
    
    print(f"Downloading artifacts to: {args.path}")
    download_artifacts(destination=args.path)
    print("Download completed successfully")

if __name__ == "__main__":
    main()