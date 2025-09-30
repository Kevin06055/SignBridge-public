"""
Start the Sign Language Detection API server
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the Sign Language Detection API server')
    parser.add_argument('--port', type=int, default=5000, help='Port for API server')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model file')
    args = parser.parse_args()
    
    # Run the API server with the provided arguments
    os.system(f"python realtime_inference.py --api --port {args.port} --model {args.model}")

if __name__ == "__main__":
    main()