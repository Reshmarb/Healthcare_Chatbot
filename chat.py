import os
import sys
import subprocess


def predict_image(image_path):
    # Define the command to run the YOLOv5 model prediction
    command = [
        "python", "yolov5\classify\predict.py", 
        "--weights", "models\best.pt", 
        "--source", image_path
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Prediction completed for the image: {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")

def main():
    # Take input image path from the user
    image_path ="ACNE.jpg"

    if os.path.exists(image_path):
        predict_image(image_path)
    else:
        print(f"Error: The file {image_path} does not exist.")
        
if __name__ == "__main__":
    main()
