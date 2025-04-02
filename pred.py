
import torch
from PIL import Image
from torchvision import transforms
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath



model = torch.hub.load(r"ultralytics/yolov5", 'custom', path=r'./models/best.pt')

img_path = r"ACNE.jpg"
image = Image.open(img_path)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model's expected input size
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])
im = transform(image).unsqueeze(0)  # Add batch dimension (BCHW)

try:   
    output = model(im)
    print(output)
except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        

# Get predictions
with torch.no_grad():
    output = model(im)  # Raw model output (logits)

# Apply softmax to get confidence scores
softmax = torch.nn.Softmax(dim=1)
probs = softmax(output)

# Get the predicted class and its confidence score
predicted_class_id = torch.argmax(probs, dim=1).item()
confidence_score = probs[0, predicted_class_id].item()

# Print predicted class and confidence score
print(f"Predicted Class ID: {predicted_class_id}")
print(f"Confidence Score: {confidence_score:.4f}")

# Print predicted class name if available
if hasattr(model, 'names'):
    class_name = model.names[predicted_class_id]
    print(f"Predicted Class Name: {class_name}")

# import torch
# import cv2  # Import OpenCV
# from torchvision import transforms
# import pathlib

# # Pathlib adjustment for Windows compatibility
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # Load pre-trained YOLOv5 model
# model = torch.hub.load(
#     r'C:\Users\RESHMA R B\OneDrive\Documents\Desktop\project_without_malayalam\chatbot2\yolov5',
#     'custom',
#     path=r"C:\Users\RESHMA R B\OneDrive\Documents\Desktop\project_without_malayalam\chatbot2\models\best.pt",
#     source="local"
# )

# # Set model to evaluation mode
# model.eval()

# # Define image transformations (for PyTorch)
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
# ])

# # Load and preprocess the image using OpenCV
# img_path = r"C:\Users\RESHMA R B\OneDrive\Documents\Desktop\project_without_malayalam\chatbot2\ACNE.jpg"
# image = cv2.imread(img_path)  # Load image in BGR format
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# image_resized = cv2.resize(image, (224, 224))  # Resize to match model's expected input size

# # Transform the image for the model
# im = transform(image_resized).unsqueeze(0)  # Add batch dimension (BCHW)

# # Get predictions
# with torch.no_grad():
#     output = model(im)  # Raw model output (logits)

# # Apply softmax to get confidence scores
# softmax = torch.nn.Softmax(dim=1)
# probs = softmax(output)

# # Get the predicted class and its confidence score
# predicted_class_id = torch.argmax(probs, dim=1).item()
# confidence_score = probs[0, predicted_class_id].item()

# # Print predicted class and confidence score
# print(f"Predicted Class ID: {predicted_class_id}")
# print(f"Confidence Score: {confidence_score:.4f}")

# # Print predicted class name if available
# if hasattr(model, 'names'):
#     class_name = model.names[predicted_class_id]
#     print(f"Predicted Class Name: {class_name}")


# cv2.imshow("Input Image", image)  
# cv2.waitKey(0)  
# cv2.destroyAllWindows()  
