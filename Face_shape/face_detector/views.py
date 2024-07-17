from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import dlib
import numpy as np
import os
import json
import logging
import random  # Add this import

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define the path to the models and JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
shape_predictor_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
haarcascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
shapes_json_path = os.path.join(BASE_DIR, 'shapes.json')

# Load models
face_detector = cv2.CascadeClassifier(haarcascade_path)
landmark_predictor = dlib.shape_predictor(shape_predictor_path)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        logger.info("Received a POST request.")
        image_file = request.FILES.get('imagefile', None)
        if not image_file:
            logger.error('No image uploaded.')
            return JsonResponse({'error': 'No image uploaded'}, status=400)

        # Check file format
        if not image_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.error('Unsupported image format.')
            return JsonResponse({'error': 'Unsupported image format'}, status=400)

        # Save the uploaded image temporarily
        temp_image_path = 'temp_image.jpg'
        with open(temp_image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Read the image with OpenCV
        image = cv2.imread(temp_image_path)
        if image is None:
            logger.error('Failed to read image.')
            os.remove(temp_image_path)
            return JsonResponse({'error': 'Failed to read image'}, status=400)

        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Detect faces and landmarks
        try:
            faces, landmarks_list = detect_faces_landmarks(image)
            if not landmarks_list:
                logger.error('No face detected.')
                os.remove(temp_image_path)
                return JsonResponse({'error': 'No face detected'}, status=400)
        except RuntimeError as e:
            logger.error(f"Error in detecting faces/landmarks: {e}")
            os.remove(temp_image_path)
            return JsonResponse({'error': str(e)}, status=400)

        # Calculate face shape for the first detected face
        landmarks = landmarks_list[0]
        face_shape = calculate_face_shape(landmarks, image)

        # Get predictions from shapes.json
        predictions = get_predictions(face_shape)
        os.remove(temp_image_path)  # Clean up temp file

        if predictions:
            return JsonResponse(predictions)
        else:
            logger.error('Face shape not found in shapes.json.')
            return JsonResponse({'error': 'Face shape not found in shapes.json'}, status=404)

    logger.error('Invalid request method.')
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def detect_faces_landmarks(image):
    if image is None:
        logger.error("Image is None, possibly due to incorrect file path or format.")
        raise RuntimeError("Unsupported image type, must be 8bit gray or RGB image.")

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info(f"Converted image to grayscale: {gray.shape}, dtype: {gray.dtype}")

        # Check if the image is 8-bit grayscale
        if gray.dtype != np.uint8 or len(gray.shape) != 2:
            logger.error("Gray image is not 8-bit or not a single channel.")
            raise RuntimeError("Unsupported image type, must be 8bit gray or RGB image.")

        # Detect faces
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        logger.info(f"Faces detected: {len(faces)}")
        if len(faces) == 0:
            raise RuntimeError("No faces detected")

        landmarks_list = []
        for (x, y, w, h) in faces:
            logger.info(f"Processing face with coordinates: x={x}, y={y}, w={w}, h={h}")
            rect = dlib.rectangle(x, y, x + w, y + h)
            logger.info(f"Created dlib.rectangle: [{rect.left()}, {rect.top()}, {rect.right()}, {rect.bottom()}]")
            landmarks = landmark_predictor(image, rect)
            landmarks_list.append(landmarks)

        return faces, landmarks_list
    except Exception as e:
        logger.error(f"Error in face detection or landmark prediction: {e}")
        raise RuntimeError("Error in face detection or landmark prediction")

def calculate_face_shape(landmarks, image):
    # Add your calculation logic here
    pass

def get_predictions(face_shape):
    with open(shapes_json_path) as f:
        shapes_data = json.load(f)

    # Ensure that shapes_data is a dictionary and has the 'shapes' key
    if not isinstance(shapes_data, dict) or 'shapes' not in shapes_data:
        logger.error("Invalid shapes.json format.")
        return None

    for shape_entry in shapes_data['shapes']:
        if shape_entry['shape'] == face_shape:
            prediction_type = random.choice(list(shape_entry['personal_traits'].keys()))
            return {
                'shape': face_shape,
                'traits': shape_entry['personal_traits'][prediction_type]
            }

    return None

# Don't forget to import your additional necessary modules and define any additional functions
