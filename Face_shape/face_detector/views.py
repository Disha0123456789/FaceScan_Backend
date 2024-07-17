from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import dlib
import numpy as np
import os
import json
import random

# Define the path to the models and JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
shape_predictor_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
haarcascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
shapes_json_path = os.path.join(BASE_DIR, 'shapes.json')

# Load models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(shape_predictor_path)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('imagefile', None)
        if not image_file:
            return JsonResponse({'error': 'No image uploaded'}, status=400)

        # Save the uploaded image temporarily
        temp_image_path = 'temp_image.jpg'
        with open(temp_image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Read the image with OpenCV
        image = cv2.imread(temp_image_path)

        # Detect faces and landmarks
        faces, landmarks_list = detect_faces_landmarks(image)
        if not landmarks_list:
            return JsonResponse({'error': 'No face detected'}, status=400)

        # Calculate face shape for the first detected face
        landmarks = landmarks_list[0]
        face_shape = calculate_face_shape(landmarks, image)

        # Get predictions from shapes.json
        predictions = get_predictions(face_shape)
        if predictions:
            return JsonResponse(predictions)
        else:
            return JsonResponse({'error': 'Face shape not found in shapes.json'}, status=404)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def detect_faces_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    landmarks_list = []
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks_list.append([(p.x, p.y) for p in landmarks.parts()])
    return faces, landmarks_list

def calculate_face_shape(landmarks, image):
    jawline_points = np.array(landmarks[4:13])
    forehead_width = np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[26]))
    jawline_width = np.linalg.norm(jawline_points[0] - jawline_points[-1])
    cheekbones_width = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[14]))
    face_height = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[25]))

    standardized_height = 100.0
    scale_factor = standardized_height / face_height
    standardized_forehead_width = forehead_width * scale_factor
    standardized_jawline_width = jawline_width * scale_factor
    standardized_cheekbones_width = cheekbones_width * scale_factor

    if standardized_cheekbones_width > standardized_forehead_width + (20 * scale_factor) and standardized_forehead_width > standardized_jawline_width + (15 * scale_factor):
        return "Heart"
    elif abs(standardized_forehead_width - standardized_cheekbones_width) <= (20 * scale_factor) and abs(standardized_forehead_width - standardized_jawline_width) <= (20 * scale_factor) and abs(standardized_cheekbones_width - standardized_jawline_width) <= (20 * scale_factor) and standardized_height > standardized_cheekbones_width + (17 * scale_factor):
        return "Oblong"
    elif abs(standardized_forehead_width - standardized_jawline_width) <= (30 * scale_factor) and abs(standardized_cheekbones_width - standardized_jawline_width) <= (37 * scale_factor):
        return "Square"
    elif standardized_cheekbones_width - max(standardized_forehead_width, standardized_jawline_width) > (25 * scale_factor) and standardized_height > standardized_cheekbones_width + (20 * scale_factor):
        return "Oval"
    elif abs(standardized_forehead_width - standardized_jawline_width) <= (30 * scale_factor) and standardized_cheekbones_width - max(standardized_forehead_width, standardized_jawline_width) > (20 * scale_factor):
        return "Round"
    else:
        return "Unknown"

def get_predictions(face_shape):
    with open(shapes_json_path) as f:
        shapes_data = json.load(f)

    for shape_entry in shapes_data:
        if shape_entry['face_shape'] == face_shape:
            selected_predictions = {}
            # Randomly select prediction type (prediction1, prediction2, prediction3)
            prediction_type = random.choice(list(shape_entry['personal_traits'].keys()))
            for category, predictions in shape_entry.items():
                if category != 'face_shape':
                    selected_predictions[category] = predictions[prediction_type]
            return selected_predictions
    return None
