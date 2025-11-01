from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2 # OpenCV for face detection and drawing
from tensorflow import keras # TensorFlow/Keras for model loading and prediction
import io
from PIL import Image # Pillow for handling image data
import os
import csv
import sys
import time

# --- Configuration ---
# --- ⬇️ RENAME this to your actual model file name ⬇️ ---
EXPRESSION_MODEL_FILENAME = 'facial_expression_model.keras'
# --- ⬆️ RENAME this to your actual model file name ⬆️ ---

HAAR_CASCADE_FILENAME = 'haarcascade_frontalface_default.xml' # Standard OpenCV face detector
CSV_FILENAME = 'expression_pixel_data.csv' # CSV filename for expression pixel data

# --- ⬇️ !!! IMPORTANT: Set the input size (height, width) your Keras model expects !!! ⬇️ ---
TARGET_SIZE = (48, 48) # Example: (48, 48) or (64, 64) etc.
# --- ⬆️ !!! IMPORTANT: Set the input size (height, width) your Keras model expects !!! ⬆️ ---

# --- Emotion Mapping (Verify this matches YOUR model's output classes) ---
# Using lowercase keys now to match backend expectation
EMOTION_MAP_INDEX_TO_LABEL = {
    0: 'angry',
    1: 'disgusted',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprised',
    6: 'neutral'
}
# Also include 'nan' for feedback purposes (lowercase)
VALID_FEEDBACK_LABELS = list(EMOTION_MAP_INDEX_TO_LABEL.values()) + ["nan"] # Use lowercase 'nan' internally

# --- CSV Header Configuration ---
PIXEL_COUNT = TARGET_SIZE[0] * TARGET_SIZE[1]
PIXEL_HEADERS = [f'pixel{i}' for i in range(PIXEL_COUNT)]
CSV_HEADER = ['label'] + PIXEL_HEADERS

# --- File Paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXPRESSION_MODEL_PATH = os.path.join(APP_DIR, EXPRESSION_MODEL_FILENAME)
HAAR_CASCADE_PATH = os.path.join(APP_DIR, HAAR_CASCADE_FILENAME)
ABSOLUTE_CSV_PATH = os.path.join(APP_DIR, CSV_FILENAME)

# --- Load Models ---
# ... (model loading remains the same) ...
if not os.path.exists(EXPRESSION_MODEL_PATH): print(f"Error: Expression Model file not found at {EXPRESSION_MODEL_PATH}", file=sys.stderr); exit()
if not os.path.exists(HAAR_CASCADE_PATH): print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}", file=sys.stderr); exit()
try:
    expression_model = keras.models.load_model(EXPRESSION_MODEL_PATH)
    print(f"Expression Model loaded successfully from {EXPRESSION_MODEL_PATH}")
    try:
        print(f"Model expected input shape: {expression_model.input_shape}")
        if len(expression_model.input_shape) >=3 and expression_model.input_shape[1:3] != TARGET_SIZE: print(f"Warning: Model input shape {expression_model.input_shape[1:3]} might not match configured TARGET_SIZE {TARGET_SIZE}", file=sys.stderr)
    except Exception as shape_err: print(f"Could not determine model input shape automatically: {shape_err}", file=sys.stderr)
except Exception as e: print(f"Error loading expression model: {e}", file=sys.stderr); exit()
try:
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty(): raise IOError("Unable to load the face cascade classifier xml file")
    print(f"Haar Cascade loaded successfully from {HAAR_CASCADE_PATH}")
except Exception as e: print(f"Error loading Haar Cascade: {e}", file=sys.stderr); exit()


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allow requests from your React app

# --- Helper Function: Preprocess Image for Model ---
# ... (preprocess_face_roi remains the same) ...
def preprocess_face_roi(roi_gray, target_size):
    if roi_gray is None or roi_gray.size == 0: raise ValueError("Input ROI is empty")
    roi_resized = cv2.resize(roi_gray, target_size, interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized / 255.0
    roi_reshaped = roi_normalized.reshape(1, target_size[0], target_size[1], 1)
    return roi_reshaped


# --- Health Check Route ---
# ... (health_check remains the same) ...
@app.route('/', methods=['GET'])
def health_check():
    csv_exists = os.path.isfile(ABSOLUTE_CSV_PATH); return jsonify({"status": "healthy", "csv_found": csv_exists}), 200


# --- UPDATED: Expression Prediction Endpoint ---
@app.route('/predict_expression', methods=['POST'])
def predict_expression():
    try:
        data = request.get_json()
        # ... (Input validation remains the same) ...
        if 'image' not in data: return jsonify({'error': 'No image data found in request'}), 400
        image_data = data['image']

        # 1. Decode Base64 to Image
        # ... (Decoding and color conversion remains the same) ...
        try:
            if "," not in image_data: raise ValueError("Invalid base64 header")
            header, encoded = image_data.split(",", 1); binary_data = base64.b64decode(encoded)
            image_pil = Image.open(io.BytesIO(binary_data)); image_np_bgr = np.array(image_pil)
            if len(image_np_bgr.shape) == 3 and image_np_bgr.shape[2] == 4: image_np_bgr = cv2.cvtColor(image_np_bgr, cv2.COLOR_RGBA2BGR)
            elif len(image_np_bgr.shape) == 3 and image_np_bgr.shape[2] == 3: image_np_bgr = cv2.cvtColor(image_np_bgr, cv2.COLOR_RGB2BGR)
            elif len(image_np_bgr.shape) == 2: image_np_bgr = cv2.cvtColor(image_np_bgr, cv2.COLOR_GRAY2BGR)
            elif len(image_np_bgr.shape) != 3 or image_np_bgr.shape[2] != 3:
                 if len(image_np_bgr.shape) == 2: image_np_bgr = cv2.cvtColor(image_np_bgr, cv2.COLOR_GRAY2BGR)
                 else: raise ValueError(f"Unexpected image shape for drawing: {image_np_bgr.shape}")
            image_to_draw_on = image_np_bgr.copy(); gray_image = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e: print(f"Image Decoding/Conversion Error: {e}", file=sys.stderr); return jsonify({'error': f'Invalid or unsupported image data: {str(e)}'}), 400


        # 2. Detect Faces
        # ... (Face detection remains the same) ...
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Detected {len(faces)} faces.")
        results = []
        # Removed is_multiple_faces flag, text drawing is now always attempted if prediction succeeds

        # 3. Process Each Detected Face
        for (x, y, w, h) in faces:
            roi_gray = gray_image[y:y+h, x:x+w]
            predicted_label = "Error"
            confidence_score = 0.0

            if roi_gray.size == 0: print(f"Skipping empty face ROI at {(x, y, w, h)}", file=sys.stderr); continue

            try:
                processed_roi = preprocess_face_roi(roi_gray, TARGET_SIZE)
                prediction_raw = expression_model.predict(processed_roi)
                predicted_index = int(np.argmax(prediction_raw))
                confidence_score = float(np.max(prediction_raw))
                predicted_label = EMOTION_MAP_INDEX_TO_LABEL.get(predicted_index, "Unknown")

            except Exception as e:
                print(f"Error predicting expression for face at {(x, y, w, h)}: {e}", file=sys.stderr)

            # --- ALWAYS Draw Bounding Box ---
            box_color = (0, 255, 0) # Green box
            box_thickness = 2
            cv2.rectangle(image_to_draw_on, (x, y), (x + w, y + h), box_color, box_thickness)

            # --- ALWAYS Draw Text if prediction didn't error ---
            if predicted_label != "Error":
                text_color = (255, 255, 255) # White text
                bg_color = (0, 0, 0) # Black background for text
                font_scale = 0.7 # --- Increased font scale slightly ---
                text_thickness = 1
                padding = 4 # --- Increased padding slightly ---

                label_text = f"{predicted_label.capitalize()}: {confidence_score:.1%}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                text_y_pos = y - 10 if y - 10 > text_height + padding else y + h + text_height + baseline + padding # Add padding below if text is below

                # Calculate background rectangle position
                rect_y_start = text_y_pos - text_height - baseline - padding
                rect_y_end = text_y_pos + baseline + padding
                # Ensure x-coords take padding into account properly
                rect_x_start = x
                rect_x_end = x + text_width + (padding * 2) # Padding on both sides

                # Adjustments for edges
                if rect_y_start < 0:
                    diff = abs(rect_y_start) + padding; rect_y_start += diff; rect_y_end += diff; text_y_pos += diff
                # Ensure start/end don't go out of bounds
                if rect_x_start < 0 : rect_x_start = 0
                if rect_x_end > image_to_draw_on.shape[1]: rect_x_end = image_to_draw_on.shape[1]

                # Draw filled rectangle behind text (using adjusted coords)
                cv2.rectangle(image_to_draw_on, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), bg_color, cv2.FILLED)
                # Draw text slightly padded
                cv2.putText(image_to_draw_on, label_text, (x + padding, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)
            # --- End text drawing ---

            # Store result
            results.append({
                'box': [int(x), int(y), int(w), int(h)],
                'expression': predicted_label,
                'confidence': confidence_score
            })

        # 4. Encode the modified image back to base64
        # ... (Encoding remains the same) ...
        try:
            image_rgb_with_drawings = cv2.cvtColor(image_to_draw_on, cv2.COLOR_BGR2RGB); pil_img = Image.fromarray(image_rgb_with_drawings); buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG"); processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            processed_image_data_url = f"data:image/png;base64,{processed_image_base64}"
        except Exception as e: print(f"Error encoding processed image: {e}", file=sys.stderr); return jsonify({'error': f'Could not encode result image: {str(e)}'}), 500


        # 5. Return JSON Response
        return jsonify({
            'processed_image': processed_image_data_url,
            'faces': results
        })

    except Exception as e:
        print(f"Unexpected error during expression prediction: {e}", file=sys.stderr)
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500


# --- Feedback Saving Endpoint ---
# ... (save_expression_feedback remains the same) ...
@app.route('/save_expression_feedback', methods=['POST'])
def save_expression_feedback():
    try:
        data = request.get_json();
        if 'image' not in data or 'label' not in data: return jsonify({'error': 'Missing image or label data in request'}), 400
        image_base64 = data['image']; label_input = data['label']
        if label_input not in VALID_FEEDBACK_LABELS: print(f"Invalid expression label received: {label_input}", file=sys.stderr); return jsonify({'error': f'Invalid label. Expected one of {VALID_FEEDBACK_LABELS}.'}), 400
        validated_label = label_input
        if not image_base64 or not isinstance(image_base64, str) or not image_base64.startswith('data:image'): print(f"Invalid image data format received for feedback.", file=sys.stderr); return jsonify({'error': 'Invalid image data format.'}), 400
        pixel_values_int = []
        try:
            if "," not in image_base64: raise ValueError("Invalid base64 header")
            header, encoded = image_base64.split(",", 1); binary_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(binary_data)); img_np_color = np.array(image)
            if len(img_np_color.shape) == 3: gray_image_feedback = cv2.cvtColor(img_np_color, cv2.COLOR_BGR2GRAY)
            elif len(img_np_color.shape) == 2: gray_image_feedback = img_np_color
            else: raise ValueError(f"Unexpected image shape for feedback processing: {img_np_color.shape}")
            faces_feedback = face_cascade.detectMultiScale(gray_image_feedback, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces_feedback) > 0:
                (x, y, w, h) = faces_feedback[0]; roi_gray_feedback = gray_image_feedback[y:y+h, x:x+w]
                if roi_gray_feedback.size == 0: print("Warning: Detected face ROI is empty for feedback saving.", file=sys.stderr); pixel_values_int = [""] * PIXEL_COUNT
                else: img_resized_raw = cv2.resize(roi_gray_feedback, TARGET_SIZE, interpolation=cv2.INTER_AREA); pixel_values = img_resized_raw.flatten(); pixel_values_int = [int(p) for p in pixel_values]; print(f"Processed first face ROI for saving ({len(pixel_values_int)} pixels).")
            else: print("Warning: No face detected in the image provided for feedback.", file=sys.stderr); pixel_values_int = [""] * PIXEL_COUNT
        except Exception as e: print(f"Error processing image for feedback CSV: {e}", file=sys.stderr); return jsonify({'error': f'Could not process image data for saving feedback: {str(e)}'}), 500
        print(f"Attempting to write expression CSV to: {ABSOLUTE_CSV_PATH}")
        new_row = [validated_label] + pixel_values_int; file_exists = os.path.isfile(ABSOLUTE_CSV_PATH)
        try:
            with open(ABSOLUTE_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile);
                if not file_exists: writer.writerow(CSV_HEADER); print(f"Created new expression CSV file: {ABSOLUTE_CSV_PATH} with header.")
                writer.writerow(new_row)
        except Exception as e: print(f"Error writing to expression CSV file '{ABSOLUTE_CSV_PATH}': {e}", file=sys.stderr); return jsonify({'error': f'Could not write to expression CSV file: {str(e)}'}), 500
        print(f"Successfully appended expression pixel data for label: {validated_label} to {ABSOLUTE_CSV_PATH}")
        return jsonify({'status': 'success', 'message': 'Feedback saved successfully.'}), 200
    except Exception as e:
        print(f"Error saving expression feedback: {e}", file=sys.stderr)
        print(f"Error occurred while trying to write to: {ABSOLUTE_CSV_PATH}", file=sys.stderr)
        return jsonify({'error': f'An internal server error occurred while saving expression feedback: {str(e)}'}), 500


# --- Download CSV Endpoint ---
# ... (download_expression_csv remains the same) ...
@app.route('/download_expression_csv', methods=['GET'])
def download_expression_csv():
    try:
        print(f"Download request received for: {ABSOLUTE_CSV_PATH}");
        if not os.path.exists(ABSOLUTE_CSV_PATH): print(f"Expression CSV file not found at {ABSOLUTE_CSV_PATH} for download.", file=sys.stderr); return jsonify({"error": "Expression CSV file not found. Submit some feedback first."}), 404
        return send_file(ABSOLUTE_CSV_PATH, mimetype='text/csv', download_name=CSV_FILENAME, as_attachment=True)
    except Exception as e: print(f"Error downloading expression CSV: {e}", file=sys.stderr); return jsonify({'error': f'An internal server error occurred while preparing the download: {str(e)}'}), 500


# --- Run for Local Testing ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

