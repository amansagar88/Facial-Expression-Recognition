# Facial Expression Recognizer - Web App with Live Feedback Loop 🧠

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Frontend](https://img.shields.io/badge/React-18.x-cyan.svg)](https://reactjs.org/)
[![Hosting](https://img.shields.io/badge/🤗%20Spaces-Hosted-yellow.svg)](https://huggingface.co/spaces)

---

## 🌟 Overview

This project is a complete, deployable **Facial Expression Recognition (FER) API**. It leverages a **TensorFlow/Keras** model to classify emotions and **OpenCV** for real-time face detection. The API is built with **Flask** and served using **Gunicorn**, all containerized within a Docker image for simple, reliable deployment.


**Demo Video**

![Demo gif](./assets/demo-video.gif)

---

## ✨ Features

* **AI-Powered Prediction**: Uses a TensorFlow/Keras model to classify 7 different facial expressions (happy, sad, angry, etc.).
* **Multi-Face Detection**: Employs OpenCV's Haar Cascade to find all faces within an uploaded image.
* **Visual & Data Response**: Returns both a JSON object with coordinates and a new image with bounding boxes and emotion labels drawn on.
* **Active Learning Loop**: Includes a feedback endpoint (`/save_expression_feedback`) that captures user corrections and saves the image's pixel data to a CSV.
* **Ready to Deploy**: Comes with a `Dockerfile` and `gunicorn` configuration, making it ready for production and cloud platforms like Hugging Face Spaces.

---

## 🛠️ Tech Stack

* **Frontend:**
    * [React](https://reactjs.org/)
    * HTML5
    * CSS3
* **Backend:**
    * [Flask](https://flask.palletsprojects.com/) (Python Web Framework)
    * [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) (for loading and running the ML model)
    * [NumPy](https://numpy.org/) (for numerical operations)
    * [OpenCV (headless)](https://pypi.org/project/opencv-python-headless/) (for image preprocessing)
    * [Pillow](https://python-pillow.org/) (for image handling)
    * [Gunicorn](https://gunicorn.org/) (WSGI Server)
* **Hosting:**
    * [Hugging Face Spaces 🤗](https://huggingface.co/spaces) (using Docker)

---

## 🚀 Live Demo

**➡️ Check out the live demo at: [https://www.amansagar.dev/projects/facial-expression-recognizer](https://www.amansagar.dev/projects/facial-expression-recognizer)**

*(Note: The first request might take a few seconds if the Space has gone to sleep)*

---

## 📖 API Endpoints

You can interact with the deployed Flask API endpoints directly.

### 1. Health Check (`/`)

* **Method:** `GET`
* **URL:** `https://aman881-facial-expression.hf.space/`
* **Description:** A simple health check endpoint to verify the API is running and if the CSV file exists.
* **Success Response (200 OK):**
    ```json
    {
      "status": "healthy",
      "csv_found": true 
    }
    ```

### 2. Predict Expression (`/predict_expression`)

* **Method:** `POST`
* **URL:** `https://aman881-facial-expression.hf.space/predict_expression`
* **Body (JSON):**
    ```json
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAANS..." 
    }
    ```
* **Success Response (200 OK):**
    Returns the processed image with drawings and a list of detected faces with their predictions.
    ```json
    {
      "processed_image": "data:image/png;base64,...",
      "faces": [
        {
          "box": [120, 150, 95, 95],
          "expression": "happy",
          "confidence": 0.985
        },
        {
          "box": [310, 180, 100, 100],
          "expression": "neutral",
          "confidence": 0.761
        }
      ]
    }
    ```
* **Error Response (400/500):**
    ```json
    {
      "error": "Descriptive error message"
    }
    ```

### 3. Save Feedback (`/save_expression_feedback`)

* **Method:** `POST`
* **URL:** `https://aman881-facial-expression.hf.space/save_expression_feedback`
* **Description:** Saves the pixel data from the first detected face in an image, along with a user-provided "correct" label, to `expression_pixel_data.csv`.
* **Body (JSON):**
    ```json
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAANS...", 
      "label": "sad"
    }
    ```
    * **Note:** Valid string labels are `angry`, `disgusted`, `fear`, `happy`, `sad`, `surprised`, `neutral`, or `nan`.
* **Success Response (200 OK):**
    ```json
    {
      "status": "success",
      "message": "Feedback saved successfully."
    }
    ```
* **Error Response (400/500):**
    ```json
    {
      "error": "Descriptive error message"
    }
    ```

### 4. Download CSV Data (`/download_expression_csv`)

* **Method:** `GET`
* **URL:** `https://aman881-facial-expression.hf.space/download_expression_csv`
* **Response:** Triggers a file download of `expression_pixel_data.csv`.
* **Error Response (404 Not Found):** If no feedback has been submitted yet and the file doesn't exist.
    ```json
    {
      "error": "Expression CSV file not found. Submit some feedback first."
    }
    ```
---

## ⚙️ Setup and Installation (Locally)

### Backend (Flask API)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amansagar88/Facial-Expression-Recognition
    cd digit-recognition-system
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask app:**
    ```bash
    python app.py
    ```
    The API will be running, typically at `http://127.0.0.1:5000`.

## 📊 Feedback & Retraining

* When a user flags a prediction as incorrect and provides the correct label (e.g., "happy", "sad", or "nan"), the backend saves the **label** and the **2304 raw pixel values** (0-255) of the $48 \times 48$ resized grayscale face to `expression_pixel_data.csv`.
* You can download this complete dataset from the `/download_expression_csv` endpoint of the deployed backend[cite: 3].
* This collected data can then be used to fine-tune or retrain the Keras model (`facial_expression_model.keras`) using a separate training script, leading to improved and more customized model accuracy over time.

## 🌱 Future Improvements

* **Automate Retraining:** Create a script or CI/CD pipeline that automatically downloads the `expression_pixel_data.csv` from the `/download_expression_csv` endpoint, uses it to fine-tune the model, and deploys the new `facial_expression_model.keras` file.
* **Upgrade Face Detector:** Replace the current OpenCV Haar Cascade with a more robust deep learning-based detector (like MTCNN, RetinaFace, or a lightweight SSD) to improve accuracy, especially on angled or partially occluded faces.
* **Implement Face Alignment:** Add a preprocessing step to detect facial landmarks and align the face (e.g., warp the image so the eyes are horizontal) before feeding it to the expression model, which can significantly boost consistency.
* **Add Video Stream Support:** Implement a WebSocket endpoint to handle real-time video feeds, allowing for continuous expression analysis from a webcam.
* **Visualize Feedback Data:** Create a simple `/dashboard` endpoint that reads the `expression_pixel_data.csv` and displays statistics, such as a bar chart showing the count of each collected emotion label.
* **Experiment with Model Architectures:** Test different CNN architectures (e.g., MobileNetV2, EfficientNet, or custom-built models) to find a better balance between prediction accuracy and inference speed.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or find bugs, please feel free to open an issue or submit a pull request.

---

## 🙏 Acknowledgements

* Thanks to the creators of React, Flask, TensorFlow, OpenCV, and Hugging Face!

---

## 🧑‍💻 Author

**<a href="https://www.amansagar.dev" style="{text-decoration:none;}">Aman Sagar</a>**  
Data Science Enthusiast | Passionate about ML Algorithms

---

