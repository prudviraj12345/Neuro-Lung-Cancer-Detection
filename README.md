
# Neuro Lung Cancer Detection

A complete Machine Learning–based Lung Cancer Detection System that uses deep learning
to analyze medical images and predict the presence of lung cancer.

This repository contains the full project including:
- Trained deep learning model
- Backend logic
- Frontend interface
- Supporting configuration files

---

## Project Description

Lung cancer is one of the leading causes of cancer-related deaths worldwide.
Early detection can significantly improve survival rates.

This project aims to assist in early lung cancer detection by applying
deep learning techniques on medical imaging data.

The system:
- Accepts lung scan images as input
- Preprocesses images
- Uses a trained deep learning model to predict cancer
- Displays results through a simple frontend interface

---

## Machine Learning Model

- Model Type: Deep Learning (CNN-based)
- Framework: TensorFlow / Keras
- Input: Medical lung images
- Output: Lung cancer prediction
- Model File: lung_cancer_model.h5

### Model Storage
The trained model file is large in size and is included in this repository
using Git Large File Storage (Git LFS).

---

## Frontend

The frontend provides a simple user interface to interact with the system.

### Frontend Features
- Image upload option
- Simple and clean UI
- Displays prediction result

### Frontend Technologies
- HTML
- CSS
- JavaScript (basic)

### Main Frontend File
index.html

---

## Backend

The backend is responsible for:
- Loading the trained model
- Preprocessing input images
- Running predictions
- Returning results

### Backend Technologies
- Python
- Flask
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow

### Backend Files
backend/
- app.py
- requirements.txt
- Procfile
- runtime.txt
- lung_cancer_model.h5

---

## Project Structure

Neuro-Lung-Cancer-Detection/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── runtime.txt
│   └── lung_cancer_model.h5
│
├── index.html
├── README.md
├── .gitignore
├── .gitattributes
└── assets / images (if any)

---

## Deployment Status

This project is not deployed on a live server.

### Reason
- The trained model is large in size
- TensorFlow has strict Python and platform compatibility requirements
- Free hosting platforms (Render, Heroku, etc.) do not reliably support:
  - TensorFlow
  - Large .h5 model files
  - Git LFS model retrieval

This project is intended for:
- Academic submission
- Project evaluation
- Learning and demonstration
- Local execution

---

## How to Run Locally

### Step 1: Clone the repository
git clone https://github.com/prudviraj12345/Neuro-Lung-Cancer-Detection.git
cd Neuro-Lung-Cancer-Detection

### Step 2: Install backend dependencies
pip install -r backend/requirements.txt

### Step 3: Run backend
python backend/app.py

### Step 4: Run frontend
Open index.html in a web browser.

---

## Use Cases

- Academic machine learning project
- Medical image analysis learning
- Deep learning demonstration
- Resume / portfolio project

---

## Technologies Used

- Programming Language: Python
- Machine Learning Framework: TensorFlow, Keras
- Backend Framework: Flask
- Frontend: HTML, CSS
- Image Processing: OpenCV, Pillow
- Version Control: Git, GitHub
- Large File Handling: Git LFS

---

## Key Learnings

- Building and training deep learning models
- Handling medical image datasets
- Using Git LFS for large ML models
- Structuring full-stack ML projects
- Understanding real-world deployment limitations
---

## Note for Reviewers / Evaluators

This repository contains the complete working project, including the trained model.
Deployment was intentionally avoided due to platform limitations.
The project is best evaluated through:
- Code review
- Model architecture
- Local execution


EXTRA INFORMATION:
# Neuro-Lung

This repository contains a Flask API for lung cancer image classification and Grad-CAM heatmap generation.

Quick facts
- Start command: `gunicorn app:app`
- Python entrypoint: `app.py`
- Model file: `lung_cancer_model.h5` (must be placed in repo root)

Run locally (Windows PowerShell)

```powershell
cd "C:\Users\USER\Desktop\neuro-lung\neuro-lung"
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r .\requirements.txt
pip install tf-keras-vis opencv-python matplotlib gunicorn
python .\app.py
```

Save a test response (PowerShell)

```powershell
# POST a sample image and save response
curl.exe -s -X POST -F "file=@C:\Users\USER\Desktop\neuro-lung\neuro-lung\dataset\benign\Benign case (99).jpg" http://127.0.0.1:5000/predict -o test_response.json
# Decode heatmap
$js = Get-Content .\test_response.json -Raw | ConvertFrom-Json
[System.IO.File]::WriteAllBytes('out_heatmap.png', [Convert]::FromBase64String($js.heatmap))
```

Deploy to Render (recommended)

1. Create a GitHub repository and push this project (see below).
2. Sign in to https://render.com and create a new **Web Service**.
3. Connect your GitHub account and select this repository.
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Add the model file to the repo or upload it to a secured storage and update `app.py` accordingly.

Alternative: deploy to Heroku (requires `HEROKU_API_KEY` for CI/CD) or Railway.

Git push example

```powershell
git init
git add .
git commit -m "Initial commit"
# create a GitHub repo using the web UI then:
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

