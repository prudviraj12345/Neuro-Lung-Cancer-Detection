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

