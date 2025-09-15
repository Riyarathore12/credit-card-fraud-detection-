# Credit Card Fraud Detection

## Setup
2. Create Virtual Environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Train the Model
python train_model.py


➡️ This will create models/best_model.joblib.

5. Run FastAPI App
uvicorn fastapi_app:app --reload


Open in browser: http://127.0.0.1:8000/docs

🚀 API Usage
Endpoint: /predict

Method: POST

Input Example:

{
  "Time": 12345,
  "Amount": 150.0,
  "V1": -1.23,
  "V2": 0.45,
  "V3": -0.67,
  "...": "...",
  "V28": 0.12
}


Output Example:

{
  "fraud_probability": 0.091,
  "fraud_flag": 0,
  "threshold": 0.5,
  "model": "LogisticRegression"
}
