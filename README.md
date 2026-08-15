# 🛡️ InsureIQ — AI Insurance Premium Predictor

An ML-powered web application that predicts a user's **insurance premium category** based on personal and demographic data.

Built with **FastAPI** (backend), a **scikit-learn RandomForest pipeline** (ML model), and a **modern HTML/CSS/JS** single-page frontend.

---

## 📂 Project Structure

```
.
├── app.py              # FastAPI backend (API + static file serving)
├── model.pkl           # Trained RandomForest pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Production Docker image
├── .dockerignore       # Docker build exclusions
└── static/
    └── index.html      # Full SPA frontend (served at /)
```

---

## 🚀 Quick Start — Local

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn app:app --reload --port 8000
```

### 3. Open in browser

```
https://insureiq-9zpyyr3ah-operapoint86-1507s-projects.vercel.app/
```

> The frontend is served automatically at `/`. API docs are at `/docs`.

---

## 🐳 Docker

### Build

```bash
docker build -t insureiq .
```

### Run

```bash
docker run -p 8000:8000 insureiq
```

Then open `http://localhost:8000`.

---

## 📡 API Reference

### `POST /predict`

Predict the insurance premium category.

**Request body (JSON):**

| Field | Type | Description |
|---|---|---|
| `age` | int | Age (1–119) |
| `weight` | float | Weight in kg |
| `height` | float | Height in metres |
| `income_lpa` | float | Annual income in LPA |
| `smoker` | bool | Smoker status |
| `city` | string | City name |
| `occupation` | string | One of: `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job` |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "weight": 75.0,
    "height": 1.75,
    "income_lpa": 12.0,
    "smoker": false,
    "city": "Mumbai",
    "occupation": "private_job"
  }'
```

**Response:**

```json
{
  "response": {
    "predicted_category": "Low",
    "confidence": 0.82,
    "class_probabilities": {
      "High": 0.05,
      "Low": 0.82,
      "Medium": 0.13
    },
    "computed": {
      "bmi": 24.49,
      "age_group": "adult",
      "lifestyle_risk": "low",
      "city_tier": 1
    }
  }
}
```

### `GET /health`

Returns service health status.

### `GET /city-tier/{city}`

Returns the tier classification for a given city name.

### `GET /docs`

Interactive Swagger UI for the full API.

---

## ☁️ Deployment

### Railway / Render / Fly.io

1. Push to a Git repository
2. Connect to your platform (Railway / Render)
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port 8000`
4. Deploy — the frontend is included automatically

### AWS EC2

```bash
# On the server
git clone <your-repo>
cd <repo>
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Use **nginx** as a reverse proxy and **systemd** to keep uvicorn running.

---

## 🤖 ML Model

The model is a **scikit-learn Pipeline** containing:
- `ColumnTransformer` with `OneHotEncoder` for categorical features
- `RandomForestClassifier`

Trained on `insurance.csv` with features: `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`.

---

*InsureIQ — Know Your Premium Before You Buy*
