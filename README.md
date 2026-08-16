# 🛡️ InsureIQ — AI Insurance Premium Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://insureiq-9zpyyr3ah-operapoint86-1507s-projects.vercel.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end Machine Learning web application that predicts a user's **insurance premium category** (`Low`, `Medium`, `High`) based on demographic, lifestyle, and health data.

🔗 **Live Application:** [https://insureiq-9zpyyr3ah-operapoint86-1507s-projects.vercel.app](https://insureiq-9zpyyr3ah-operapoint86-1507s-projects.vercel.app)

---

## ✨ Features

- 🧠 **ML-Driven Predictions**: Random Forest classification pipeline predicting premium tier with calculated confidence scores and class probability breakdown.
- ⚡ **High Performance Backend**: Built on FastAPI with asynchronous request handling and Pydantic schema validation.
- 🎨 **Modern Responsive UI**: Clean, interactive single-page application with real-time BMI computation and dynamic city tier resolution.
- 🐳 **Containerized & Production Ready**: Fully Dockerized with multi-stage build optimization and health check support.
- 📖 **Interactive API Documentation**: Swagger UI documentation auto-generated at `/docs`.

---

## 📂 Project Structure

```
InsureIQ/
├── app.py              # FastAPI application (API endpoints + static asset mounting)
├── model.pkl           # Trained scikit-learn RandomForest pipeline
├── requirements.txt    # Production Python dependencies
├── Dockerfile          # Production Docker container definition
├── .dockerignore       # Docker build exclusions
├── vercel.json         # Vercel serverless deployment configuration
├── api/                # Serverless entrypoints (for Vercel deployment)
└── static/
    └── index.html      # Responsive Single Page Application frontend
```

---

## 🚀 Quick Start — Local Development

### 1. Clone the repository
```bash
git clone https://github.com/Geetansh124/InsureIQ.git
cd InsureIQ
```

### 2. Set up virtual environment & install dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the FastAPI server
```bash
uvicorn app:app --reload --port 8000
```

### 4. Open in browser
- **Web App**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t insureiq .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name insureiq-app insureiq
```

Access the app at `http://localhost:8000`.

---

## 📡 API Reference

### `POST /predict`
Predict the insurance premium tier based on user input parameters.

#### Request Body (`application/json`)
```json
{
  "age": 35,
  "weight": 75.0,
  "height": 1.75,
  "income_lpa": 12.0,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

| Field | Type | Valid Values / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `age` | `integer` | `1 - 119` | Age in years |
| `weight` | `float` | `> 0` | Weight in kilograms (kg) |
| `height` | `float` | `0.1 - 2.5` | Height in meters (m) |
| `income_lpa`| `float` | `> 0` | Annual income in Lakhs Per Annum (LPA) |
| `smoker` | `boolean` | `true` / `false` | Smoking status |
| `city` | `string` | e.g. `"Mumbai"`, `"Delhi"` | Resident city (mapped to Tier 1, 2, or 3) |
| `occupation`| `string` | `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job` | Primary occupation |

#### Response (`200 OK`)
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

---

### Additional Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status check |
| `GET` | `/city-tier/{city}` | Resolve city tier categorization (1, 2, or 3) |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/redoc` | ReDoc API documentation format |

---

## 🤖 Machine Learning Pipeline

The prediction engine utilizes a scikit-learn `Pipeline` composed of:
1. **Feature Engineering**:
   - `BMI` calculation: $\text{BMI} = \frac{\text{weight (kg)}}{(\text{height (m)})^2}$
   - `Lifestyle Risk` categorization (`low`, `medium`, `high`) from smoking status and BMI.
   - `Age Group` grouping (`young`, `adult`, `middle_aged`, `senior`).
   - `City Tier` classification (Tier 1, Tier 2, Tier 3).
2. **Preprocessing**: `ColumnTransformer` with `OneHotEncoder` for categorical features.
3. **Model Classifier**: `RandomForestClassifier` trained for multi-class classification (`Low`, `Medium`, `High`).

---

## 🌐 Live Deployment

The application is deployed on **Vercel** with full serverless functionality:
👉 **[Open InsureIQ on Vercel](https://insureiq-9zpyyr3ah-operapoint86-1507s-projects.vercel.app)**

---

*InsureIQ — Know Your Premium Before You Buy*
