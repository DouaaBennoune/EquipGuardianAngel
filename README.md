# ⚡ Equip-GuardianAngel

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikit-learn)

> AI-powered equipment health monitoring — predicts Remaining Useful Life (RUL) 
> from raw sensor data and classifies units as Healthy, Warning, or Critical.

---

## 📸 Demo
### Home Page
<img width="2408" height="1223" alt="image" src="https://github.com/user-attachments/assets/1d9d6da3-d3fc-4b4f-a519-4583950c690b" />

### Dashboard Page
<img width="2373" height="1242" alt="image" src="https://github.com/user-attachments/assets/5f9cc42d-9479-4842-b5ef-f8d936e63fc5" />
<img width="2405" height="1288" alt="image" src="https://github.com/user-attachments/assets/59eb5f33-a00b-4c7c-8c12-d53c4b053e4f" />

### Docs Page
<img width="2361" height="850" alt="image" src="https://github.com/user-attachments/assets/1a710836-3444-442d-a0e3-77a0a283323f" />


---

## 🏗️ Architecture

```mermaid
flowchart TD
    A["👤 User\nuploads CSV"] --> B

    B["⚡ Streamlit Frontend\nlocalhost:8501\nHome · Upload · Dashboard · Docs"]
    
    B -->|"POST /api/v1/predict"| C

    C["🚀 FastAPI Backend\nlocalhost:8000\nValidate · Preprocess · Window"]

    C --> D["🤖 Random Forest Model\n.pkl · trained on C-MAPSS"]

    D --> E["🟢 Healthy\n> 50 cycles"]
    D --> F["🟡 Warning\n30–50 cycles"]
    D --> G["🔴 Critical\n< 30 cycles"]

    style A fill:#1e293b,stroke:#334155,color:#94a3b8
    style B fill:#0f2744,stroke:#1d4ed8,color:#60a5fa
    style C fill:#0e2a1f,stroke:#059669,color:#34d399
    style D fill:#1a1333,stroke:#7c3aed,color:#a78bfa
    style E fill:#052e16,stroke:#16a34a,color:#22c55e
    style F fill:#2d1a00,stroke:#d97706,color:#fbbf24
    style G fill:#2d0a0a,stroke:#dc2626,color:#f87171
```

## 📊 Dataset

Uses NASA **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation):
- `FD001` — single operating condition, single fault mode.
- Each row = one sensor snapshot from one engine at one cycle.

---
## 🤖 Model

- **Algorithm**: Random Forest trained on windowed sensor sequences
- **Output**: Predicted remaining cycles per engine unit
- **Classes**:
  - 🟢 Healthy → > 50 cycles remaining
  - 🟡 Warning → 30–50 cycles remaining  
  - 🔴 Critical → < 30 cycles remaining

## 🚀 Run with Docker

```bash
git clone https://github.com/DouaaBennoune/EquipGuardianAngel.git
cd EquipGuardianAngel
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| Backend API docs | http://localhost:8000/docs |

---

## 📁 Project Structure

```
├── app/
│   ├── api/          # FastAPI endpoints
│   ├── models/       # Pydantic schemas
│   ├── services/     # ML inference engine
│   └── utils/        # Preprocessing & validation
├── frontend.py        # Streamlit UI
├── Dockerfile.Backend
├── Dockerfile.frontend
└── docker-compose.yml
```
