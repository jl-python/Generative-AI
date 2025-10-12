# Track B - Streamlit & Fast API

This repository contains my Week 7 submission for the Generative AI Integration course.  
It includes **Track B (End-to-End Application Deployment with Project Models)**


## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/jl-python/Generative-AI.git
cd week7
```

### 2. Create Virtual Environment
``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r app/requirements.txt
```

### 4. Start Fast API
```bash
cd week7/backend
uvicorn main:app --reload --port 8000
```

### 5. Run Streamlit Front End
```bash
cd ../app
streamlit run app.py
```



