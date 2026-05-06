# Project Strategy: Depression Analysis Web Desk (Spring Boot + Python AI)

This document outlines the architecture and implementation plan for building a full-stack student screening system.

## 1. System Architecture

The application will use a **polyglot architecture** to bridge the gap between Java (Enterprise Web) and Python (Machine Learning).

- **Frontend**: Pure HTML5, CSS3, and JavaScript (Vanilla) for a lightweight and responsive student/admin interface.
- **Backend (Spring Boot)**: Handles business logic, student management, database persistence (H2/MySQL), and orchestration.
- **AI Microservice (Python)**: A lightweight Flask API that loads the pre-trained **Gradient Boosting** model and provides real-time predictions.
- **Database**: Stores student profiles and screening history for admin insights.

---

## 2. Implementation Roadmap

### Phase 1: Model Serving (Python)
Since the model is trained in Python, we will expose it via a REST API:
- **Files**: `app.py` (Flask), `model.joblib`.
- **Endpoint**: `POST /predict` (Accepts form features, returns "Depressed" or "Healthy" with probability).

### Phase 2: Backend Development (Spring Boot)
- **Entities**: `Student`, `ScreeningResult`.
- **Controllers**:
    - `StudentController`: Handles form submission, calls Python AI service, and saves results.
    - `AdminController`: Fetches all screening records for the dashboard.
- **Service Layer**: Integration with Python API using `RestTemplate` or `WebClient`.

### Phase 3: Frontend Development
- **Student Desk**:
    - `index.html`: Professional landing page.
    - `form.html`: Interactive questionnaire matching the dataset features.
    - `result.html`: Displays screening status and recovery suggestions.
- **Admin Desk**:
    - `admin.html`: Dashboard with a table of results, search by Enrollment ID, and high-level charts (Chart.js).

---

## 3. Deployment & Integration Strategy

### How to use the Gradient Boosting model behind the app?
Spring Boot will act as a "Client" to the Python script:
1. Student fills out the form.
2. Spring Boot receives the data.
3. Spring Boot sends a JSON to `localhost:5000/predict`.
4. Python script processes the data through the model and returns the result.
5. Spring Boot saves the record and shows the user their result.

### Setup Instructions
1. **Backend-ML**: 
   - Ensure `model.joblib` exists in `outputs/gradient_boosting/`.
   - Run the Flask app (`python serve_model.py`).
2. **Backend-Java**:
   - Initialize Spring Boot with `Web`, `JPA`, `H2`, and `Lombok`.
   - Configure it to talk to the Flask port.
3. **Frontend**:
   - Serve static HTML files from `src/main/resources/static`.

---

## 4. Feature Mapping (Form Questions)
Based on `config.py`, the form will include:
- Academic Pressure (1-5)
- Study Satisfaction (1-5)
- Sleep Duration (mapped to ordinal values)
- Dietary Habits (Healthy/Unhealthy)
- Suicidal Thoughts (Yes/No)
- Financial Stress (1-5)
- ...and other CORE_FEATURES.
