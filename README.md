<div align="center">

# MindCare — AI-Assisted Student Mental Health Risk Assessment & Institutional Analytics Platform

**A full-stack, cloud-native web platform for AI-powered student depression risk assessment.**

[![Java](https://img.shields.io/badge/Java-17-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.5-6DB33F?style=for-the-badge&logo=spring-boot&logoColor=white)](https://spring.io/projects/spring-boot)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

![Project Status](https://img.shields.io/badge/Status-Complete_%26_Maintained-2EA043?style=flat-square)
![Live Demo](https://img.shields.io/badge/Live_Demo-Offline-9E9E9E?style=flat-square)

<sub>Deployment stack: [Vercel](https://vercel.com/) (frontend) · [Render](https://render.com/) (backend) · [Railway](https://railway.app/) (database)</sub>

---

[Problem Statement](#the-problem) · [Architecture](#system-architecture) · [ML Pipeline](#machine-learning-pipeline) · [Features](#key-features) · [Screenshots](#screenshots) · [Setup](#setup-and-installation) · [API Reference](#api-reference)

</div>

> [!NOTE]
> **Live Demo:** The hosted deployment is temporarily unavailable because the cloud hosting subscription has expired. The project itself is complete and actively maintained. The full source code, documentation, system architecture, screenshots, implementation details, and results remain available in this repository. Please explore the README below for the complete workflow, images, references, and setup instructions. The live deployment may be restored in the future.

---

## Table of Contents

- [The Problem](#the-problem)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Model Selection and Evaluation](#model-selection-and-evaluation)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Backend Architecture - Spring Boot](#backend-architecture-spring-boot)
- [Frontend Architecture - React and Vite](#frontend-architecture-react-and-vite)
- [Deployment Architecture](#deployment-architecture)
- [API Reference](#api-reference)
- [Setup and Installation](#setup-and-installation)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## The Problem

Depression among university students is a **growing global crisis**. According to the WHO, depression is one of the leading causes of disability worldwide, and university-age populations are disproportionately affected. Studies estimate that **30-40% of university students** experience clinically significant depressive symptoms during their academic careers.

Despite the scale of this issue, most institutions **lack systematic, data-driven screening mechanisms**. Traditional approaches rely on voluntary self-reporting or overburdened counselling centres, which means at-risk students often go unidentified until they reach a crisis point.

**MindCare** addresses this gap by providing an **AI-assisted screening platform** that:

- Identifies students who may exhibit patterns consistent with depression risk based on academic, lifestyle, and psychosocial indicators.
- Provides institutions with aggregate analytics to allocate mental health resources effectively.
- Delivers a **hybrid dual-axis risk interpretation** combining absolute pattern-matching severity with relative institutional prioritization.

> **Important:**
> This platform is a screening and decision-support tool, NOT a clinical diagnostic instrument. Predictions are intended to support, never replace, professional mental health evaluation by licensed practitioners.

---

## Project Objectives

1. **Build an end-to-end machine learning pipeline** for depression risk classification using real-world student survey data.
2. **Design a microservices architecture** with clear separation of concerns across the frontend, backend, and ML service.
3. **Implement a dual-axis risk framework** that provides both absolute severity interpretation and relative institutional priority ranking.
4. **Develop an administrative dashboard** with institutional analytics, student management, and report generation capabilities.
5. **Deploy the full system to the cloud** with production-ready infrastructure across Vercel, Render, and Railway.
6. **Ensure ethical AI practices** with transparency about model limitations, data privacy, and appropriate use disclaimers.

---

## Key Features

### Student-Facing

| Feature | Description |
|---|---|
| **Student Verification** | Enrollment-ID-based identity verification before assessment access |
| **Assessment Questionnaire** | Structured screening questionnaire capturing academic, lifestyle, and psychosocial factors |
| **AI-Based Risk Prediction** | Real-time inference from a Gradient Boosting model via Flask microservice |
| **Dual-Axis Risk Report** | Severity Interpretation (absolute) + Institutional Priority (percentile-based) |
| **Prediction Probability Score** | Transparent probability output with interpretable risk categorization |

### Administrator-Facing

| Feature | Description |
|---|---|
| **Secure Admin Authentication** | Signup/Login with Spring Security and college-scoped data isolation |
| **Institutional Dashboard** | Summary metrics, risk distribution charts, and high-risk student alerts |
| **Student Management** | Add individual students or bulk upload via CSV, with duplicate detection |
| **Analytics and Reports** | Visual charts (Chart.js), risk breakdowns, and trend analysis |
| **Dark / Light Theme** | Full theme support with context-based toggling |

### Platform

| Feature | Description |
|---|---|
| **REST API Integration** | Spring Boot to Flask ML microservice communication |
| **Cloud Deployment** | Multi-platform deployment (Vercel + Render + Railway) |
| **Dockerized Backend** | Multi-stage Docker build for the Java backend |
| **Responsive Design** | Mobile-friendly UI with reusable component library |

---

## System Architecture

```
+---------------------------------------------------------------------+
|                          CLIENT BROWSER                              |
|                   React 19 + Vite 8 (SPA)                           |
|           Deployed on Vercel - Dark/Light Theme Support              |
+-----------------------------+---------------------------------------+
                              |  HTTPS (REST API)
                              v
+---------------------------------------------------------------------+
|                      BACKEND API SERVER                              |
|               Spring Boot 3.5 - Java 17 - Maven                     |
|           Deployed on Render (Docker Container)                      |
|                                                                      |
|   +--------------+  +---------------+  +------------------+          |
|   |  Controllers |  |   Services    |  |  Repositories    |          |
|   |  - Student   |  |  - Screening  |  |  (Spring Data    |          |
|   |  - Screening |  |  - Dashboard  |  |   JPA + MySQL)   |          |
|   |  - Dashboard |  |  - Student    |  |                  |          |
|   |  - Admin     |  |  - Admin      |  |                  |          |
|   |  - Health    |  |  - College    |  |                  |          |
|   +------+-------+  +-------+-------+  +--------+---------+         |
|          |                  |                    |                    |
|          |     +------------+                    |                    |
|          |     |  HTTP POST /predict             |                    |
|          |     v                                 v                    |
|   +------------------+                 +------------------+          |
|   |  Flask ML        |                 |  MySQL Database   |         |
|   |  Microservice    |                 |  (Railway)        |         |
|   |  (Render)        |                 |                   |         |
|   |                  |                 |  - students       |         |
|   |  - /predict      |                 |  - screening_     |         |
|   |  - /health       |                 |    results        |         |
|   |  - /features     |                 |  - screening_     |         |
|   |                  |                 |    responses      |         |
|   |  Gradient        |                 |  - admins         |         |
|   |  Boosting Model  |                 |  - colleges       |         |
|   +------------------+                 +------------------+          |
+---------------------------------------------------------------------+
```

**Data flow for a screening request:**

1. **Student** completes the assessment questionnaire on the React frontend.
2. **Frontend** sends responses to the Spring Boot backend via `POST /api/screening/submit`.
3. **Spring Boot** forwards the feature vector to the Flask ML service via `POST /predict`.
4. **Flask** runs the Gradient Boosting pipeline, returns prediction probability + hybrid risk interpretation.
5. **Spring Boot** persists the screening result and responses in MySQL, then returns the result to the frontend.
6. **Frontend** renders the dual-axis risk report with probability scores and actionable interpretation.

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| React | 19 | Component-based UI library |
| Vite | 8 | Build tool and dev server |
| React Router | 7 | Client-side routing (SPA) |
| Chart.js + react-chartjs-2 | 4.5 / 5.3 | Data visualization and analytics charts |
| Lucide React | 1.14 | Icon library |
| Vanilla CSS | - | Custom styling with CSS variables for theming |

### Backend

| Technology | Version | Purpose |
|---|---|---|
| Java | 17 | Core language (LTS) |
| Spring Boot | 3.5 | REST API framework |
| Spring Data JPA | 3.5 | ORM and database abstraction |
| Spring Security | 6 | Authentication and authorization |
| Spring Validation | 3.5 | Request validation with Jakarta annotations |
| Lombok | Latest | Boilerplate reduction (getters, setters, constructors) |
| MySQL Connector/J | Latest | JDBC driver for MySQL |
| Maven | 3.9 | Build and dependency management |
| Docker | Multi-stage | Containerized deployment |

### Machine Learning Service

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12 | Core language |
| Flask | 3.1 | Lightweight REST API server |
| Flask-CORS | 6.0 | Cross-origin resource sharing |
| scikit-learn | 1.5 | Model training and pipeline |
| pandas | 2.2 | Data manipulation and preprocessing |
| NumPy | 2.1 | Numerical computation |
| joblib | 1.4 | Model serialization and loading |
| Gunicorn | 21+ | Production WSGI server |

### Infrastructure

| Service | Provider | Purpose |
|---|---|---|
| Frontend Hosting | Vercel | Static SPA deployment with edge CDN |
| Backend Hosting | Render | Dockerized Java application hosting |
| ML Service Hosting | Render | Python Flask API hosting |
| Database | Railway | Managed MySQL 8 instance |

---

## Machine Learning Pipeline

```
+---------------+    +---------------+    +---------------+    +---------------+
|  Raw Dataset  |--->|  Data         |--->|  Exploratory  |--->|  Feature      |
|  (Kaggle)     |    |  Cleaning     |    |  Data         |    |  Engineering  |
|  27,901 rows  |    |  18,707 rows  |    |  Analysis     |    |  and Encoding |
+---------------+    +---------------+    +---------------+    +---------------+
                                                                       |
       +---------------------------------------------------------------+
       v
+---------------+    +---------------+    +---------------+    +---------------+
|  Correlation  |--->|  PCA          |--->|  Model        |--->|  Model        |
|  Analysis     |    |  (Exploratory |    |  Training and |    |  Deployment   |
|  (Spearman)   |    |   Only)       |    |  Evaluation   |    |  (Flask API)  |
+---------------+    +---------------+    +---------------+    +---------------+
```

### Pipeline Stages

| Stage | Script | Description |
|---|---|---|
| **Configuration** | `training/config.py` | Centralized feature definitions, thresholds, and encoding maps |
| **Data Cleaning** | `training/cleaning.py` | Missing value handling, outlier detection, domain-specific validation |
| **EDA** | `training/eda.py` | Distribution analysis, group comparisons, depression-stratified visualizations |
| **Correlation Analysis** | `training/eda.py` | Spearman correlation heatmaps to identify contributing factors |
| **PCA** | `training/pca.py` | Latent structure identification (exploratory only, not used for feature reduction) |
| **Model Training** | `training/build_pipeline.py` | Unified sklearn Pipeline with encoding, scaling, and classification |
| **Risk Classification** | `training/risk_analysis.py` | Percentile-based Q1/Q3 stratification and risk tier distribution analysis |
| **Inference** | `inference/predictor.py` | Production predictor class loading pipeline.joblib |
| **Risk Interpretation** | `inference/risk.py` | Hybrid dual-axis risk engine (severity + institutional priority) |
| **API Server** | `serve_model.py` | Flask REST API exposing /predict, /health, /features |

> **Note:**
> PCA was used exclusively for exploratory latent structure analysis to understand how features cluster and co-vary. The final model uses all 11 original features without dimensionality reduction, preserving interpretability.

---

## Dataset and Preprocessing

**Source:** [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle)

| Metric | Value |
|---|---|
| Original Records | 27,901 |
| Records After Cleaning | 18,707 |
| Features Selected | 11 |
| Target Variable | Depression (Binary: 0 / 1) |

### Selected Features

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Age | Numerical | Student's age |
| 2 | Gender | Categorical | Male / Female |
| 3 | Degree Group | Categorical | Undergraduate / Postgraduate grouping |
| 4 | Academic Pressure | Ordinal (1-5) | Self-reported academic pressure level |
| 5 | Study Satisfaction | Ordinal (1-5) | Satisfaction with academic environment |
| 6 | CGPA | Numerical | Cumulative Grade Point Average |
| 7 | Work/Study Hours | Numerical | Daily hours spent on work or study |
| 8 | Sleep Duration | Categorical | Binned sleep duration category |
| 9 | Financial Stress | Ordinal (1-5) | Self-reported financial stress level |
| 10 | Suicidal Thoughts | Binary | History of suicidal ideation (Yes / No) |
| 11 | Family History of Mental Illness | Binary | Family history indicator (Yes / No) |

### Preprocessing Steps

1. **Missing Value Handling** - Rows with critical missing values dropped; minor gaps imputed based on domain logic.
2. **Domain Validation** - CGPA range checks, age constraints, and logical consistency enforcement.
3. **Feature Encoding** - Ordinal encoding for ordered categories, one-hot/label encoding for nominal categories.
4. **Outlier Treatment** - Statistical outlier detection and handling for numerical features.
5. **Pipeline Unification** - All encoding and scaling steps integrated into a single sklearn Pipeline object for reproducible inference.

---

## Model Selection and Evaluation

Three classification models were trained and evaluated using stratified cross-validation:

### Models Compared

| Model | Accuracy | F1 Score | ROC-AUC | Why Considered |
|---|---|---|---|---|
| Logistic Regression | 82.56% | 86.71% | 89.20% | Baseline linear model; interpretable coefficients |
| Random Forest | 83.00% | 87.10% | 89.80% | Ensemble of decision trees; captures non-linearity |
| **Gradient Boosting** | **84.31%** | **88.36%** | **90.91%** | **Sequential boosting; best generalization** |

### Why Gradient Boosting Was Selected

1. **Highest across all three metrics** - Accuracy (84.31%), F1 Score (88.36%), and ROC-AUC (90.91%).
2. **Superior probability calibration** - Critical for the dual-axis risk framework, which relies on well-calibrated probability scores rather than just class labels.
3. **Handles mixed feature types** - Effectively processes the mix of ordinal, categorical, and numerical features in the dataset.
4. **Sequential error correction** - Each boosting iteration focuses on previously misclassified samples, improving performance on harder-to-classify students.
5. **Ablation study validated robustness** - Feature ablation confirmed that the model is not over-reliant on any single predictor.

---

## Performance Metrics

### Final Model: Gradient Boosting Classifier

| Metric | Score |
|---|---|
| **Accuracy** | 84.31% |
| **F1 Score** | 88.36% |
| **ROC-AUC** | 90.91% |

### Dual-Axis Risk Interpretation Framework

**Axis 1: Severity Interpretation (Absolute Thresholds)**

| Level | Probability Range | Meaning |
|---|---|---|
| High Risk Tendency | > 0.85 | High similarity to depressive-class patterns |
| Elevated Tendency | 0.60 - 0.85 | Moderate-to-high similarity |
| Mild Tendency | 0.35 - 0.60 | Some similarity detected |
| Minimal Tendency | < 0.35 | Low similarity |

**Axis 2: Institutional Priority (Percentile-Based Q1/Q3)**

| Tier | Percentile Group | Recommended Action |
|---|---|---|
| High | Top 25% (> Q3) | Requires priority attention and further evaluation |
| Moderate | Middle 50% (Q1-Q3) | Suggests monitoring and supportive interventions |
| Low | Bottom 25% (< Q1) | Indicates general awareness level |

---

## Project Structure

```
Depression-Analysis-using-ML/
|
+-- frontend/                          # React + Vite Frontend (SPA)
|   +-- src/
|   |   +-- components/
|   |   |   +-- common/                # Reusable UI components
|   |   |       +-- Button.jsx/.css
|   |   |       +-- Card.jsx/.css
|   |   |       +-- Input.jsx/.css
|   |   |       +-- Modal.jsx/.css
|   |   |       +-- RiskBadge.jsx/.css
|   |   |       +-- ThemeToggle.jsx/.css
|   |   +-- config/
|   |   |   +-- chartDefaults.js       # Chart.js theme-aware defaults
|   |   +-- context/
|   |   |   +-- ThemeContext.jsx        # Dark/Light theme context provider
|   |   +-- pages/
|   |   |   +-- student/               # Student-facing flow
|   |   |   |   +-- EnrollmentPage     # Step 1: Enrollment verification
|   |   |   |   +-- QuestionnairePage  # Step 2: Assessment questionnaire
|   |   |   |   +-- ResultPage         # Step 3: Risk report display
|   |   |   +-- admin/                 # Admin-facing dashboard
|   |   |       +-- LoginPage / SignupPage
|   |   |       +-- AdminLayout        # Sidebar navigation layout
|   |   |       +-- DashboardPage      # Analytics dashboard
|   |   |       +-- StudentsPage       # Student management (CRUD)
|   |   |       +-- ReportsPage        # Visual reports and charts
|   |   |       +-- SettingsPage       # Admin settings
|   |   +-- services/
|   |   |   +-- api.js                 # Centralized API client
|   |   +-- App.jsx                    # Root component with routing
|   |   +-- main.jsx                   # Application entry point
|   +-- vercel.json                    # Vercel SPA rewrite rules
|   +-- package.json
|
+-- backend-java/                      # Spring Boot Backend
|   +-- mindcare/
|       +-- src/main/java/com/bhuvanshu/mindcare/
|       |   +-- controller/            # REST API controllers
|       |   |   +-- AdminController        # Admin auth endpoints
|       |   |   +-- DashboardController    # Analytics and summary endpoints
|       |   |   +-- HealthController       # Health check endpoints
|       |   |   +-- ScreeningController    # Screening submission endpoint
|       |   |   +-- StudentController      # Student CRUD and verification
|       |   +-- service/               # Business logic layer
|       |   |   +-- AdminService           # Authentication logic
|       |   |   +-- CollegeService         # College management
|       |   |   +-- DashboardService       # Analytics aggregation
|       |   |   +-- ScreeningService       # ML service orchestration
|       |   |   +-- StudentService         # Student management logic
|       |   +-- entity/                # JPA entity classes
|       |   |   +-- Admin, College, Student
|       |   |   +-- ScreeningResult, ScreeningResponse
|       |   +-- repository/            # Spring Data JPA repositories
|       |   +-- dto/                   # Data Transfer Objects
|       |   |   +-- ScreeningRequest / ScreeningResultResponse
|       |   |   +-- AdminLoginRequest / AdminLoginResponse
|       |   |   +-- DashboardStudentResponse
|       |   |   +-- SeverityInterpretation / InstitutionalPriority
|       |   |   +-- Probability / BulkAddResponse
|       |   +-- config/               # Spring configuration
|       |       +-- SecurityConfig         # CORS + Security rules
|       |       +-- AppConfig              # RestTemplate bean
|       |       +-- GlobalExceptionHandler # Centralized error handling
|       +-- Dockerfile                 # Multi-stage Docker build
|       +-- pom.xml                    # Maven dependencies
|
+-- backend-ml/                        # Python ML Service
|   +-- training/                      # Training pipeline scripts
|   |   +-- config.py                  # Feature defs, thresholds, encoding maps
|   |   +-- cleaning.py               # Data cleaning and validation
|   |   +-- eda.py                     # Exploratory data analysis
|   |   +-- pca.py                     # PCA exploratory analysis
|   |   +-- build_pipeline.py         # Unified pipeline builder and model trainer
|   |   +-- risk_analysis.py          # Risk stratification analysis
|   |   +-- utils.py                   # Shared utility functions
|   +-- inference/                     # Production inference modules
|   |   +-- predictor.py              # DepressionPredictor class
|   |   +-- risk.py                    # Hybrid dual-axis risk engine
|   |   +-- schema.py                 # Input field metadata / schema
|   +-- data/
|   |   +-- raw/                       # Original Kaggle dataset
|   |   +-- processed/                 # Cleaned dataset
|   +-- outputs/                       # Training artifacts and visualizations
|   |   +-- eda/                       # EDA charts
|   |   +-- correlation/              # Correlation heatmaps
|   |   +-- pca/                       # PCA visualizations
|   |   +-- logistic_regression/      # LR evaluation artifacts
|   |   +-- random_forest/            # RF evaluation artifacts
|   |   +-- gradient_boosting/        # GB evaluation artifacts
|   |   +-- gb_ablation/             # Ablation study results
|   |   +-- risk_classification/      # Risk distribution analysis
|   +-- explanations/                  # Documentation for each module
|   +-- serve_model.py                # Flask API server entry point
|   +-- requirements.txt              # Python dependencies
|
+-- Project Screenshots/              # Application screenshots
+-- .gitignore
+-- README.md
```

---

## Backend Architecture (Spring Boot)

The backend follows a **layered architecture** with clear separation of concerns:

```
Controller Layer  -->  Service Layer  -->  Repository Layer  -->  MySQL
       |                    |
   Validation          Business Logic
   Routing             ML Service Call (RestTemplate --> Flask)
   Response            Data Aggregation
```

### Key Design Decisions

- **College-Scoped Data Isolation** - Every request carries an `X-College-Name` header. The `DashboardService` and `StudentService` filter all queries by the admin's college, ensuring multi-tenant data isolation without complex auth tokens.
- **ML Microservice Orchestration** - `ScreeningService` acts as an orchestrator: it receives the student's responses, transforms them into the ML-expected format, calls the Flask `/predict` endpoint via `RestTemplate`, and persists both the raw responses and the prediction result.
- **Global Exception Handling** - `GlobalExceptionHandler` provides centralized, consistent error responses across all controllers.
- **Docker Deployment** - Multi-stage Dockerfile (maven:3.9.6-eclipse-temurin-17 for build, eclipse-temurin:17-jre-jammy for runtime) minimizes the production image size.

### Entity-Relationship Overview

| Entity | Key Fields | Relationships |
|---|---|---|
| `College` | name (unique) | One-to-Many: Admin, Student |
| `Admin` | email, password, college | Many-to-One: College |
| `Student` | enrollmentId, name, college | Many-to-One: College; One-to-Many: ScreeningResult |
| `ScreeningResult` | prediction, probability, severity, priority | Many-to-One: Student; One-to-Many: ScreeningResponse |
| `ScreeningResponse` | question, answer | Many-to-One: ScreeningResult |

---

## Frontend Architecture (React and Vite)

### Design Principles

- **Component-Driven UI** - Reusable component library (Button, Card, Input, Modal, RiskBadge, ThemeToggle) ensures visual consistency.
- **CSS Variables for Theming** - Dark/Light mode implemented via CSS custom properties toggled through ThemeContext, enabling seamless theme switching without CSS-in-JS overhead.
- **Route-Based Code Splitting** - Student flow (/, /questionnaire, /result) and Admin flow (/admin/*) are cleanly separated.
- **Centralized API Layer** - `services/api.js` provides a single fetch-wrapper with automatic auth header injection (X-College-Name from localStorage).

### Application Routes

| Route | Component | Description |
|---|---|---|
| `/` | `EnrollmentPage` | Student enrollment ID verification |
| `/questionnaire` | `QuestionnairePage` | Multi-step assessment form |
| `/result` | `ResultPage` | Dual-axis risk report display |
| `/admin/login` | `LoginPage` | Admin authentication |
| `/admin/signup` | `SignupPage` | New admin registration |
| `/admin/dashboard` | `DashboardPage` | Analytics overview with charts |
| `/admin/students` | `StudentsPage` | Student management (add, bulk upload, view) |
| `/admin/reports` | `ReportsPage` | Detailed reports and visualizations |
| `/admin/settings` | `SettingsPage` | Admin account settings |

---

## Deployment Architecture

```
                    +--------------+
                    |    Users     |
                    +------+-------+
                           |
              +------------v------------+
              |      Vercel CDN         |
              |   (Frontend - React)    |
              |   Global Edge Network   |
              +------------+------------+
                           | API Calls
              +------------v------------+
              |        Render           |
              |   (Backend - Spring     |
              |    Boot in Docker)      |
              +-----+-----------+-------+
                    |           |
       +------------v--+  +----v-----------+
       |    Render      |  |    Railway      |
       |  (ML Service   |  |  (MySQL 8       |
       |   - Flask)     |  |   Database)     |
       +----------------+  +----------------+
```

| Component | Platform | Configuration |
|---|---|---|
| **Frontend** | Vercel | Auto-deploy from frontend/ directory; SPA rewrites via vercel.json |
| **Backend** | Render | Docker deployment using multi-stage Dockerfile; env vars for DB and ML URLs |
| **ML Service** | Render | Python environment; Gunicorn WSGI server; pipeline.joblib loaded at startup |
| **Database** | Railway | Managed MySQL 8; connection string via environment variables |

---

## API Reference

### Student Endpoints

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| `POST` | `/api/student/verify` | Verify student enrollment | `{ enrollmentId }` |
| `POST` | `/api/student/add` | Add a new student | `{ enrollmentId, name }` |
| `POST` | `/api/student/bulk` | Bulk upload students | `[{ enrollmentId, name }, ...]` |

### Screening Endpoints

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| `POST` | `/api/screening/submit` | Submit screening assessment | ScreeningRequest object |

### Dashboard Endpoints

| Method | Endpoint | Description | Auth Header |
|---|---|---|---|
| `GET` | `/api/dashboard/summary` | Institutional summary metrics | `X-College-Name` |
| `GET` | `/api/dashboard/students` | All students with screening data | `X-College-Name` |
| `GET` | `/api/dashboard/charts` | Risk distribution chart data | `X-College-Name` |
| `GET` | `/api/dashboard/high-risk` | High-risk students list | `X-College-Name` |

### Admin Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/admin/signup` | Register new admin |
| `POST` | `/api/admin/login` | Admin authentication |

### ML Service Endpoints (Flask)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Submit student data, get prediction + risk levels |
| `GET` | `/health` | Health check with model metadata and risk framework |
| `GET` | `/features` | Return expected input fields with metadata |

---

## Setup and Installation

### Prerequisites

- **Java 17** (JDK)
- **Maven 3.9+**
- **Node.js 18+** and **npm**
- **Python 3.10+**
- **MySQL 8** (local or remote)

### 1. Clone the Repository

```bash
git clone https://github.com/bhuvanshu/Depression-Analysis-using-ML.git
cd Depression-Analysis-using-ML
```

### 2. ML Service Setup

```bash
cd backend-ml

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python serve_model.py --host 0.0.0.0
# Server starts at http://localhost:5000
```

### 3. Backend Setup

```bash
cd backend-java/mindcare

# Configure database connection in application.properties:
# spring.datasource.url=jdbc:mysql://localhost:3306/mindcare
# spring.datasource.username=root
# spring.datasource.password=yourpassword
# app.ml-service.url=http://localhost:5000

# Build and run
./mvnw spring-boot:run
# Server starts at http://localhost:8080
```

### 4. Frontend Setup

```bash
cd frontend

# Create .env file
echo "VITE_API_URL=http://localhost:8080/api" > .env

# Install dependencies and run
npm install
npm run dev
# App available at http://localhost:5173
```

### 5. Verify the Stack

| Service | URL | Health Check |
|---|---|---|
| Frontend | `http://localhost:5173` | Open in browser |
| Backend | `http://localhost:8080` | `GET /api/health` |
| ML Service | `http://localhost:5000` | `GET /health` |

---

## Screenshots

<div align="center">

| | |
|---|---|
| **Login Page** | **Student Verification** |
| ![Login Page](Project%20Screenshots/Login%20Page.png) | ![Verification Page](Project%20Screenshots/Verification%20page.png) |
| **Assessment Questionnaire** | **Prediction Report** |
| ![Questionnaire](Project%20Screenshots/Questionare.png) | ![Prediction Report](Project%20Screenshots/Prediction%20report.png) |
| **Admin Dashboard** | **Student Management** |
| ![Dashboard](Project%20Screenshots/Dashboard.png) | ![Students Page](Project%20Screenshots/Students%20page.png) |
| **Reports and Analytics** | |
| ![Reports Page](Project%20Screenshots/Reports%20Page.png) | |

</div>

---

## Future Enhancements

| Enhancement | Description |
|---|---|
| **JWT-Based Authentication** | Replace header-based college scoping with stateless JWT tokens for production-grade security |
| **Longitudinal Tracking** | Track student risk scores over time to identify trends and measure intervention effectiveness |
| **SHAP / LIME Explainability** | Integrate model interpretability to show which factors contributed most to each individual prediction |
| **Multilingual Support** | Localize the questionnaire and reports for non-English-speaking institutions |
| **Notification System** | Automated alerts for administrators when students are flagged as high-risk |
| **PDF Report Export** | Generate downloadable PDF screening reports for counsellors and administrators |
| **Advanced Analytics** | Cohort analysis, department-level comparisons, and seasonal trend detection |
| **Model Retraining Pipeline** | Automated periodic retraining with new screening data to improve model performance over time |
| **Mobile Application** | Native iOS/Android app for on-the-go screening and administration |

---

## Ethical Considerations

> **Caution:**
> This platform is a decision-support tool. It is NOT a clinical diagnostic instrument.

### Important Disclaimers

1. **Not a Diagnosis** - The model predicts statistical similarity to patterns observed in a dataset of depressed students. A "High Risk" result does not constitute a clinical diagnosis of depression.

2. **Supplementary to Professional Care** - Predictions are intended to support, never replace, evaluation by licensed mental health professionals. No automated action (e.g., academic penalties, mandatory counselling) should be taken based solely on model output.

3. **Dataset Limitations** - The model is trained on the [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) from Kaggle. It reflects patterns in that specific population and may not generalize to all cultural, demographic, or institutional contexts.

4. **Privacy and Consent** - Institutions deploying this platform must:
   - Obtain informed consent from students before screening.
   - Comply with applicable data protection regulations (GDPR, FERPA, or local equivalents).
   - Implement appropriate data access controls and retention policies.
   - Ensure screening results are accessible only to authorized personnel.

5. **Bias and Fairness** - The model should be regularly audited for demographic bias across gender, age, and cultural subgroups. No single screening tool can capture the full complexity of mental health.

6. **Transparency** - The platform provides probability scores, severity interpretations, and institutional priority rankings alongside every prediction. This transparency is intentional. Stakeholders should understand the basis and limitations of every result.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository.
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with descriptive messages:
   ```bash
   git commit -m "feat: add longitudinal tracking for student risk scores"
   ```
4. **Push** to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** with a clear description of your changes.

### Commit Convention

| Prefix | Purpose |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Build/tooling changes |

---

## Acknowledgements

- **Dataset:** [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) by HopesB on Kaggle
- **Frameworks:** Spring Boot, React, Flask, scikit-learn
- **Deployment:** Vercel, Render, Railway
- **Icons:** [Lucide Icons](https://lucide.dev/)

---

<div align="center">

**Built for student well-being**

If this project was useful, please consider giving it a star.

</div>
