# 🎓 Student Placement Prediction System

An end-to-end **Machine Learning web application** that predicts whether a student will get placed based on their academic background, skill scores, and extracurricular profile. Built with **Python, Scikit-learn, Streamlit, and SQLite**.

> 🚀 *Predict smart. Prepare better. Get Placed!*

---

## ✨ Features

* 🔐 **User Authentication** — Secure signup/login with bcrypt password hashing
* 🔮 **AI Placement Prediction** — Trained on **100,000 student records**
* 📊 **Probability Score** — Displays the exact probability of placement
* 💡 **Personalized Suggestions** — Actionable recommendations to improve placement chances
* 📜 **Prediction History** — View all previous predictions with timestamps
* 🎨 **Modern Dark UI** — Glassmorphism-inspired design with responsive layouts

---

## 📁 Project Structure

```text
StudentPlacementPrediction/
│── app.py                                   # Streamlit web application
│── auth.py                                  # Authentication logic
│── database.py                              # SQLite database functions
│── student_placement_prediction_dataset_2026.csv
│── requirements.txt                         # Python dependencies
│── README.md                                # Project documentation
│── StudentPlacementPrediction.ipynb         # Model training notebook
│── model_accuracy.txt                       # Best model accuracy

# Generated after running the notebook
│── model.pkl                                # Trained ML model
│── features.pkl                             # Feature columns
│── student.db                               # SQLite database
```

> **Note:** `model.pkl`, `features.pkl`, and `student.db` are excluded from GitHub using `.gitignore`. Generate them locally by running the notebook.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Farheen-Banu26/student-placement-prediction-system.git
cd student-placement-prediction-system/StudentPlacementPrediction
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 Train the Machine Learning Model

Run the Jupyter notebook to preprocess the data, train the models, and generate the required files.

```bash
jupyter notebook StudentPlacementPrediction.ipynb
```

Run all cells from top to bottom.

This generates:

* `model.pkl` — Trained best-performing model
* `features.pkl` — Feature column names
* `model_accuracy.txt` — Best model metrics

---

## 🚀 Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

### Application Workflow

1. Sign Up → Create a new account
2. Login → Authenticate securely
3. Predict Placement → Enter student details and get predictions
4. View History → Access previous predictions

---

## 📊 Dataset Information

| Property        | Value                                           |
| --------------- | ----------------------------------------------- |
| Dataset File    | `student_placement_prediction_dataset_2026.csv` |
| Records         | 100,000 student profiles                        |
| Features        | 25 columns                                      |
| Target Variable | `placement_status`                              |
| Dataset Type    | Synthetic dataset for 2026 academic profiles    |

### Key Features Used

#### 🎓 Academic

* CGPA
* Branch
* College Tier
* Backlogs
* Attendance Percentage

#### 🏆 Activities

* Internships
* Projects
* Certifications
* Hackathons
* GitHub Repositories

#### 💻 Skills

* Coding Skill Score
* Aptitude Score
* Communication Skills
* Logical Reasoning
* Mock Interview Score

#### 🌐 Social & Leadership

* LinkedIn Connections
* Volunteer Experience
* Leadership Score

#### 🧘 Lifestyle

* Sleep Hours
* Study Hours per Day

---

## 🤖 Machine Learning Models

| Model               | Type              | Purpose                   |
| ------------------- | ----------------- | ------------------------- |
| Logistic Regression | Linear Classifier | Baseline model            |
| Decision Tree       | Tree-Based        | Interpretable predictions |
| Random Forest       | Ensemble          | Best-performing model 🏆  |

---

## 📈 Evaluation Metrics

The models were evaluated using:

* ✅ Accuracy
* ✅ Precision
* ✅ Recall
* ✅ F1 Score
* ✅ Confusion Matrix
* ✅ Classification Report

---

## 📸 Screenshots

Add screenshots after deployment.

| Login Page                 | Prediction Form          | Prediction Result            |
| -------------------------- | ------------------------ | ---------------------------- |
| ![Login](images/login.png) | ![Form](images/form.png) | ![Result](images/result.png) |

---

## 🔮 Future Enhancements

* 📧 Email verification during signup
* 📱 Improved mobile responsiveness
* 🧠 Integration of XGBoost and Neural Networks
* 📈 Advanced analytics dashboards
* ☁️ Deployment on Streamlit Cloud
* 🗂️ Export prediction history as CSV/PDF
* 🔍 Explainable AI (SHAP/LIME) for model interpretability

---

## 🧰 Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Frontend:** Streamlit
* **Database:** SQLite
* **Authentication:** bcrypt
* **Visualization:** Matplotlib, Seaborn, Plotly

---

## 👨‍💻 Developer Information

* **Project Type:** College Mini Project
* **Domain:** Machine Learning / AI
* **Year:** 2026

---

⭐ If you found this project useful, consider giving it a star on GitHub!
