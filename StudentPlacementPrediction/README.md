# 🎓 Student Placement Prediction System

An end-to-end **Machine Learning web application** that predicts whether a student will get placed based on their academic background, skill scores, and extracurricular profile. Built with **Python, Scikit-learn, and Streamlit**.

---

## ✨ Features

- 🔐 **User Authentication** — Secure signup/login with bcrypt password hashing
- 🔮 **AI Placement Prediction** — Trained on 100,000 student records
- 📊 **Probability Score** — Shows exact probability of placement
- 💡 **Personalised Suggestions** — Actionable tips to improve placement chances  
- 📜 **Prediction History** — View all past predictions with timestamps
- 🎨 **Premium Dark UI** — Glassmorphism design with gradient aesthetics

---

## 📁 Folder Structure

```
StudentPlacementPrediction/
│── app.py                                   # Streamlit web application
│── auth.py                                  # Authentication (signup/login/logout)
│── database.py                              # SQLite database functions
│── student_placement_prediction_dataset_2026.csv  # Training dataset
│── model.pkl                                # Trained best ML model
│── features.pkl                             # Feature column names
│── model_accuracy.txt                       # Best model accuracy (auto-generated)
│── requirements.txt                         # Python dependencies
│── README.md                                # This file
│── student.db                               # SQLite database (auto-created)
│── StudentPlacementPrediction.ipynb         # Jupyter ML notebook
```

---

## 🛠️ Installation

### 1. Clone / Download the Project

```bash
cd d:/student_placement_prediction_system/StudentPlacementPrediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 How to Run the Jupyter Notebook

The notebook **must be run first** to train the model and save `model.pkl` and `features.pkl`.

```bash
jupyter notebook StudentPlacementPrediction.ipynb
```

Run all cells from top to bottom. After completion you will see:
- `model.pkl` — trained best model
- `features.pkl` — feature column names
- `model_accuracy.txt` — best model accuracy

---

## 🚀 How to Run the Streamlit App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Flow:
1. **Sign Up** → create account
2. **Login** → authenticate
3. **Predict Placement** → fill the form, get instant result
4. **My History** → view past predictions

---

## 📊 Dataset Information

| Property      | Value                                      |
|---------------|--------------------------------------------|
| File          | student_placement_prediction_dataset_2026.csv |
| Rows          | 100,000 student records                    |
| Features      | 25 columns                                 |
| Target Column | `placement_status` (Placed / Not Placed)  |
| Source        | Synthetically generated for 2026 academics |

### Key Features Used:
- **Academic:** CGPA, Branch, College Tier, Backlogs, Attendance
- **Activities:** Internships, Projects, Certifications, Hackathons, GitHub Repos
- **Skills:** Coding, Aptitude, Communication, Logical Reasoning, Mock Interview
- **Social:** LinkedIn Connections, Volunteer Experience, Leadership Score
- **Lifestyle:** Sleep Hours, Study Hours/Day

---

## 🤖 Algorithms Used

| Model                    | Type      | Notes                          |
|--------------------------|-----------|-------------------------------|
| Logistic Regression       | Linear    | Baseline classifier            |
| Decision Tree Classifier  | Tree      | Interpretable, fast            |
| Random Forest Classifier  | Ensemble  | Best performer, selected 🏆   |

---

## 📈 Evaluation Metrics

For each model:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ Confusion Matrix
- ✅ Classification Report

---

## 📸 Screenshots

> _Run the app and take screenshots to add here._

| Login Page | Prediction Form | Result |
|------------|-----------------|--------|
| ![login]() | ![form]()       | ![result]() |

---

## 🔮 Future Improvements

- 📧 Email verification during signup
- 📱 Mobile-responsive layout enhancements
- 🧠 Add XGBoost / Neural Network models
- 📈 Interactive charts for prediction insights
- 🌐 Deploy to Streamlit Cloud / Heroku
- 🗂️ Export prediction history as CSV/PDF

---

## 👨‍💻 Developer Info

- **Project Type:** College Mini Project
- **Domain:** Machine Learning / AI
- **Year:** 2026
- **Stack:** Python · Scikit-learn · Streamlit · SQLite · bcrypt

---

> *"Predict smart. Prepare better. Get Placed!"* 🚀
