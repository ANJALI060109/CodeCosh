<div align="center">

# 🧠 CodeCosh
### *"Know Your Code. Instantly."*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-34A853?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-codecosh.streamlit.app-1a3a6b?style=for-the-badge&logo=streamlit)](https://codecosh.streamlit.app)

<br/>

> **CodeCosh** (Code + कोश) is a full-stack machine learning web application that automatically detects the programming language of any code snippet — with confidence scores, probability charts, and model comparison.

<br/>

[🌐 Live Demo](https://codecosh.streamlit.app) &nbsp;·&nbsp;
[📂 Dataset](data/sample_dataset.csv) &nbsp;·&nbsp;
[🐛 Report Bug](https://github.com/anjali060109/CodeCosh/issues) &nbsp;·&nbsp;
[✨ Request Feature](https://github.com/anjali060109/CodeCosh/issues)

</div>

---

## 📸 Screenshots

<div align="center">

### 🎯 Language Detection
> Paste any code snippet and get instant detection with confidence scores

| Detection Result | Probability Chart |
|---|---|
| ![Detection](https://placehold.co/480x280/1a3a6b/ffffff?text=Detection+Result) | ![Chart](https://placehold.co/480x280/2e5fa3/ffffff?text=Probability+Chart) |

### 🏋️ Model Training
> Upload your own CSV dataset and train both models in seconds

![Training](https://placehold.co/960x320/0d1b2a/F4A827?text=Train+Models+%E2%80%94+Upload+CSV+%E2%80%94+See+Accuracy)

### 📊 Model Metrics
> Accuracy comparison, confusion matrix, and per-language F1 scores

![Metrics](https://placehold.co/960x320/1a3a6b/ffffff?text=Confusion+Matrix+%2B+Accuracy+Charts)

</div>

> 💡 **Tip:** Replace the placeholder images above with real screenshots after deployment!

---

## ✨ Features

### 🎯 Core Features
- **Instant Detection** — Identifies programming language from any code snippet in milliseconds
- **Dual Model Comparison** — Runs Naive Bayes and Logistic Regression side by side
- **Ensemble Prediction** — Automatically picks the most confident model's result
- **Confidence Scores** — Displays probability percentage for every supported language
- **Probability Chart** — Interactive horizontal bar chart showing all language probabilities

### 🏋️ Training Features
- **CSV Upload** — Train on your own labeled dataset with any languages
- **Live Progress Bar** — Animated step-by-step training pipeline visualization
- **Dataset Preview** — Auto-generates distribution chart from uploaded data
- **Configurable Settings** — Adjust test size and TF-IDF feature count

### 📊 Analytics Features
- **Accuracy Comparison** — Visual bar chart comparing both model accuracies
- **Confusion Matrix** — Interactive heatmap for detailed error analysis
- **Per-Language Metrics** — Precision, Recall, and F1-Score for every language

### 🎨 UI/UX Features
- **Quick Example Buttons** — One-click Python, Java, JavaScript code examples
- **Fully Responsive** — Works on desktop and tablet screens
- **Navy + Saffron Theme** — Professional Indian-inspired color palette 🇮🇳

---

## 🧠 Supported Languages

| Language | Icon | Default Samples |
|---|---|---|
| Python | 🐍 | 10 |
| Java | ☕ | 10 |
| C++ | ⚙️ | 10 |
| JavaScript | 🟨 | 10 |
| SQL | 🗄️ | 10 |
| Bash | 🖥️ | 10 |

> ✅ Extend to **any language** by adding labeled rows to your CSV!

---

## 🔄 ML Pipeline

```
Code Input
    ↓
Preprocessor
(remove comments · normalize strings · replace numbers)
    ↓
TF-IDF Vectorizer
(bigrams · sublinear TF · up to 5000 features)
    ↓
┌─────────────────────┬──────────────────────────┐
│   Naive Bayes       │   Logistic Regression    │
│   MultinomialNB     │   max_iter=1000, C=1.0   │
│   alpha=0.1         │   random_state=42        │
└─────────────────────┴──────────────────────────┘
    ↓
Ensemble → picks highest-confidence prediction
    ↓
Language + Confidence Score + Full Probability Distribution
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35 | Interactive web UI |
| **ML Models** | Scikit-learn 1.5 | Naive Bayes + Logistic Regression |
| **Vectorization** | TF-IDF | Text feature extraction |
| **Data Handling** | Pandas 2.2, NumPy 1.26 | Dataset processing |
| **Charts** | Plotly 5.22 | Interactive visualizations |
| **Language** | Python 3.10+ | Core runtime |

---

## 📁 Project Structure

```
CodeCosh/
├── app.py                   ← Streamlit frontend (main entry point)
├── predictor.py             ← ML engine (training + prediction logic)
├── requirements.txt         ← Python dependencies
├── data/
│   └── sample_dataset.csv   ← 60 labeled code snippets (6 languages)
├── model/                   ← Auto-generated after training (.pkl files)
│   ├── vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── label_encoder.pkl
│   └── metrics.pkl
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anjali060109/CodeCosh.git
cd CodeCosh

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser 🎉

### Quick Usage

1. Go to the **🏋️ Train Model** tab
2. Upload `data/sample_dataset.csv` (or your own CSV)
3. Click **Start Training**
4. Switch to **🎯 Detect Language**
5. Paste any code and click **Detect Language**

---

## 📋 CSV Dataset Format

```csv
code,language
"def hello(): print('Hello World')",Python
"System.out.println(\"Hello World\");",Java
"cout << \"Hello World\" << endl;",C++
"console.log('Hello World');",JavaScript
"SELECT * FROM users WHERE id = 1;",SQL
"echo 'Hello World'",Bash
```

**Rules:**
- Minimum **10 rows** required
- At least **2 samples per language** for stratified splitting
- Column names must be exactly `code` and `language`
- Any language name is supported — just keep it consistent

---

## ☁️ Deployment on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your forked repo
5. Set **Main file path** → `app.py`
6. Set **App URL** → `codecosh`
7. Click **Deploy** 🚀

---

## 🤝 Contributing

Contributions are welcome!

```bash
# 1. Fork the repo
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m "Add AmazingFeature"

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request
```

### Ideas for Contribution
- 🌐 Add more programming languages to the dataset
- 🧠 Integrate a deep learning model (LSTM / Transformer)
- 🎨 Add syntax highlighting to the code input
- 📱 Improve mobile responsiveness
- 🔌 Add a REST API layer for third-party integration
- 📊 Add ROC curve and AUC score visualizations

---

## 📄 License

Distributed under the **MIT License** — free to use, modify, and distribute with attribution.
See [`LICENSE`](LICENSE) for full details.

---

## 👩‍💻 Author

**Anjali**

[![GitHub](https://img.shields.io/badge/GitHub-anjali060109-181717?style=flat-square&logo=github)](https://github.com/anjali060109)

---

<div align="center">

**🧠 CodeCosh** &nbsp;·&nbsp; *"Know Your Code. Instantly."*

Built with ❤️ using Python · Scikit-learn · Streamlit

⭐ **Star this repo if you found it helpful!** ⭐

</div>
