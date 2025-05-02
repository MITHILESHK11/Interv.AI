# 🎙️ AI-Powered Virtual Interview Simulator

This project is an interactive Streamlit-based application that simulates a realistic interview environment powered by **Google Gemini 1.5 Pro**. It dynamically generates questions, evaluates answers, and produces comprehensive performance reports based on your interview round and topic selections.

---

## 🚀 Features

- 🔧 **Custom Interview Configuration**: Choose from multiple interview types, topics, difficulty levels, and number of questions.
- 🧠 **AI-Generated Questions**: Gemini API generates tailored interview questions on demand.
- 💬 **Real-Time Answer Evaluation**: Evaluates user responses (text/code) using AI with scores and actionable feedback.
- 📈 **Performance Reports**: Generates visual and textual summaries, radar charts, and downloadable CSV reports.
- 💻 **Support for Coding and Non-Coding Rounds**: Includes a code editor and special evaluation for coding-based rounds.

---

## 📂 File Structure

- `app.py`: Main application file with complete Streamlit interface, Gemini integration, and logic.
- Uses Gemini 1.5 Pro API (via `google.generativeai`) for question generation and answer evaluation.

---

## 🧰 Requirements

Install dependencies using pip:

```bash
pip install streamlit pandas plotly matplotlib google-generativeai
```

---

## 🔑 Environment Setup

You need a **Gemini API key** to use the app. Set it via environment variable or input it in the sidebar on app startup.

To set via environment:

```bash
export GEMINI_API_KEY=your_api_key_here
```

Or, enter it manually in the sidebar when prompted.

---

## ▶️ How to Run

Run the app using Streamlit:

```bash
streamlit run app.py
```

---

## 🛠️ Interview Rounds & Topics

- **Technical Round**: Data Structures, Algorithms, OOP, etc.
- **Coding Round**: Python, Java, C++, SQL, Full Stack, etc.
- **HR/Behavioral Round**: Leadership, Teamwork, Conflict Resolution, etc.
- **Concept Round**: ML, AI, DevOps, Cybersecurity, etc.
- **Aptitude Round**: Logical Reasoning, Quantitative, Verbal, etc.
- **Algorithm Coding Round**: DP, Graphs, Recursion, etc.

Each supports **Easy**, **Medium**, and **Hard** levels.

---

## 📊 Reporting Features

After completing an interview, you'll receive:

- **AI Summary Report**: Overall performance, strengths, improvements, recommendations, final verdict.
- **Question Breakdown**: Individual question, answer, score, and feedback.
- **Radar Chart**: Visual representation of performance across questions.
- **Bar Chart**: Score distribution.
- **Download Option**: Export results as CSV.

---

## 🔒 Notes

- Responses are evaluated by Gemini based on correctness, clarity, efficiency, and quality.
- For coding rounds, include well-indented, commented code.
- Make sure the API key is valid and you have billing enabled on your Google Cloud account.

---

## 📜 License

© 2023 AI Interview Simulator | Powered by Google Gemini 1.5 Pro API
