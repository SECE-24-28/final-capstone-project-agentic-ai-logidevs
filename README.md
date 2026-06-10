[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/s7J27iqd)

-----------------------------------------------------------------------------
# 💊 Drug Interaction Checker

A web-based application that helps users identify potential interactions between medications, powered by the Claude AI API.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

---

## ⚠️ Medical Disclaimer

> **This tool is for informational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare provider or pharmacist before making any decisions about your medications.

---

## 📖 Overview

Drug Interaction Checker allows users to input multiple medications and receive an AI-generated analysis of potential interactions, severity levels, and general safety notes. It is designed to be a quick reference aid for patients, caregivers, and healthcare students.

---

## ✨ Features

- 🔍 Check interactions between two or more drugs simultaneously
- 🟡 Color-coded severity indicators (mild / moderate / severe)
- 📋 Clear, plain-language explanations of each interaction
- 💬 Powered by Claude AI for natural language understanding
- 📱 Responsive design — works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| Frontend  | HTML, CSS, JavaScript (or React) |
| AI Engine | Anthropic Claude API    |
| Hosting   | Vercel / Netlify / GitHub Pages |

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/drug-interaction-checker.git
cd drug-interaction-checker

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### Environment Variables

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

> **Never commit your `.env` file.** It is already included in `.gitignore`.

### Running Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Usage

1. Enter the names of the drugs you want to check (e.g., `Warfarin`, `Aspirin`).
2. Click **Check Interactions**.
3. Review the results — each interaction is shown with a severity level and explanation.

**Example input:**
```
Drug 1: Warfarin
Drug 2: Ibuprofen
Drug 3: Aspirin
```

**Example output:**
```
⚠️  Warfarin + Ibuprofen — SEVERE
    May significantly increase bleeding risk. Avoid combination unless directed by a doctor.

🟡  Warfarin + Aspirin — MODERATE
    Low-dose aspirin may be used cautiously with warfarin, but increases bleeding risk.
    Monitor INR closely.
```

---

## 📁 Project Structure

```
drug-interaction-checker/
├── public/
│   └── index.html
├── src/
│   ├── components/       # UI components
│   ├── api/              # Anthropic API integration
│   └── utils/            # Helper functions
├── .env.example
├── .gitignore
├── package.json
└── README.md
```

---

## 🔒 Privacy & Security

- No medication data is stored or logged.
- All API calls are made server-side to protect your API key.
- No user accounts or personal data collection.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com/) for the Claude API
- Open-source drug interaction databases for reference data
- All contributors and testers

---

## 📬 Contact

Have questions or suggestions? Open an [issue](https://github.com/your-username/drug-interaction-checker/issues) or reach out via GitHub Discussions.
------------------------------------------------------------------------------------------------------