---
title: Healthcare Claims AI Agent
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---
# 🏥 Healthcare Claims AI Agent

> **TP01 — Optimized** | LangChain · LangGraph · Google Gemini 2.5 Flash · Gradio

An AI-powered healthcare claims analytics agent that ingests Excel claims data, detects billing anomalies, and lets users investigate month-over-month trends through a conversational interface — all powered by a LangChain ReAct agent with persistent memory.

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| 📊 Baseline Trend Visualization | Stacked bar chart of monthly billed amounts by region |
| 🔍 Anomaly Detection | Automatically finds the largest month-over-month dollar swing and identifies its drivers |
| 📉 MoM Driver Decomposition | Breaks down paid claims change into percentage-point contributions by Region and Claim Type |
| 💬 Conversational AI Interface | ReAct agent (Gemini 2.5 Flash) answers natural language questions about the data |
| 🌐 Deployable Web App | Full Gradio UI — runs locally or on Hugging Face Spaces |

---

## 🏗️ Architecture

```
User (Gradio UI)
      │
      ▼
LangGraph ReAct Agent  ←──  Google Gemini 2.5 Flash (LLM)
      │
      ├──→ Tool 1: investigate_claims_spike()
      │         └── Finds biggest MoM $ swing in BILLED amounts
      │             Plots stacked bar + anomaly annotation
      │
      └──→ Tool 2: analyze_incremental_paid_claims()
                └── Decomposes MoM % change in PAID amounts
                    by Region × Claim Type (percentage points)
                    Plots faceted bar chart
```

---

## 📁 Repository Structure

```
healthcare-claims-ai-agent/
├── app.py                  # Gradio web app — run locally or deploy to Hugging Face
├── claims_tools.py         # LangChain @tool definitions (the two AI tools)
├── requirements.txt        # Python dependencies
├── TP01_optimized.ipynb    # Original Google Colab notebook
└── README.md
```

---

## 🚀 Quickstart

### Option A — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Katredandred/healthcare-claims-ai-agent.git
cd healthcare-claims-ai-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY="your_key_here"        # Mac/Linux
set GEMINI_API_KEY=your_key_here             # Windows

# 4. Launch the app
python app.py
```

Then open http://localhost:7860 in your browser.

### Option B — Google Colab

1. Open `TP01_optimized.ipynb` in [Google Colab](https://colab.research.google.com)
2. Add your Gemini API key to Colab Secrets (🔑 sidebar → `GEMINI_API_KEY`)
3. Run all cells in order

### Option C — Hugging Face Spaces *(live deployment)*

See the [Hugging Face Spaces setup guide](#hugging-face-spaces-deployment) below.

---

## 📊 Data Format

Your Excel file must contain two sheets:

**Sheet: `fake enrollment`**
| Member_ID | Region | ... |
|---|---|---|
| M001 | Northeast | ... |

**Sheet: `fake claims`**
| Member_ID | Service_Date | Billed_Amt | Paid_Amt | Type | ... |
|---|---|---|---|---|---|
| M001 | 2024-01-15 | 1200.00 | 980.00 | Hospital | ... |

---

## 🤖 AI Tools Reference

### `investigate_claims_spike(file_path)`
- Finds the month with the largest absolute dollar change in **billed** amounts
- Identifies the top driver by Region × Claim Type
- Renders an annotated stacked bar chart with the anomaly circled in red

### `analyze_incremental_paid_claims(file_path)`
- Calculates month-over-month % change in **paid** claims
- Decomposes the change into percentage-point contributions per Region × Type
- Renders a faceted bar chart showing each driver's contribution

---

## 🌐 Hugging Face Spaces Deployment

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space** → name it `healthcare-claims-ai-agent`
3. Set SDK to **Gradio**
4. Connect to this GitHub repo (or upload files manually)
5. Go to **Settings → Variables and Secrets** → add `GEMINI_API_KEY`
6. Your live URL: `https://huggingface.co/spaces/YOUR_USERNAME/healthcare-claims-ai-agent`

---

## 👥 Contributors

| Name | GitHub | Role |
|---|---|---|
| Kate Lawal | [@Katredandred](https://github.com/Katredandred) | Project Lead · AI Agent Architecture |
| Nathan Burns | [@burnsnathanielpcbz-jpg](https://github.com/burnsnathanielpcbz-jpg) | Project Lead · AI Agent Architecture|
| Olufemi Isijola | [@placeholder_username3](https://github.com/placeholder_username3) | *(update with teammate's username)* |
| Kundan Singh | [@placeholder_username4](https://github.com/placeholder_username4) | *(update with teammate's username)* |

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash via `langchain-google-genai`
- **Agent Framework**: LangGraph ReAct agent with persistent conversation memory
- **Tools**: Custom `@tool` decorated functions via LangChain
- **Visualization**: Plotly Express + Plotly Graph Objects
- **UI**: Gradio Blocks
- **Data**: Pandas, openpyxl, xlrd

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built as part of TP01 — William & Mary, Raymond A. Mason School of Business*
