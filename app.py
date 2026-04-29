"""
Healthcare Claims AI Agent — Gradio Web App
===========================================
Run locally:  python app.py
Deploy to:    Hugging Face Spaces (set GEMINI_API_KEY in Space secrets)
"""

import os
import importlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# ------------------------------------------------------------------
# Load agent tools
# ------------------------------------------------------------------
import claims_tools
importlib.reload(claims_tools)
from claims_tools import investigate_claims_spike, analyze_incremental_paid_claims

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# ------------------------------------------------------------------
# API key — set GEMINI_API_KEY as a Hugging Face Space secret
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in environment secrets.")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

llm            = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools          = [investigate_claims_spike, analyze_incremental_paid_claims]
agent_executor = create_react_agent(model=llm, tools=tools)


# ------------------------------------------------------------------
# Helper: baseline plot
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Helper: baseline plot
# ------------------------------------------------------------------
def plot_baseline(file_obj):
    if file_obj is None:
        return None, [("", "⚠️ Please upload an Excel claims file first.")]

    file_path = file_obj.name
    try:
        enrollment = pd.read_excel(file_path, sheet_name='fake enrollment')
        claims     = pd.read_excel(file_path, sheet_name='fake claims')
        
        # 1. Strip spaces just in case, to ensure IDs match perfectly
        enrollment['Member_ID'] = enrollment['Member_ID'].astype(str).str.strip()
        claims['Member_ID'] = claims['Member_ID'].astype(str).str.strip()

        # 2. LEFT MERGE: Keep all claims, even if demographic data is missing
        df = pd.merge(claims, enrollment, on='Member_ID', how='left')

        # 3. CRITICAL FIX: Fill missing regions so Pandas doesn't drop them in the groupby
        df['Region'] = df['Region'].fillna('Unknown Region')

        # 4. Parse Dates
        if pd.api.types.is_numeric_dtype(df['Service_Date']):
            df['Date'] = pd.to_datetime(df['Service_Date'], unit='D', origin='1899-12-30')
        else:
            df['Date'] = pd.to_datetime(df['Service_Date'], format='%m/%d/%Y', errors='coerce')
            df['Date'] = df['Date'].fillna(pd.to_datetime(df['Service_Date'], errors='coerce'))

        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        # 5. Group by Billed_Amt to get the absolute totals
        stacked = (
            df.groupby(['YearMonth', 'Region'])['Billed_Amt']
            .sum().reset_index().sort_values('YearMonth')
        )
        stacked['Plot_Date'] = stacked['YearMonth'].dt.to_timestamp()

        # Render the chart
        fig = px.bar(
            stacked, x='Plot_Date', y='Billed_Amt', color='Region', barmode='stack',
            title=f"Absolute Monthly Billed Trend — {os.path.basename(file_path)}",
            labels={'Billed_Amt': 'Absolute Billed Amount ($)', 'Plot_Date': 'Month'}
        )
        fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-30)

        intro = [(
            "",
            "📊 File loaded successfully! Here is your true absolute billed trend.\n\n"
            "You can now ask me to investigate anomalies, decompose month-over-month drivers, "
            "or summarize any patterns you see."
        )]
        return fig, intro

    except Exception as e:
        return None, [("", f"❌ Error loading file: {e}")]

# ------------------------------------------------------------------
# Helper: chat with agent
# ------------------------------------------------------------------
def chat_with_agent(message, file_obj, history):
    if not message.strip():
        return history, ""

    if file_obj is None:
        history = history or []
        history.append((message, "⚠️ Please upload an Excel claims file before asking questions."))
        return history, ""

    file_path = file_obj.name
    system_prompt = (
        "You are a precise healthcare claims analytics assistant. "
        "Use your tools to investigate data files when asked. "
        "Respond concisely and in plain English."
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"[File: '{file_path}']\n"
        f"User: {message}"
    )
    try:
        response = agent_executor.invoke({"messages": [("user", full_prompt)]})
        raw      = response['messages'][-1].content
        reply    = (
            " ".join(item.get('text', '') for item in raw if isinstance(item, dict)).strip()
            if isinstance(raw, list) else str(raw).strip()
        )
    except Exception as e:
        reply = f"❌ Error: {e}"

    history = history or []
    history.append((message, reply))
    return history, ""


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(
    title="🏥 Healthcare Claims AI Agent",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    gr.Markdown("""
    # 🏥 Healthcare Claims AI Agent
    **TP01 — Optimized** | LangChain · LangGraph · Google Gemini 2.5 Flash · Gradio

    Upload your Excel claims file to get started. The agent will plot your baseline trend
    and answer natural language questions about anomalies and month-over-month drivers.

    > **Required:** Excel file with sheets named `fake enrollment` and `fake claims`,
    > both containing a `Member_ID` column.
    """)

    gr.Markdown("---")

    # ── Step 1: File upload ──────────────────────────────────
    gr.Markdown("### Step 1 — Upload Your Claims Data")
    with gr.Row():
        file_upload = gr.File(
            label="📂 Upload Excel Claims File (.xls or .xlsx)",
            file_types=[".xls", ".xlsx"],
            scale=4
        )
        plot_btn = gr.Button("📊 Plot Baseline Trend", variant="primary", scale=1)

    chart_output = gr.Plot(label="Baseline Monthly Billed Trend")

    gr.Markdown("---")

    # ── Step 2: Chat ─────────────────────────────────────────
    gr.Markdown("### Step 2 — Chat with the AI Agent")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=380,
        bubble_full_width=False,
        show_label=False
    )

    with gr.Row():
        chat_input = gr.Textbox(
            label="",
            placeholder="Ask a question about your claims data…",
            scale=5
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

    gr.Examples(
        examples=[
            ["Investigate the month with the biggest claims spike."],
            ["What drove the month-over-month change in paid claims?"],
            ["Which region contributed most to the anomaly?"],
            ["Summarize the overall claims trend for me."],
        ],
        inputs=chat_input,
        label="💡 Example questions"
    )

    # ── Wire up events ────────────────────────────────────────
    plot_btn.click(
        fn=plot_baseline,
        inputs=[file_upload],
        outputs=[chart_output, chatbot]
    )
    send_btn.click(
        fn=chat_with_agent,
        inputs=[chat_input, file_upload, chatbot],
        outputs=[chatbot, chat_input]
    )
    chat_input.submit(
        fn=chat_with_agent,
        inputs=[chat_input, file_upload, chatbot],
        outputs=[chatbot, chat_input]
    )


if __name__ == "__main__":
    demo.launch()
