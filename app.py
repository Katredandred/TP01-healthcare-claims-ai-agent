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
# API key — set GEMINI_API_KEY as an environment variable or
# as a Hugging Face Space secret before deploying
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Set it as an environment variable locally, "
        "or add it to your Hugging Face Space secrets."
    )

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [investigate_claims_spike, analyze_incremental_paid_claims]
agent_executor = create_react_agent(model=llm, tools=tools)


# ------------------------------------------------------------------
# Helper: baseline plot
# ------------------------------------------------------------------
def plot_baseline(file_path: str):
    try:
        enrollment = pd.read_excel(file_path, sheet_name='fake enrollment')
        claims = pd.read_excel(file_path, sheet_name='fake claims')
        df = pd.merge(claims, enrollment, on='Member_ID', how='inner')

        if pd.api.types.is_numeric_dtype(df['Service_Date']):
            df['Date'] = pd.to_datetime(df['Service_Date'], unit='D', origin='1899-12-30')
        else:
            df['Date'] = pd.to_datetime(df['Service_Date'], format='%m/%d/%Y', errors='coerce')
            df['Date'] = df['Date'].fillna(pd.to_datetime(df['Service_Date'], errors='coerce'))

        df['YearMonth'] = df['Date'].dt.to_period('M')
        stacked = df.groupby(['YearMonth', 'Region'])['Billed_Amt'].sum().reset_index().sort_values('YearMonth')
        stacked['Plot_Date'] = stacked['YearMonth'].dt.to_timestamp()

        fig = px.bar(
            stacked, x='Plot_Date', y='Billed_Amt', color='Region', barmode='stack',
            title=f"Baseline Monthly Billed Trend — {os.path.basename(file_path)}",
            labels={'Billed_Amt': 'Total Billed Amount ($)', 'Plot_Date': 'Month'}
        )
        fig.update_xaxes(dtick="M1", tickformat="%b", tickangle=0)
        return fig

    except Exception as e:
        return f"❌ Error loading data: {e}"


# ------------------------------------------------------------------
# Helper: chat with agent
# ------------------------------------------------------------------
def chat_with_agent(user_message: str, file_path: str) -> str:
    system_instruction = (
        "You are a helpful healthcare AI data assistant. "
        "Use your tools to investigate data files when asked. "
        "Summarize the tool output clearly in plain text."
    )
    full_prompt = (
        f"{system_instruction}\n\n"
        f"[Context: The target file is '{file_path}']. "
        f"User says: {user_message}"
    )
    try:
        response = agent_executor.invoke({"messages": [("user", full_prompt)]})
        raw = response['messages'][-1].content
        if isinstance(raw, list):
            return " ".join(item.get('text', '') for item in raw if isinstance(item, dict)).strip()
        return str(raw).strip()
    except Exception as e:
        return f"❌ Error processing request: {e}"


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
def handle_plot(file_path):
    fig = plot_baseline(file_path)
    if isinstance(fig, str):
        return fig, fig          # error string → both outputs
    intro = (
        "📊 Baseline trend loaded successfully!\n\n"
        "🤖 Agent: Here is your monthly billed trend by region. "
        "Would you like me to investigate the month with the most significant spike, "
        "or analyze month-over-month paid claims drivers?"
    )
    return fig, intro


def handle_chat(message, file_path, history):
    reply = chat_with_agent(message, file_path)
    history = history or []
    history.append((message, reply))
    return history, ""


with gr.Blocks(
    title="🏥 Healthcare Claims AI Agent",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    gr.Markdown("""
    # 🏥 Healthcare Claims AI Agent
    **TP01 — Optimized** | LangChain · LangGraph · Google Gemini · Gradio

    Upload your Excel claims file, plot the baseline trend, then chat with the AI agent
    to investigate anomalies and decompose month-over-month drivers.

    > **Required Excel sheets:** `fake enrollment` and `fake claims` with a shared `Member_ID` column.
    """)

    with gr.Row():
        file_input = gr.Textbox(
            label="📁 Excel File Path",
            value="excel claims.xls",
            placeholder="e.g. excel claims.xls or /path/to/file.xls",
            scale=4
        )
        plot_btn = gr.Button("📊 Plot Baseline Trend", variant="primary", scale=1)

    chart_output = gr.Plot(label="Baseline Monthly Trend")

    gr.Markdown("---")
    gr.Markdown("### 💬 Chat with the AI Agent")

    chatbot = gr.Chatbot(label="Conversation", height=350, bubble_full_width=False)

    with gr.Row():
        chat_input = gr.Textbox(
            label="Your message",
            placeholder="e.g. Which month had the biggest spike? What drove the increase?",
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

    # Wire up events
    plot_btn.click(fn=handle_plot, inputs=[file_input], outputs=[chart_output, chatbot])
    send_btn.click(fn=handle_chat, inputs=[chat_input, file_input, chatbot], outputs=[chatbot, chat_input])
    chat_input.submit(fn=handle_chat, inputs=[chat_input, file_input, chatbot], outputs=[chatbot, chat_input])


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
