## Technical Analysis Dashboard powered by AI using Gemini 2.0

import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import streamlit as st
import yfinance as yf
import tempfile
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

st.set_page_config(layout="wide")
st.title("Technical Analysis Dashboard powered by AI using Gemini 2.0")
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,NVDA,TSLA")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

#Date Range
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

#Technical Indicators Selection
indicators = st.sidebar.multiselect(
    "Select Technical Indicators", 
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)

if st.sidebar.button("Get Data"):
    stock_data = {}
    for ticker in tickers:
        # Download data from yfinance
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
        if not data.empty:
            stock_data[ticker] = data
        else:
            st.warning(f"No data available for {ticker}")
    st.session_state["stock_data"] = stock_data
    st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))

#Check if stock data is available
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    #Create function to build chart
    def analyze_ticker(ticker, data):
        #Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            )
        ])

        #Calculate and add technical indicators to chart
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data["Close"].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode="lines", name="20-Day SMA"))
            elif indicator == "20-Day EMA":
                ema = data["Close"].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode="lines", name="20-Day EMA"))
            elif indicator == "20-Day Bollinger Bands":
                sma = data["Close"].rolling(window=20).mean()
                std = data["Close"].rolling(window=20).std()
                upper_band = sma + 2 * std
                lower_band = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=upper_band, mode="lines", name="Upper Band"))
                fig.add_trace(go.Scatter(x=data.index, y=lower_band, mode="lines", name="Lower Band"))
            elif indicator == "VWAP":
                data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP"))        
        for indicator in indicators:
            add_indicator(indicator)
        fig.update_layout(xaxis_rangeslider_visible=False)

        #Convert chart to a temporary PNG file and read image bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            fig.write_image(temp_file.name)
            tempfile_path = temp_file.name
        with open(tempfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tempfile_path)

        #Image prompt
        image_prompt = {
            "data": image_bytes,
            "mime_type": "image/png",
        }

        #Analysis prompt
        analysis_prompt = (
            f"You are a technical analyst for stocks at a top financial institution."
            f"Based on its candlestick chart and the displayed technical indicators, Analyze the stock chart for {ticker}."
            f"Provide a detailed explaination of your analysis, by mentioning what patterns, trends and signals you see in the chart."
            f"Based on your analysis of the chart, provide a recommendation from the following options:"
            f"'Strong Buy', 'Buy', 'Lean Buy', 'Neutral', 'Lean Sell', 'Sell', 'Strong Sell'."
            f"Return your output as a JSON object with two keys: 'analysis' and 'recommendation'."
        )

        #Send image prompt and analysis prompt to Gemini 2.0
        contents = [
            {"role": "user", "parts": [analysis_prompt]},
            {"role": "user", "parts": [image_prompt]}
        ]

        response = gen_model.generate_content(
            contents=contents
        )

        #Parse JSON response
        try:
            result_text = response.text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("Invalid JSON response")
            
        except json.JSONDecodeError as e:
            result = {"analysis": f"JSON Parsing error: {e}. Raw response text: {response.text}", "recommendation": "Error"}
        except ValueError as ve:
            result = {"analysis": f"ValueError: {ve}. Raw response text: {response.text}", "recommendation": "Error"}
        except Exception as e:
            result = {"analysis": f"Unexpected error: {e}. Raw response text: {response.text}", "recommendation": "Error"}

        return fig, result
    
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("recommendation", "N/A")})
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Analysis:**")
            st.write(result.get("analysis", "No analysis available"))

    with tabs[0]:
        st.subheader("Overall Summary")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)

else:
    st.info("Please fetch stock data using the sidebar.")