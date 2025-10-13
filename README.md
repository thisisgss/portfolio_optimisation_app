
# 🏆 Portfolio Optimizer Pro

## Overview

The **Portfolio Optimizer Pro** is a comprehensive financial web application built on **Streamlit** that utilizes the principles of **Modern Portfolio Theory (MPT)**, primarily the **Markowitz Efficient Frontier**, to help users find the optimal allocation of assets. It is designed to be an all-in-one tool for analysis, personalization, and planning.

## ✨ Key Features

  * **Efficient Frontier & Optimization:**
      * Performs **Monte Carlo simulations** (for unconstrained portfolios) or **SLSQP Constrained Optimization** (for custom min/max weight rules) to identify the portfolio with the **Maximum Sharpe Ratio** and the **Minimum Volatility**.
  * **Personalized Allocation:**
      * A built-in risk assessment questionnaire determines the user's **risk profile** (Conservative, Moderate, or Aggressive).
      * It then adjusts the optimal risky portfolio's weights by allocating capital between the optimal assets and a **risk-free asset** (cash/bonds) based on the user's risk tolerance.
  * **Investment Plan Generation:**
      * Generates a precise, real-world **Investment Breakdown** showing the exact **Amount to Invest** and the **Number of Shares** to purchase for each asset, based on a user-defined total investment amount and the latest market prices.
      * Handles currency conversion automatically for USD and INR stocks.
  * **Historical Backtesting:**
      * Compares your personalized portfolio's cumulative growth against a major market **benchmark** (e.g., ^NSEI for NIFTY 50) over the entire historical data range.
  * **Comprehensive PDF Report:**
      * Generates a detailed, downloadable PDF report that summarizes the core metrics, personalized weights, investment breakdown, and the historical backtest chart.

## 🚀 Getting Started

The easiest way to use the app is to access the live deployment.

### 1\. ☁️ Streamlit Community Cloud (Live App)

Click the badge below to launch the application instantly in your browser:

[](https://www.google.com/search?q=https://thisisgss-portfolio-optimisation-app-app-l4lrbv.streamlit.app/)
*(Please replace the above link with your actual Streamlit Cloud deployment URL.)*

### 2\. 💻 Local Setup

To run the application on your machine:

#### **Prerequisites**

You must have **Python 3.8+** installed.

#### **Step 1: Clone the Repository**

Open your terminal or command prompt and clone the project:

```bash
git clone https://github.com/thisisgss/portfolio_optimisation_app.git
cd portfolio_optimisation_app
```

#### **Step 2: Install Dependencies**

Install all required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### **Step 3: Run the App**

Start the Streamlit application from the terminal:

```bash
streamlit run app.py
```

The application will launch automatically in your default web browser, usually at `http://localhost:8501`.

## ⚙️ Dependencies

The application relies on the following key Python libraries, all listed in `requirements.txt`:

| Library | Purpose |
| :--- | :--- |
| **`streamlit`** | The core framework for creating the web interface. |
| **`yfinance`** | Fetches historical and real-time stock and currency data. |
| **`pandas`/`numpy`/`scipy`** | Fundamental tools for data manipulation, mathematical operations, and complex financial optimization. |
| **`plotly`** | Used to generate all interactive charts, including the Efficient Frontier, allocation pie charts, and backtest lines. |
| **`fpdf2`** | Used to create and format the downloadable PDF financial report. |
| **`kaleido`** | Plotly dependency required to export charts as static images for the PDF report. |

## 🤝 Contribution & Contact

This application was initially developed by **Shyam Sundar** and enhanced with AI assistance from **Gemini**.

  * **Developer:** Shyam Sundar
      * [LinkedIn Profile](https://www.linkedin.com/in/shyam-sundar-837b97192/)
  * **Repository:** [GitHub: thisisgss/portfolio\_optimisation\_app](https://github.com/thisisgss/portfolio_optimisation_app)

If you find any bugs or have feature suggestions, please use the **Issues** section of the GitHub repository.
