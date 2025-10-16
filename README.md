
# 🏆 Portfolio Optimizer Pro

## Overview

The **Portfolio Optimizer Pro** is a comprehensive financial web application built on **Streamlit** that utilizes the principles of **Modern Portfolio Theory (MPT)**, primarily the **Markowitz Efficient Frontier**, to help users find the optimal allocation of assets. It is designed to be an all-in-one tool for analysis, personalization, and investment planning.

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
  * **Global Data Support:** Fetches real-time and historical stock data from Yahoo Finance (`yfinance`) for both US and Indian markets (NSE/BSE).

-----

## 🚀 Getting Started

The easiest way to use the app is to access the live deployment.

### 1\. ☁️ Streamlit Community Cloud (Live App)

Click the badge below to launch the application instantly in your browser:

[Link](https://portfoliooptimisationapp-78xxhappezqfssapucb9qjr.streamlit.app)

* Deployment in progress *

### 2\. 💻 Local Setup (Recommended)

To run the application on your local machine:

#### **Prerequisites**

You must have **Python 3.8+** installed.

#### **Step 1: Clone the Repository**

Open your terminal or command prompt and clone the project:

```bash
git clone https://github.com/thisisgss/portfolio_optimisation_app.git
cd portfolio_optimisation_app
```

#### **Step 2: Install Dependencies Reliably**

Install all required libraries using the provided `requirements.txt` file. Using **`python3 -m pip`** is the most reliable way to ensure the packages are installed correctly for your current Python 3 environment:

```bash
python3 -m pip install -r requirements.txt
```

#### **Step 3: Run the App**

Start the Streamlit application from the terminal:

```bash
streamlit run app.py
```

The application will launch automatically in your default web browser, usually at `http://localhost:8501`.

-----

## ⚙️ Dependencies

The application relies on the following key Python libraries, as listed in the simplified `requirements.txt`:

| Library | Purpose |
| :--- | :--- |
| **`streamlit`** | The core framework for creating the web interface. |
| **`yfinance`** | Fetches historical and real-time stock and currency data. |
| **`pandas`/`numpy`/`scipy`** | Fundamental tools for data manipulation, numerical operations, and complex financial optimization. |
| **`plotly`** | Used to generate all interactive charts, including the Efficient Frontier, allocation pie charts, and backtest lines. |

-----

## 🤝 Contribution & Contact

This application was initially developed by **Shyam Sundar** and enhanced with AI assistance from **Gemini**.

  * **Developer:** Shyam Sundar
      * [LinkedIn Profile](https://www.linkedin.com/in/shyam-sundar-837b97192/)
  * **Repository:** [GitHub: thisisgss/portfolio\_optimisation\_app](https://github.com/thisisgss/portfolio_optimisation_app)

If you find any bugs or have feature suggestions, please use the **Issues** section of the GitHub repository.
