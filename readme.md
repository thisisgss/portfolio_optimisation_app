
# 🏆 Portfolio Optimizer Pro

## Overview

**Portfolio Optimizer Pro** is a comprehensive, end-to-end financial web application built with Streamlit. It guides users through the entire investment process—from defining a set of assets and finding the mathematically optimal allocation using **Modern Portfolio Theory (MPT)**, to personalizing that portfolio based on individual risk tolerance, generating a precise investment plan, and backtesting its historical performance.

This tool replaces subjective guesswork with a data-driven approach, leveraging the **Markowitz Efficient Frontier** and the **Capital Allocation Line (CAL)** to construct and validate sophisticated investment strategies.

## ✨ Key Features

  * **Dual-Mode Portfolio Optimization:**

      * **Monte Carlo Simulation:** For unconstrained portfolios, the app runs thousands of simulations to map the Efficient Frontier and identify the optimal risk-return trade-offs.
      * **Constrained Optimization (SLSQP):** Allows users to set custom **minimum/maximum weight constraints** for each asset, tailoring the optimization to specific strategies or preferences.

  * **Interactive Analytics Dashboard:**

      * **Efficient Frontier Plot:** Visualizes the risk vs. return of all simulated portfolios, highlighting the **Max Sharpe Ratio** (optimal) and **Minimum Volatility** portfolios.
      * **Optimal Allocation Breakdown:** Displays the ideal asset weights in an intuitive pie chart and a detailed table.
      * **Correlation Matrix:** A heatmap that reveals the correlations between assets, helping users understand the benefits of diversification in their portfolio.
      * **Key Financial Metrics:** Calculates and displays essential metrics including **Expected Return**, **Volatility (Risk)**, **Sharpe Ratio**, **95% Value at Risk (VaR)**, and **Max Drawdown**.

  * **Personalized Allocation via Capital Allocation Line (CAL):**

      * Instead of a rigid questionnaire, an **interactive slider** allows you to dynamically allocate your capital between the optimized risky portfolio and a risk-free asset.
      * This immediately visualizes your personalized portfolio on the **Capital Allocation Line (CAL)**, showing the direct impact of your risk tolerance on expected returns.

  * **Actionable Investment Plan Generation:**

      * Based on your total investment amount, the app generates a real-world **Investment Breakdown**.
      * It calculates the exact **amount to invest** and the **number of shares** to purchase for each stock using the latest market prices.
      * Intelligently handles leftover cash from share rounding, allowing you to allocate it to your risk-free asset.

  * **Historical Performance Backtesting:**

      * Validate your personalized portfolio by comparing its historical growth against any **benchmark ticker** (e.g., `^NSEI` for NIFTY 50, or a specific stock like `TCS.NS`).
      * The cumulative growth is plotted over time, providing a clear picture of your strategy's performance relative to the market.

  * **Save & Compare Portfolios:**

      * **Save multiple optimized portfolios** to your session.
      * Access a dedicated "Compare" page to view the key metrics of all your saved portfolios in a single, clear table, making it easy to choose the best one for your needs.

-----

## 🚀 Getting Started

You can run the application on your local machine by following these steps.

**[StreamLit Link](https://portfoliooptimisationapp.streamlit.app/))**  
#Yet to be deployed

#### Prerequisites

You must have **Python 3.8+** installed on your system.

#### Step 1: Clone the Repository

Open your terminal or command prompt, and clone the project:

```bash
git clone https://github.com/thisisgss/portfolio_optimisation_app.git
cd portfolio_optimisation_app
```

#### Step 2: Install Dependencies

Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### Step 3: Run the App

Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser, typically at `http://localhost:8501`.

-----

## ⚙️ Core Technologies & Libraries

The application is built with the following key Python libraries:

| Library | Purpose |
| :--- | :--- |
| **`streamlit`** | The core framework for building the interactive web interface. |
| **`yfinance`** | Fetches historical and real-time stock data from Yahoo Finance. |
| **`pandas` & `numpy`** | Used for data manipulation, cleaning, and numerical operations. |
| **`scipy`** | Powers the constrained optimization algorithm (SLSQP). |
| **`plotly`** | Generates all interactive charts, including the Efficient Frontier, CAL, pie charts, and backtesting graphs. |

-----

## 🤝 Contribution & Contact

This application was developed by **Shyam Sundar** with AI-powered code generation and enhancement from **Google's Gemini**.

  * **Developer:** Shyam Sundar
      * [LinkedIn Profile](https://www.linkedin.com/in/shyam-sundar-837b97192/)
  * **Repository:** [GitHub: thisisgss/portfolio\_optimisation\_app](https://github.com/thisisgss/portfolio_optimisation_app)

If you find any bugs or have feature suggestions, please raise an issue in the GitHub repository.

> The application was built completely using prompts from me and the coding was done by Gemini AI - I have zero knowledge in coding - it's awesome that AI, if put to use in a good way, gives great outcomes.
