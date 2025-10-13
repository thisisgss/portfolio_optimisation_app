# Setting up environment
# To run this app, save the code as a .py file (e.g., app.py) and run `streamlit run app.py` in your terminal.
# You will need to install some new libraries:
# pip install streamlit pandas numpy yfinance plotly scipy
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import optimize
import base64
import plotly.io as pio
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/shyam-sundar-837b97192/',
        'Report a bug': "https://github.com/thisisgss",
        'About': """
        ### 🏆 Portfolio Optimizer Pro

        This application provides a comprehensive suite of tools for portfolio optimization, personalization, and analysis.

        Built by Shyam Sundar, Enhanced by Gemini.
        """
    }
)

# --- Initialize Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}
if 'saved_portfolios' not in st.session_state:
    st.session_state.saved_portfolios = []

# --- FINANCIAL CALCULATION CORE ---

@st.cache_data
def get_stock_data(symbols, start_date, end_date):
    symbols = sorted(list(set(symbols)))
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    data.dropna(axis=1, how='all', inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    valid_symbols = data.columns.tolist()
    invalid_symbols = [s for s in symbols if s not in valid_symbols]
    ticker_to_name = {s: yf.Ticker(s).info.get('shortName', s) for s in valid_symbols}
    return data, valid_symbols, invalid_symbols, ticker_to_name

@st.cache_data
def get_currency_info(symbols):
    is_usd = any(not s.endswith(('.NS', '.BO')) for s in symbols)
    if is_usd:
        try:
            rate = yf.Ticker("USDINR=X").history(period='1d')['Close'].iloc[-1]
            return "USD", "$", rate
        except Exception:
            st.warning("Could not fetch live currency conversion rate. Using a fallback rate of 83.0.")
            return "USD", "$", 83.0  # Fallback rate
    return "INR", "₹", 1.0

def run_monte_carlo_simulation(daily_returns, num_simulations, risk_free_rate):
    num_assets = len(daily_returns.columns)
    results = []
    for _ in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(daily_returns.mean() * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        result = {"Expected Return": portfolio_return, "Risk (Std Dev)": portfolio_std_dev, "Sharpe Ratio": sharpe_ratio}
        for i, symbol in enumerate(daily_returns.columns):
            result[symbol] = weights[i]
        results.append(result)
    return pd.DataFrame(results)

def run_constrained_optimization(daily_returns, risk_free_rate, bounds):
    num_assets = len(daily_returns.columns)
    def negative_sharpe(weights):
        p_return = np.sum(daily_returns.mean() * weights) * 252
        p_std_dev = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
        if p_std_dev == 0: return -np.inf
        return -((p_return - risk_free_rate) / p_std_dev)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    initial_guess = np.array(num_assets * [1. / num_assets])
    result = optimize.minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    optimal_return = np.sum(daily_returns.mean() * optimal_weights) * 252
    optimal_std_dev = np.sqrt(np.dot(optimal_weights.T, np.dot(daily_returns.cov() * 252, optimal_weights)))
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_std_dev
    portfolio_data = {"Expected Return": optimal_return, "Risk (Std Dev)": optimal_std_dev, "Sharpe Ratio": optimal_sharpe}
    for i, symbol in enumerate(daily_returns.columns):
        portfolio_data[symbol] = optimal_weights[i]
    return pd.Series(portfolio_data)

def calculate_advanced_metrics(daily_returns, weights):
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    var_95 = portfolio_returns.quantile(0.05)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return var_95, max_drawdown

# --- PDF REPORT GENERATION (REMOVED) ---
# The PDF class and generate_pdf_report function are removed.

# --- UI RENDERING FUNCTIONS ---
def render_main_page():
    with st.sidebar:
        st.image("https://cdn.pixabay.com/photo/2024/09/23/23/38/piggy-bank-9070156_1280.jpg", width=200)
        st.markdown("<h1 style='text-align: left;'>Optimizer Pro</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 🗂️ Saved Portfolios")
        if not st.session_state.saved_portfolios:
            st.info("No portfolios saved yet.")
        else:
            for i, p in enumerate(st.session_state.saved_portfolios):
                st.markdown(f"**Portfolio {i+1}:** {', '.join(p['symbols'])}")
            if st.button("Compare Saved Portfolios", use_container_width=True, key="compare_btn"):
                st.session_state.page = 'compare'
                st.rerun()
            if st.button("Clear All Portfolios", use_container_width=True, key="clear_btn"):
                st.session_state.saved_portfolios = []
                st.rerun()
        st.markdown("---")

        st.markdown("### ⚙️ Portfolio Controls")
        num_stocks = st.number_input("Number of Stocks", min_value=2, max_value=10, value=4, key='num_stocks')
        stock_symbols_list = [st.text_input(f"Stock Ticker {i+1}", key=f'stock_{i}').strip().upper() for i in range(num_stocks)]
        append_ns = st.toggle('Append ".NS" for Indian Stocks (NSE)', value=True)
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        with col2: end_date = st.date_input("End Date", datetime.now())
        
        with st.expander("✨ Advanced Options"):
            use_constraints = st.toggle("Enable Weight Constraints")
            bounds = []
            if use_constraints:
                st.info("Set min/max allocation for each stock.")
                for sym in stock_symbols_list:
                    if sym:
                        c1, c2 = st.columns(2)
                        min_w = c1.slider(f"Min % for {sym}", 0, 100, 0, key=f"min_{sym}") / 100
                        max_w = c2.slider(f"Max % for {sym}", 0, 100, 100, key=f"max_{sym}") / 100
                        bounds.append((min_w, max_w))
            else:
                num_valid_stocks = len([s for s in stock_symbols_list if s])
                bounds = [(0,1)] * num_valid_stocks

        num_simulations = st.number_input("Number of Simulations (for Monte Carlo)", 1000, 100000, 10000, 1000, disabled=use_constraints)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 7.0, 0.1) / 100
        run_button = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    st.title("🏆 Portfolio Optimizer Pro")
    st.markdown("Define your portfolio, set optional constraints, and run the optimization.")

    if run_button:
        final_symbols = sorted(list(set([s for s in stock_symbols_list if s])))
        if append_ns:
            final_symbols = [s + ".NS" if not s.endswith((".NS", ".BO")) else s for s in final_symbols]
        if len(final_symbols) < 2:
            st.warning("Please enter at least two unique, valid stock tickers.")
            st.stop()

        with st.spinner("Fetching data..."):
            stock_data, valid_symbols, invalid_symbols, ticker_to_name = get_stock_data(final_symbols, start_date, end_date)
        if invalid_symbols: st.warning(f"Could not fetch data for: {', '.join(invalid_symbols)}.")
        if len(valid_symbols) < 2:
            st.error("Not enough valid stock data to perform optimization.")
            st.stop()
        
        daily_returns = stock_data[valid_symbols].pct_change().dropna()
        if len(daily_returns) < len(valid_symbols):
            st.error(f"Insufficient historical data for the selected date range. Please select a longer date range.")
            st.stop()

        if use_constraints:
            with st.spinner("Running constrained optimization..."):
                max_sharpe_portfolio = run_constrained_optimization(daily_returns, risk_free_rate, bounds)
                sim_results_df = None
        else:
            with st.spinner("Running Monte Carlo simulations..."):
                sim_results_df = run_monte_carlo_simulation(daily_returns, num_simulations, risk_free_rate)
                max_sharpe_portfolio = sim_results_df.loc[sim_results_df['Sharpe Ratio'].idxmax()]

        min_vol_portfolio = sim_results_df.loc[sim_results_df['Risk (Std Dev)'].idxmin()] if sim_results_df is not None else max_sharpe_portfolio
        var_95, max_drawdown = calculate_advanced_metrics(daily_returns, max_sharpe_portfolio[valid_symbols])

        st.session_state.portfolio_data = {
            'max_sharpe_portfolio': max_sharpe_portfolio, 'min_vol_portfolio': min_vol_portfolio,
            'sim_results_df': sim_results_df, 'valid_symbols': valid_symbols,
            'ticker_to_name': ticker_to_name, 'risk_free_rate': risk_free_rate,
            'var_95': var_95, 'max_drawdown': max_drawdown,
            'daily_returns': daily_returns, 'start_date': start_date, 'end_date': end_date
        }
        st.success("Optimization Complete!")

    if st.session_state.portfolio_data:
        data = st.session_state.portfolio_data
        max_sharpe_portfolio = data['max_sharpe_portfolio']
        return_value = max_sharpe_portfolio['Expected Return']
        return_color = "green" if return_value > data['risk_free_rate'] else "red"
        
        st.markdown("---")
        st.header("📊 General Optimal Portfolio Results")

        m_cols = st.columns(5)
        with m_cols[0]: st.markdown(f'<div class="metric-container"><div class="metric-label">Optimal Return</div><div class="metric-value" style="color:{return_color};">{return_value:.2%}</div></div>', unsafe_allow_html=True)
        with m_cols[1]: st.markdown(f'<div class="metric-container"><div class="metric-label">Volatility (Risk)</div><div class="metric-value" style="color:orange;">{max_sharpe_portfolio["Risk (Std Dev)"]:.2%}</div></div>', unsafe_allow_html=True)
        with m_cols[2]: st.markdown(f'<div class="metric-container"><div class="metric-label">Sharpe Ratio</div><div class="metric-value" style="color:violet;">{max_sharpe_portfolio["Sharpe Ratio"]:.2f}</div></div>', unsafe_allow_html=True)
        with m_cols[3]: st.markdown(f'<div class="metric-container"><div class="metric-label">95% VaR (Daily)</div><div class="metric-value" style="color:#add8e6;">{data["var_95"]:.2%}</div></div>', unsafe_allow_html=True)
        with m_cols[4]: st.markdown(f'<div class="metric-container"><div class="metric-label">Max Drawdown</div><div class="metric-value" style="color:#E57373;">{data["max_drawdown"]:.2%}</div></div>', unsafe_allow_html=True)

        b_cols = st.columns([1,1])
        with b_cols[0]:
            if st.button("💾 Save this Portfolio", use_container_width=True):
                saved_item = {
                    'name': f"Portfolio {len(st.session_state.saved_portfolios) + 1}",
                    'symbols': data['valid_symbols'],
                    'metrics': {
                        'Return': max_sharpe_portfolio['Expected Return'],
                        'Risk': max_sharpe_portfolio['Risk (Std Dev)'],
                        'Sharpe': max_sharpe_portfolio['Sharpe Ratio'],
                        'VaR': data['var_95'],
                        'Max Drawdown': data['max_drawdown']
                    }
                }
                st.session_state.saved_portfolios.append(saved_item)
                st.toast(f"✅ Saved '{saved_item['name']}'!")
        with b_cols[1]:
            if st.button("👉 Go to Personalize This Portfolio", type="primary", use_container_width=True):
                st.session_state.page = 'questionnaire'
                st.rerun()

        tab1, tab2, tab3, tab4 = st.tabs(["Efficient Frontier", "Optimal Allocation", "Asset Correlations", "🎓 Key Concepts"])
        with tab1:
            if data['sim_results_df'] is not None:
                fig = px.scatter(data['sim_results_df'], x='Risk (Std Dev)', y='Expected Return', color='Sharpe Ratio', color_continuous_scale=px.colors.sequential.Viridis, title='Efficient Frontier: Risk vs. Return')
                fig.add_trace(go.Scatter(x=[data['max_sharpe_portfolio']['Risk (Std Dev)']], y=[data['max_sharpe_portfolio']['Expected Return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe Ratio'))
                fig.add_trace(go.Scatter(x=[data['min_vol_portfolio']['Risk (Std Dev)']], y=[data['min_vol_portfolio']['Expected Return']], mode='markers', marker=dict(color='cyan', size=15, symbol='diamond'), name='Minimum Volatility'))
                fig.update_layout(height=600, xaxis_tickformat=".2%", yaxis_tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.portfolio_data['frontier_fig'] = fig
            else:
                st.info("Efficient Frontier plot is only available for Monte Carlo simulation (when constraints are disabled).")

        with tab2:
            st.subheader("Optimal Portfolio Allocation (Max Sharpe Ratio)")
            weights_df = data['max_sharpe_portfolio'][data['valid_symbols']][data['max_sharpe_portfolio'][data['valid_symbols']] > 0.001]
            weights_df.index = [data['ticker_to_name'].get(t, t) for t in weights_df.index]
            pie_fig = px.pie(weights_df, values=weights_df.values, names=weights_df.index, title='Asset Allocation', hole=.3)
            st.plotly_chart(pie_fig, use_container_width=True)
            st.session_state.portfolio_data['pie_fig'] = pie_fig
            
            st.markdown("##### Detailed Weights")
            optimal_weights_table = pd.DataFrame(data['max_sharpe_portfolio'][data['valid_symbols']])
            optimal_weights_table.columns = ['Weight']
            optimal_weights_table.index = [data['ticker_to_name'].get(ticker, ticker) for ticker in optimal_weights_table.index]
            st.dataframe(optimal_weights_table.style.format("{:.2%}").background_gradient(cmap='Greens'), use_container_width=True)

        with tab3:
            st.subheader("Correlation Matrix of Assets")
            daily_returns_corr = data['daily_returns'].corr()
            daily_returns_corr.columns = [data['ticker_to_name'].get(t,t) for t in daily_returns_corr.columns]
            daily_returns_corr.index = [data['ticker_to_name'].get(t,t) for t in daily_returns_corr.index]
            fig_corr = px.imshow(daily_returns_corr, text_auto=True, color_continuous_scale='RdYlGn_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab4:
            st.subheader("Learning Corner: Understanding the Basics")
            st.markdown("""
            #### What is a Portfolio?
            A portfolio is simply a collection of financial investments like stocks, bonds, and cash. The goal of building a portfolio is to combine different assets to achieve your financial objectives, usually by balancing risk and return.

            #### The Markowitz Efficient Frontier
            This is the core concept behind our optimizer. The Efficient Frontier is a curve on a graph that represents the set of "optimal" portfolios. For any given level of risk, the portfolio on the frontier offers the highest possible expected return. Conversely, for any given level of return, the portfolio on the frontier has the lowest possible risk. Our app runs thousands of simulations to find this frontier for your chosen stocks.

            #### Sharpe Ratio
            This is arguably the most important metric for evaluating a portfolio. It tells you how much return you are getting for each unit of risk you take. A higher Sharpe Ratio is always better. The "Optimal Portfolio" our app calculates is the one with the highest possible Sharpe Ratio.
            - **Formula:** `(Portfolio Return - Risk-Free Rate) / Portfolio Risk`

            #### Volatility (Standard Deviation)
            In finance, volatility is the most common way to measure risk. It quantifies how much the price of an asset or a portfolio fluctuates over time. A stock with high volatility will have large price swings, making it riskier. A portfolio with low volatility is more stable.

            #### Correlation
            Correlation measures how two assets move in relation to each other. It ranges from -1 to +1.
            - **+1 (Perfect Positive Correlation):** The assets move in perfect sync. This is bad for diversification.
            - **0 (No Correlation):** The assets' movements are random and unrelated.
            - **-1 (Perfect Negative Correlation):** The assets move in opposite directions. This is the holy grail for diversification, as one asset zigs while the other zags, smoothing out your returns.
            
            A well-diversified portfolio combines assets with low or negative correlation to each other to reduce overall risk.
            """)

def render_questionnaire():
    st.header("👤 Personalize Your Portfolio")
    st.markdown("Answer these questions to determine your risk profile.")

    with st.form("risk_assessment_form"):
        q1 = st.radio(
            "**1. What is your primary investment goal?**",
            ('Capital Preservation: I want to protect my money from losses.',
             'Balanced Growth: I want a mix of safety and returns.',
             'Aggressive Growth: I want to maximize my returns, and I\'m comfortable with higher risk.'),
            index=1
        )
        q2 = st.radio(
            "**2. If your portfolio lost 20% of its value in a month, how would you react?**",
            ('Sell some or all of my investments to cut my losses.',
             'Do nothing and wait for the market to recover.',
             'Invest more money, seeing it as a buying opportunity.'),
            index=1
        )
        q3 = st.radio(
            "**3. What is your investment time horizon?**",
            ('Short-term (Less than 3 years)',
             'Medium-term (3 to 10 years)',
             'Long-term (More than 10 years)'),
            index=1
        )
        q4 = st.radio(
            "**4. How would you describe your knowledge of investments?**",
            ('Beginner: I have limited knowledge.',
             'Intermediate: I have a good understanding of the basics.',
             'Advanced: I am a very experienced investor.'),
            index=1
        )
        submitted = st.form_submit_button("✅ Assess My Risk Profile", use_container_width=True)

    if submitted:
        score = 0
        score += {'Capital Preservation': 1, 'Balanced Growth': 2, 'Aggressive Growth': 3}[q1.split(':')[0]]
        if q2.startswith('Sell some'): score += 1
        elif q2.startswith('Do nothing'): score += 2
        elif q2.startswith('Invest more'): score += 3
        score += {'Short-term': 1, 'Medium-term': 2, 'Long-term': 3}[q3.split(' ')[0]]
        score += {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[q4.split(':')[0]]
        st.session_state.portfolio_data['risk_score'] = score
        st.session_state.page = 'personalized_results'
        st.rerun()

    if st.button("⬅️ Back to Main Results"):
        st.session_state.page = 'main'
        st.rerun()

def render_personalized_results():
    st.header("✨ Your Personalized Portfolio Allocation")
    data = st.session_state.portfolio_data
    score = data.get('risk_score', 0)
    
    if score <= 6:
        profile = "Conservative (Risk Averse)"
        A = 4.0
    elif score <= 9:
        profile = "Moderate (Risk Neutral)"
        A = 2.5
    else:
        profile = "Aggressive (Risk Seeker)"
        A = 1.0

    st.info(f"Based on your answers, you have a **{profile}** risk profile.", icon="🎯")
    
    E_r = data['max_sharpe_portfolio']['Expected Return']
    sigma_sq = data['max_sharpe_portfolio']['Risk (Std Dev)']**2
    rf = data['risk_free_rate']
    y_star = (E_r - rf) / (A * sigma_sq)
    y_star = max(0, min(y_star, 1.0))

    personalized_weights = data['max_sharpe_portfolio'][data['valid_symbols']] * y_star
    personalized_weights['Risk-Free Asset'] = 1 - y_star
    st.session_state.portfolio_data['personalized_weights'] = personalized_weights

    st.subheader("Personalized Asset Allocation")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        fig_pie = px.pie(
            values=personalized_weights.values,
            names=[data['ticker_to_name'].get(i, i) for i in personalized_weights.index],
            title='Your Personalized Portfolio Mix',
            hole=.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("##### Detailed Personalized Weights")
        p_weights_df = pd.DataFrame(personalized_weights)
        p_weights_df.columns = ['Weight']
        p_weights_df.index = [data['ticker_to_name'].get(i, i) for i in p_weights_df.index]
        st.dataframe(
            p_weights_df.style.format("{:.2%}")
            .background_gradient(cmap='Blues'),
            height=380,
            use_container_width=True
        )

    if st.button("💰 Proceed to Investment Plan", type="primary", use_container_width=True):
        st.session_state.page = 'investment_plan'
        st.rerun()

def render_investment_plan():
    st.header("💰 Investment Plan")
    data = st.session_state.portfolio_data

    currency, currency_symbol, conversion_rate = get_currency_info(data['valid_symbols'])
    st.session_state.portfolio_data['currency_symbol'] = currency_symbol

    investment_amount = st.number_input(
        f"Enter your total investment amount (in ₹):",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
        help="Enter the total capital you wish to invest across this personalized portfolio."
    )

    investment_amount_converted = investment_amount
    if currency == 'USD':
        investment_amount_converted = investment_amount / conversion_rate
        st.info(f"US stocks detected. Your investment of ₹{investment_amount:,.2f} will be converted to **${investment_amount_converted:,.2f}** (at a rate of {conversion_rate:.2f} INR/USD).")

    if investment_amount and 'personalized_weights' in data:
        with st.spinner("Fetching latest prices for allocation..."):
            try:
                latest_prices = yf.download(data['valid_symbols'], period='1d', progress=False)['Close'].iloc[-1]
                
                allocation_df = pd.DataFrame(index=data['personalized_weights'].index)
                allocation_df['Personalized Weight'] = data['personalized_weights']
                allocation_df['Amount to Invest'] = allocation_df['Personalized Weight'] * investment_amount_converted
                
                allocation_df['Latest Price'] = latest_prices.reindex(allocation_df.index)
                allocation_df.loc['Risk-Free Asset', 'Latest Price'] = 1.0

                allocation_df['Number of Shares'] = (allocation_df['Amount to Invest'] / allocation_df['Latest Price']).apply(np.floor)
                allocation_df.loc['Risk-Free Asset', 'Number of Shares'] = allocation_df.loc['Risk-Free Asset', 'Amount to Invest']

                invested_amount_stocks = (allocation_df.drop('Risk-Free Asset')['Number of Shares'] * allocation_df.drop('Risk-Free Asset')['Latest Price']).sum()
                initial_risk_free_amount = allocation_df.loc['Risk-Free Asset', 'Amount to Invest']
                leftover_cash = investment_amount_converted - invested_amount_stocks - initial_risk_free_amount
                
                st.markdown("---")
                st.subheader("Investment Breakdown")

                if leftover_cash > 0.01:
                    reinvest_choice = st.radio(
                        f"You have **{currency_symbol}{leftover_cash:,.2f}** leftover from rounding shares. What should be done?",
                        ('Add to Risk-Free Asset', 'Keep as uninvested cash'),
                        key='reinvest_choice', horizontal=True
                    )
                    if reinvest_choice == 'Add to Risk-Free Asset':
                        allocation_df.loc['Risk-Free Asset', 'Amount to Invest'] += leftover_cash
                        leftover_cash = 0
                
                display_df = allocation_df.copy()
                display_df.index = [data['ticker_to_name'].get(i, i) for i in display_df.index]
                display_df.rename(columns={'Number of Shares': 'Number of Shares (Rounded)'}, inplace=True)
                st.session_state.portfolio_data['investment_breakdown'] = display_df[['Personalized Weight', 'Amount to Invest', 'Latest Price', 'Number of Shares (Rounded)']]

                st.dataframe(
                    display_df[['Personalized Weight', 'Amount to Invest', 'Latest Price', 'Number of Shares (Rounded)']].style
                    .format({
                        'Personalized Weight': '{:.2%}',
                        'Amount to Invest': f'{currency_symbol}{{:,.2f}}',
                        'Latest Price': f'{currency_symbol}{{:,.2f}}',
                        'Number of Shares (Rounded)': '{:,.0f}'
                    })
                    .background_gradient(cmap='Greens', subset=['Amount to Invest']),
                    use_container_width=True
                )

                with st.expander("Important Considerations", expanded=True):
                    st.markdown(f"""
                    - **Risk-Free Asset**: The amount allocated here should be invested in low-risk instruments like government bonds or fixed deposits.
                    - **Uninvested Cash**: You have **{currency_symbol}{leftover_cash:,.2f}** remaining uninvested.
                    - **Execution**: Stock prices are dynamic. The 'Latest Price' is from the last trading day.
                    """)
            except Exception as e:
                st.error(f"An error occurred while fetching latest prices: {e}")
                st.warning("Could not generate the investment plan.")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back"):
            st.session_state.page = 'personalized_results'
            st.rerun()
    with c2:
        if st.button("📈 Go to Backtesting", type="primary", use_container_width=True):
            st.session_state.page = 'backtesting'
            st.rerun()

def render_backtesting_page():
    st.header("⏳ Historical Performance Backtest")
    data = st.session_state.portfolio_data
    if 'personalized_weights' not in data:
        st.warning("Please personalize your portfolio first.")
        st.stop()
    
    with st.spinner("Running backtest..."):
        weights = data['personalized_weights']
        risky_weights = weights.drop('Risk-Free Asset', errors='ignore')
        
        y_star = risky_weights.sum()
        risky_weights_normalized = risky_weights / y_star if y_star > 0 else risky_weights

        risky_portfolio_returns = (data['daily_returns'][risky_weights_normalized.index] * risky_weights_normalized).sum(axis=1)
        portfolio_returns = risky_portfolio_returns * y_star + (1 - y_star) * (data['risk_free_rate'] / 252)
        portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
        
        benchmark_ticker = st.text_input("Enter a Benchmark Ticker (e.g., ^NSEI for NIFTY 50)", "^NSEI")
        if benchmark_ticker:
            benchmark_data = yf.download(benchmark_ticker, start=data['start_date'], end=data['end_date'], progress=False)
            
            if benchmark_data.empty:
                st.error(f"No data found for benchmark ticker: {benchmark_ticker}")
                st.stop()
            
            benchmark_close = benchmark_data['Close']
            if isinstance(benchmark_close, pd.DataFrame): 
                benchmark_close = benchmark_close.iloc[:, 0]
            
            benchmark_returns = benchmark_close.pct_change()
            benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
            
            df_plot = pd.DataFrame({
                'Your Personalized Portfolio': portfolio_cumulative_returns,
                'Benchmark': benchmark_cumulative_returns
            }).dropna()
            
            initial_investment = 100
            df_plot *= initial_investment

            fig = px.line(df_plot, title=f"Portfolio Growth vs. Benchmark ({st.session_state.portfolio_data.get('currency_symbol', '₹')}{initial_investment} Invested)")
            fig.update_layout(height=600, yaxis_title="Cumulative Growth", legend_title="Asset", yaxis_tickprefix=st.session_state.portfolio_data.get('currency_symbol', '₹'))
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.portfolio_data['backtest_fig'] = fig

    if st.button("⬅️ Back to Investment Plan"):
        st.session_state.page = 'investment_plan'
        st.rerun()


def render_compare_page():
    st.header("⚖️ Compare Saved Portfolios")
    if not st.session_state.saved_portfolios:
        st.info("You haven't saved any portfolios yet. Run an optimization and click 'Save'.")
        st.stop()
        
    metrics_data = []
    for p in st.session_state.saved_portfolios:
        row = p['metrics'].copy()
        row['Name'] = p['name']
        row['Symbols'] = ", ".join(p['symbols'])
        metrics_data.append(row)
    
    compare_df = pd.DataFrame(metrics_data).set_index("Name")
    
    cols_order = ['Symbols', 'Return', 'Risk', 'Sharpe', 'VaR', 'Max Drawdown']
    compare_df = compare_df[cols_order]

    st.dataframe(compare_df.style.format({
        'Return': '{:.2%}', 'Risk': '{:.2%}', 'Sharpe': '{:.2f}',
        'VaR': '{:.2%}', 'Max Drawdown': '{:.2%}'
    }).background_gradient(cmap='viridis_r', subset=['Sharpe']), use_container_width=True)

    if st.button("⬅️ Back to Main Page"):
        st.session_state.page = 'main'
        st.rerun()

# --- PAGE ROUTER ---
page_functions = {
    'main': render_main_page, 'questionnaire': render_questionnaire,
    'personalized_results': render_personalized_results, 'investment_plan': render_investment_plan,
    'backtesting': render_backtesting_page, 'compare': render_compare_page
}

st.markdown("""<style>.metric-container{border:1px solid rgba(255,25.5,25f,0.2); border-radius:0.5rem; padding:1.5rem; background-color:#1a1a1a; text-align:center; margin-bottom:1rem; height:100%;} .metric-label{font-size:1rem; font-weight:500; color:#a0a0a0;} .metric-value{font-size:2.25rem; font-weight:700; letter-spacing:-1px;}</style>""", unsafe_allow_html=True)

if st.session_state.page in page_functions:
    page_functions[st.session_state.page]()
else:
    st.session_state.page = 'main'
    page_functions['main']()
