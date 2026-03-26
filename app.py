import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import os
import json
import copy

# Page configuration
st.set_page_config(page_title="Stock Analysis App", layout="wide")

st.title("Stock Analysis")

# --- CONFIGURATION & PERSISTENCE ---
CONFIG_FILE = "config.json"

def load_config():
    """Loads configuration from config.json."""
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            config['dividend_analysis']['start_date'] = datetime.strptime(config['dividend_analysis']['start_date'], '%Y-%m-%d').date()
            config['pe_analysis']['start_date'] = datetime.strptime(config['pe_analysis']['start_date'], '%Y-%m-%d').date()
            return config
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {
            "dividend_analysis": {"tickers": ["VOO", "BKLN", "JEPI"], "start_date": date.today() - timedelta(days=365)},
            "pe_analysis": {"tickers": ["MSFT", "AAPL"], "start_date": date.today() - timedelta(days=365*5)}
        }

def save_config(config):
    """Saves the current configuration to config.json."""
    config_to_save = copy.deepcopy(config)
    config_to_save['dividend_analysis']['start_date'] = config['dividend_analysis']['start_date'].strftime('%Y-%m-%d')
    config_to_save['pe_analysis']['start_date'] = config['pe_analysis']['start_date'].strftime('%Y-%m-%d')
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_to_save, f, indent=4)

# --- ANALYSIS FUNCTIONS (with caching) ---
@st.cache_data
def get_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info or 'regularMarketPrice' not in info or info.get('regularMarketPrice') is None: return None, None
    return info, stock.quarterly_income_stmt

@st.cache_data
def get_history(ticker, start_date, end_date):
    return yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)

@st.cache_data
def get_daily_price(ticker):
    price = yf.Ticker(ticker).history(period="max", auto_adjust=False)["Close"].dropna()
    price.index = pd.to_datetime(price.index).tz_localize(None)
    return price

# --- PLOTTING & CALCULATION LOGIC ---

def ps_ratio_analysis(ticker, plot_start_date):
    info, q_inc = get_info(ticker)
    price = get_daily_price(ticker)
    if q_inc is None or q_inc.empty: raise ValueError("Quarterly income statement data not available.")
    
    q_inc.columns = pd.to_datetime(q_inc.columns).tz_localize(None)
    revenue = next((q_inc.loc[r] for r in ["Total Revenue", "Revenue"] if r in q_inc.index), None)
    if revenue is None: raise ValueError("Could not find Revenue row.")

    shares = info.get("sharesOutstanding", 0)
    if shares == 0: raise ValueError("Share count is zero.")

    rps_q = (revenue / shares).sort_index()
    rps_ttm = rps_q.rolling(window=4, min_periods=1).mean() * 4
    rps_ttm_daily = rps_ttm.reindex(price.index, method='ffill')
    ps = (price / rps_ttm_daily).dropna()

    price_plot, ps_plot, revenue_plot = price.loc[plot_start_date:], ps.loc[plot_start_date:], rps_q.loc[plot_start_date:]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax_ps = plt.subplots(figsize=(12, 6))

    line_ps, = ax_ps.plot(ps_plot.index, ps_plot.values, color="tab:blue", lw=1, label="Trailing P/S", zorder=3)
    ax_ps.set_ylabel("P/S Ratio", color="tab:blue", fontsize=12)
    ax_ps.tick_params(axis="y", labelcolor="tab:blue")
    ax_ps.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', zorder=0)

    ax_price = ax_ps.twinx()
    ax_price.spines["left"].set_position(("axes", -0.08)); ax_price.spines["left"].set_visible(True)
    ax_price.spines["right"].set_visible(False); ax_price.yaxis.set_label_position("left"); ax_price.yaxis.set_ticks_position("left")
    line_price, = ax_price.plot(price_plot.index, price_plot.values, color="tab:green", lw=1, label="Price", zorder=3)
    ax_price.set_ylabel("Price", color="tab:green", fontsize=12)
    ax_price.tick_params(axis="y", labelcolor="tab:green")
    
    ax_rev = ax_ps.twinx()
    bars_rev = ax_rev.bar(revenue_plot.index, revenue_plot.values, width=10, alpha=0.3, color="tab:orange", label="Quarterly Revenue", zorder=2)
    ax_rev.set_ylabel("Quarterly Revenue", color="tab:orange", fontsize=12)
    ax_rev.tick_params(axis="y", labelcolor="tab:orange")
    
    ax_ps.set_title(f"{info.get('shortName', ticker)} ({ticker}) - Price, Trailing P/S, and Quarterly Revenue", fontsize=16)
    handles = [line_price, line_ps, bars_rev]
    legend = ax_ps.legend(handles, [h.get_label() for h in handles], loc="upper left")
    legend.set_zorder(10)
    fig.tight_layout()
    return fig

def pe_ratio_analysis(ticker, plot_start_date):
    info, q_inc = get_info(ticker)
    price = get_daily_price(ticker)
    if q_inc is None or q_inc.empty: raise ValueError("Quarterly income statement data not available.")
    
    q_inc.columns = pd.to_datetime(q_inc.columns).tz_localize(None)
    net_income = next((q_inc.loc[r] for r in ["Net Income", "Net Income Common Stockholders"] if r in q_inc.index), None)
    if net_income is None: raise ValueError("Could not find Net Income row.")
    
    shares = next((q_inc.loc[r] for r in ["Diluted Average Shares", "Basic Average Shares"] if r in q_inc.index), None)
    if shares is None: shares = info.get("sharesOutstanding", 0)

    eps_q = (net_income / shares).sort_index()
    eps_ttm = eps_q.rolling(window=4, min_periods=1).mean() * 4
    eps_ttm_daily = eps_ttm.reindex(price.index, method="ffill")
    pe = (price / eps_ttm_daily).dropna()

    price_plot, pe_plot, eps_q_plot = price.loc[plot_start_date:], pe.loc[plot_start_date:], eps_q.loc[plot_start_date:]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax_pe = plt.subplots(figsize=(12, 6))

    line_pe, = ax_pe.plot(pe_plot.index, pe_plot.values, color="tab:blue", lw=1, label="Trailing P/E", zorder=3)
    ax_pe.set_ylabel("P/E Ratio", color="tab:blue", fontsize=12)
    ax_pe.tick_params(axis="y", labelcolor="tab:blue")
    ax_pe.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', zorder=0)

    ax_price = ax_pe.twinx()
    ax_price.spines["left"].set_position(("axes", -0.08)); ax_price.spines["left"].set_visible(True)
    ax_price.spines["right"].set_visible(False); ax_price.yaxis.set_label_position("left"); ax_price.yaxis.set_ticks_position("left")
    line_price, = ax_price.plot(price_plot.index, price_plot.values, color="tab:green", lw=1, label="Price", zorder=3)
    ax_price.set_ylabel("Price", color="tab:green", fontsize=12)
    ax_price.tick_params(axis="y", labelcolor="tab:green")
    
    ax_eps = ax_pe.twinx()
    bars_eps = ax_eps.bar(eps_q_plot.index, eps_q_plot.values, width=10, alpha=0.3, color="tab:orange", label="Quarterly EPS", zorder=2)
    ax_eps.set_ylabel("Quarterly EPS", color="tab:orange", fontsize=12)
    ax_eps.tick_params(axis="y", labelcolor="tab:orange")
    
    ax_pe.set_title(f"{info.get('shortName', ticker)} ({ticker}) - Price, Trailing P/E, and Quarterly EPS", fontsize=16)
    handles = [line_pe, line_price, bars_eps]
    legend = ax_pe.legend(handles, [h.get_label() for h in handles], loc="upper left")
    legend.set_zorder(10)
    fig.tight_layout()
    return fig

def adj_close_analysis(ticker, start_date):
    info, _ = get_info(ticker)
    if not info: return None, None
    
    # Use date.today() as the end date for fetching history
    end_date_fetch = date.today()
    df = get_history(ticker, start_date, end_date_fetch)
    if df.empty: return None, None

    # Determine actual start and end dates from the fetched data
    actual_start_date = df.index.min().strftime('%Y-%m-%d')
    actual_end_date = df.index.max().strftime('%Y-%m-%d')

    fundamentals_df = pd.DataFrame({"Price": info.get("regularMarketPrice"), "Dividend Yield (%)": info.get("dividendYield", 0), "Trailing P/E": info.get("trailingPE"), "Asset Type": info.get("quoteType")}.items(), columns=["Field", ticker]).set_index("Field")
    
    returns = df[['Close', 'Adj Close']].pct_change().fillna(0.0)
    return_1y = returns.mean() * 252
    fundamentals_df.loc['Start Date', ticker] = actual_start_date
    fundamentals_df.loc['End Date', ticker] = actual_end_date
    fundamentals_df.loc['Annual Return (%)', ticker] = return_1y["Close"] * 100
    fundamentals_df.loc['Annual Return Adj (%)', ticker] = return_1y["Adj Close"] * 100
    stdev_1y = returns.std() * np.sqrt(252)
    fundamentals_df.loc['Annual Volatility (%)', ticker] = stdev_1y["Close"] * 100
    sharpe = (return_1y - 0.03) / stdev_1y
    fundamentals_df.loc['Sharpe Ratio', ticker] = sharpe["Close"]
    fundamentals_df.loc['Sharpe Ratio Adj', ticker] = sharpe["Adj Close"]

    df['Adj Close'] += (df['Close'].iloc[0] - df['Adj Close'].iloc[0])
    
    dividends_to_plot = df[df['Dividends'] > 0]
    div_1y_count = len(df[(df.index >= df.index.max() - pd.DateOffset(years=1)) & (df["Dividends"] > 0)])
    bar_labels = [f'{(d * 100 * div_1y_count) / c:.2f}%' for d, c in zip(dividends_to_plot['Dividends'], dividends_to_plot['Close'])] if div_1y_count > 0 else []

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df.index, df['Close'], label='Close', color='royalblue', lw=1, zorder=3)
    ax1.plot(df.index, df['Adj Close'], label='Adj Close (rebased)', color='orange', linestyle='-', lw=1, zorder=3)
    ax1.set_title(f"{info.get('shortName', ticker)} ({ticker}) - Adjusted Close and Dividends", fontsize=16)
    ax1.set_xlabel("Date", fontsize=12); ax1.set_ylabel("Close Price", fontsize=12)
    
    legend = ax1.legend(loc="upper left"); legend.set_zorder(10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', zorder=0)

    ax2 = ax1.twinx()
    bar = ax2.bar(dividends_to_plot.index, dividends_to_plot['Dividends'], color='green', alpha=0.4, label='Dividends', width=5, zorder=2)
    if bar_labels: ax2.bar_label(bar, labels=bar_labels, label_type='edge', fontsize=9, padding=3)
    ax2.set_ylabel('Dividend Amount ($)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green'); ax2.set_ylim(bottom=0)

    fig.tight_layout()
    return fundamentals_df, fig

def format_df_for_display(df):
    """Creates a copy of a dataframe and formats all its cells into strings for safe display in Streamlit."""
    display_df = df.copy()
    for col in display_df.columns:
        for idx in display_df.index:
            value = display_df.loc[idx, col]
            if isinstance(value, (int, float)):
                display_df.loc[idx, col] = f'{value:.2f}' if not pd.isna(value) else 'N/A'
            elif value is None:
                display_df.loc[idx, col] = 'N/A'
            else:
                display_df.loc[idx, col] = str(value)
    return display_df

def get_display_fundamentals(ticker, start_date):
    """Helper to get a formatted fundamentals table for a single ticker."""
    info, _ = get_info(ticker)
    if not info: return None

    end_date_fetch = date.today()
    df = get_history(ticker, start_date, end_date_fetch)
    if df.empty:
        actual_start_date = "N/A"
        actual_end_date = "N/A"
    else:
        actual_start_date = df.index.min().strftime('%Y-%m-%d')
        actual_end_date = df.index.max().strftime('%Y-%m-%d')

    funda_data = {
        "Price": info.get("regularMarketPrice"), 
        "Dividend Yield (%)": info.get("dividendYield", 0), 
        "Trailing P/E": info.get("trailingPE"), 
        "Asset Type": info.get("quoteType"),
        "Start Date": actual_start_date,
        "End Date": actual_end_date
    }
    
    funda_df = pd.DataFrame(funda_data.items(), columns=["Field", "Value"]).set_index("Field")
    
    return format_df_for_display(funda_df)

# --- UI & APP LOGIC ---
config = load_config()

if 'div_results' not in st.session_state: st.session_state.div_results = None
if 'pe_results' not in st.session_state: st.session_state.pe_results = None
if 'pe_tickers_list' not in st.session_state: st.session_state.pe_tickers_list = config.get('pe_analysis', {}).get('tickers', [])

st.sidebar.header("Controls")
with st.sidebar.expander("Dividend & Price Analysis", expanded=True):
    div_conf = config.get("dividend_analysis", {})
    div_tickers_input = st.text_area("Tickers:", value="\n".join(div_conf.get("tickers", [])), height=100, key="div_tickers")
    div_start_date_input = st.date_input("Start Date:", value=div_conf.get("start_date"), key="div_start_date")
    if st.button("Analyze Dividends & Price"):
        div_tickers = [t.strip().upper() for t in div_tickers_input.splitlines() if t.strip()]
        config["dividend_analysis"]["tickers"] = div_tickers
        config["dividend_analysis"]["start_date"] = div_start_date_input
        save_config(config)
        
        div_results = {}
        for ticker in div_tickers:
            try:
                funda_df, div_fig = adj_close_analysis(ticker, div_start_date_input)
                div_results[ticker] = {"funda": funda_df, "fig": div_fig}
            except Exception as e:
                st.error(f"Error analyzing {ticker} dividends: {e}")
        st.session_state.div_results = div_results
        st.rerun()

with st.sidebar.expander("Historical Valuation Analysis", expanded=True):
    pe_conf = config.get("pe_analysis", {})
    pe_tickers_input = st.text_area("Tickers:", value="\n".join(pe_conf.get("tickers", [])), height=100, key="pe_tickers")
    pe_start_date_input = st.date_input("Start Date:", value=pe_conf.get("start_date"), key="pe_start_date")
    if st.button("Analyze Valuation Ratios"):
        pe_tickers = [t.strip().upper() for t in pe_tickers_input.splitlines() if t.strip()]
        config["pe_analysis"]["tickers"] = pe_tickers
        config["pe_analysis"]["start_date"] = pe_start_date_input
        save_config(config)

        pe_results = {}
        for ticker in pe_tickers:
            try:
                with st.spinner(f"Analyzing {ticker}..."):
                    pe_fig = pe_ratio_analysis(ticker, pe_start_date_input)
                    ps_fig = ps_ratio_analysis(ticker, pe_start_date_input)
                    pe_results[ticker] = {"pe": pe_fig, "ps": ps_fig}
            except Exception as e:
                pe_results[ticker] = f"Valuation analysis not available. Reason: {e}"
        st.session_state.pe_results = pe_results
        st.session_state.pe_tickers_list = pe_tickers
        st.rerun()

# --- Main Display Area ---
tab_names = ["Summary & Dividends"] + [f"{t} Valuation" for t in st.session_state.pe_tickers_list]
tabs = st.tabs(tab_names)

with tabs[0]:
    if st.session_state.div_results:
        st.header("Combined Fundamentals & Metrics")
        all_fundamentals = [res['funda'] for res in st.session_state.div_results.values() if res and res.get('funda') is not None]
        if all_fundamentals:
            combined_df = pd.concat(all_fundamentals, axis=1)
            st.dataframe(format_df_for_display(combined_df))

        st.header("Individual Charts")
        for ticker, results in st.session_state.div_results.items():
            st.subheader(f"Analysis for {ticker}")
            if results and results.get('fig'):
                col1, col2 = st.columns([0.5, 2])
                with col1:
                    st.dataframe(format_df_for_display(results['funda']))
                with col2:
                    st.pyplot(results['fig'])
                    plt.close(results['fig'])
    else:
        st.info("Click 'Analyze Dividends & Price' to see results.")

for i, ticker in enumerate(st.session_state.pe_tickers_list, 1):
    with tabs[i]:
        st.header(f"Valuation Analysis for {ticker}")
        
        col1, col2 = st.columns([0.5, 2])
        with col1:
            st.subheader("Fundamentals")
            funda_df = get_display_fundamentals(ticker, pe_start_date_input)
            if funda_df is not None:
                st.dataframe(funda_df)
            else:
                st.warning("Could not retrieve fundamentals.")

        with col2:
            if st.session_state.pe_results and ticker in st.session_state.pe_results:
                result = st.session_state.pe_results[ticker]
                if isinstance(result, dict):
                    st.subheader("Price-to-Earnings (P/E) Ratio")
                    st.pyplot(result["pe"])
                    plt.close(result["pe"])
                    
                    st.subheader("Price-to-Sales (P/S) Ratio")
                    st.pyplot(result["ps"])
                    plt.close(result["ps"])
                else:
                    st.warning(result, icon="⚠️")
            else:
                st.info("Click 'Analyze Valuation Ratios' to see results.")

st.sidebar.info("Inputs are saved in `config.json`.", icon="💡")
