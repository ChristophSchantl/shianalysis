
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, date, timedelta

warnings.simplefilter("ignore", FutureWarning)
sns.set_theme(style="darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")

RISK_FREE_RATE = 0.02

# --- Feste CSV-Dateien
CSV_PATHS = {
    'SHI Income': 'SHI_INCOME_28Mai2025.csv',
    'SHI Alpha': 'SHI_ALPHA_28Mai2025.csv'
}

st.set_page_config(layout="wide", page_title="Strategieanalyse Dashboard")
st.title("📊 Strategie-Analyse & Risiko-Kennzahlen")

st.sidebar.header("🔄 Datenquellen auswählen")

# --- Zeitraum-Auswahl
default_start = date.today() - timedelta(days=2*365)
default_end = date.today()
start_date = st.sidebar.date_input("Startdatum", default_start)
end_date = st.sidebar.date_input("Enddatum", default_end)
if start_date > end_date:
    st.sidebar.error("Das Startdatum muss vor dem Enddatum liegen.")
    st.stop()

# --- SHI CSV Upload (optional)
shi_csv_files = st.sidebar.file_uploader(
    "Weitere SHI Zertifikate (optional, CSV)", 
    type=["csv"], 
    accept_multiple_files=True
)

# --- Yahoo-Ticker Eingabe
st.sidebar.subheader("Yahoo Finance Ticker auswählen")
ticker_1 = st.sidebar.text_input("Ticker 1", value="0P0000J5K3.F")
ticker_2 = st.sidebar.text_input("Ticker 2", value="0P0000J5K8.F")
ticker_3 = st.sidebar.text_input("Ticker 3", value="0P0000JM36.F")
ticker_4 = st.sidebar.text_input("Ticker 4", value="0P0001HPL2.F")

ticker_inputs = {
    'Yahoo 1': ticker_1,
    'Yahoo 2': ticker_2,
    'Yahoo 3': ticker_3,
    'Yahoo 4': ticker_4
}

def load_returns_from_csv(path_or_file, start_date=None, end_date=None):
    df = pd.read_csv(path_or_file, index_col=0, parse_dates=True)
    if start_date and end_date:
        # Filter nach Datum
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    close = pd.to_numeric(df['Close'], errors='coerce').ffill().dropna()
    returns = close.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def load_returns_from_yahoo(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)['Close'].dropna()
    returns = df.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    return returns, cumulative

def get_yahoo_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker

@st.cache_data
def load_and_sync_data(csv_paths, shi_csv_files, ticker_inputs, start_date, end_date):
    returns_dict, cumulative_dict = {}, {}
    # Feste CSVs laden
    for name, path in csv_paths.items():
        try:
            ret, cum = load_returns_from_csv(path, start_date, end_date)
            returns_dict[name] = ret
            cumulative_dict[name] = cum
        except Exception as e:
            st.warning(f"Fehler beim Laden von {name}: {e}")
    # Optional hochgeladene CSVs
    for file in shi_csv_files:
        name = file.name.replace(".csv", "")
        ret, cum = load_returns_from_csv(file, start_date, end_date)
        returns_dict[name] = ret
        cumulative_dict[name] = cum
    # Yahoo Ticker - mit echtem Namen
    for ticker in ticker_inputs.values():
        if ticker:
            try:
                yahoo_name = get_yahoo_name(ticker)
                ret, cum = load_returns_from_yahoo(ticker, start_date, end_date)
                returns_dict[yahoo_name] = ret
                cumulative_dict[yahoo_name] = cum
            except Exception as e:
                st.warning(f"Fehler beim Laden von Yahoo Ticker {ticker}: {e}")
    if len(returns_dict) == 0:
        return {}, {}
    # Zeitachsen synchronisieren
    try:
        common_index = sorted(set.intersection(*(set(r.index) for r in returns_dict.values())))
        for name in returns_dict:
            returns_dict[name] = returns_dict[name].loc[common_index]
            cumulative_dict[name] = cumulative_dict[name].loc[common_index]
    except Exception as e:
        st.warning("Fehler beim Synchronisieren der Zeitachsen. Eventuell gibt es zu wenig überlappende Daten.")
        return {}, {}
    return returns_dict, cumulative_dict

def calculate_metrics(returns_dict, cumulative_dict):
    metrics = pd.DataFrame()
    for name in returns_dict:
        ret = returns_dict[name]
        cum = cumulative_dict[name]
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        ret = pd.to_numeric(ret, errors='coerce').dropna()
        if ret.empty or cum.empty:
            continue
        days = (cum.index[-1] - cum.index[0]).days
        total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1)
        annual_ret = (1 + total_ret)**(365/days) - 1 if days > 0 else np.nan
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = (annual_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
        downside_returns = ret[ret < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annual_ret - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan
        drawdowns = (cum / cum.cummax()) - 1
        mdd = float(drawdowns.min())
        calmar = annual_ret / abs(mdd) if mdd < 0 else np.nan
        var_95 = ret.quantile(0.05)
        cvar_95 = ret[ret <= var_95].mean()
        win_rate = len(ret[ret > 0]) / len(ret)
        avg_win = ret[ret > 0].mean()
        avg_loss = ret[ret < 0].mean()
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else np.nan
        monthly_ret = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        positive_months = (monthly_ret > 0).mean()
        metrics.loc[name, 'Total Return'] = total_ret
        metrics.loc[name, 'Annual Return'] = annual_ret
        metrics.loc[name, 'Annual Volatility'] = annual_vol
        metrics.loc[name, 'Sharpe Ratio'] = sharpe
        metrics.loc[name, 'Sortino Ratio'] = sortino
        metrics.loc[name, 'Max Drawdown'] = mdd
        metrics.loc[name, 'Calmar Ratio'] = calmar
        metrics.loc[name, 'VaR (95%)'] = var_95
        metrics.loc[name, 'CVaR (95%)'] = cvar_95
        metrics.loc[name, 'Win Rate'] = win_rate
        metrics.loc[name, 'Avg Win'] = avg_win
        metrics.loc[name, 'Avg Loss'] = avg_loss
        metrics.loc[name, 'Profit Factor'] = profit_factor
        metrics.loc[name, 'Positive Months'] = positive_months
    return metrics

returns_dict, cumulative_dict = load_and_sync_data(CSV_PATHS, shi_csv_files, ticker_inputs, start_date, end_date)

if len(returns_dict) == 0:
    st.info("Bitte mindestens eine CSV-Datei bereitstellen oder einen gültigen Yahoo Ticker eintragen.")
    st.stop()

metrics = calculate_metrics(returns_dict, cumulative_dict)

tab1, tab2, tab3 = st.tabs(["🔍 Metriken", "📈 Performance", "📉 Drawdown & Korrelationen"])

with tab1:
    st.subheader("Erweiterte Risikokennzahlen")
    st.dataframe(metrics.style.format({
        'Total Return': '{:.2%}',
        'Annual Return': '{:.2%}',
        'Annual Volatility': '{:.2%}',
        'Sharpe Ratio': '{:.2f}',
        'Sortino Ratio': '{:.2f}',
        'Max Drawdown': '{:.2%}',
        'Calmar Ratio': '{:.2f}',
        'VaR (95%)': '{:.2%}',
        'CVaR (95%)': '{:.2%}',
        'Win Rate': '{:.2%}',
        'Avg Win': '{:.2%}',
        'Avg Loss': '{:.2%}',
        'Profit Factor': '{:.2f}',
        'Positive Months': '{:.2%}'
    }), use_container_width=True)

with tab2:
    st.subheader("Kumulative Performance")
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, cum in cumulative_dict.items():
        ax.plot(cum.index, cum / cum.iloc[0], label=name)
    ax.set_title("Kumulative Performance (Start = 1.0)")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("📉 Drawdown-Verlauf")
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, cum in cumulative_dict.items():
        drawdown = (cum / cum.cummax()) - 1
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.iloc[:, 0]
        ax.plot(drawdown.index, drawdown, label=name, alpha=0.8)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Korrelation der Tagesrenditen")
    returns_cleaned = {
        k: (v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v).dropna()
        for k, v in returns_dict.items()
    }
    df_corr = pd.DataFrame(returns_cleaned)
    corr = df_corr.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Korrelationsmatrix der täglichen Renditen")
    st.pyplot(fig_corr)
