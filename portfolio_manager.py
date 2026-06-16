"""
╔══════════════════════════════════════════════════════════════╗
║          PORTFOLIO MANAGER  ·  Gestión & Rebalanceo          ║
║          Seguimiento · Comparación vs Benchmark              ║
╚══════════════════════════════════════════════════════════════╝

Uso: python -m streamlit run portfolio_manager.py
"""

import os
import json
import warnings
import traceback
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
import requests
from io import StringIO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PORTFOLIO_DIR = "pm_portfolios"
os.makedirs(PORTFOLIO_DIR, exist_ok=True)

_ANTHROPIC_KEY_FILE = os.path.join(PORTFOLIO_DIR, ".anthropic_key")
_BANXICO_TOKEN_FILE  = os.path.join(PORTFOLIO_DIR, ".banxico_token")

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────
# PERSISTENCIA DE CREDENCIALES
# ─────────────────────────────────────────────────────────────

def _read_credential(filepath: str) -> str:
    """Lee una credencial de archivo local o de st.secrets."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                val = f.read().strip()
            if val:
                return val
    except Exception:
        pass
    return ""


def _write_credential(filepath: str, value: str) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(value.strip())
    except Exception:
        pass


def _load_anthropic_key() -> str:
    # 1. st.secrets (Streamlit Cloud)
    try:
        k = st.secrets.get("ANTHROPIC_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    # 2. Archivo local
    return _read_credential(_ANTHROPIC_KEY_FILE)


def _load_banxico_token() -> str:
    try:
        k = st.secrets.get("BANXICO_TOKEN", "")
        if k:
            return k
    except Exception:
        pass
    return _read_credential(_BANXICO_TOKEN_FILE)


# ─────────────────────────────────────────────────────────────
# GITHUB GIST — PERSISTENCIA EN LA NUBE
# No requiere librerías extra: usa requests (ya en requirements).
# Guarda portafolios Y credenciales (Anthropic, Banxico) en un
# Gist privado. Al reiniciar la app, todo se restaura solo.
# ─────────────────────────────────────────────────────────────

_GITHUB_TOKEN_FILE  = os.path.join(PORTFOLIO_DIR, ".github_token")
_GITHUB_GIST_ID_FILE = os.path.join(PORTFOLIO_DIR, ".github_gist_id")
_GIST_PF_PREFIX     = "pf_"          # portafolios: pf_NombrePortafolio.json
_GIST_CREDS_FILE    = "credentials.json"  # tokens: anthropic_key, banxico_token


def _load_github_token() -> str:
    try:
        k = st.secrets.get("GITHUB_TOKEN", "")
        if k:
            return k
    except Exception:
        pass
    return _read_credential(_GITHUB_TOKEN_FILE)


def _load_gist_id() -> str:
    try:
        k = st.secrets.get("GITHUB_GIST_ID", "")
        if k:
            return k
    except Exception:
        pass
    return _read_credential(_GITHUB_GIST_ID_FILE)


def _gist_headers() -> dict:
    return {
        "Authorization": f"Bearer {_load_github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gist_create() -> str:
    """Crea un Gist privado nuevo y devuelve su ID."""
    try:
        r = requests.post(
            "https://api.github.com/gists",
            headers=_gist_headers(),
            json={
                "description": "Portfolio Manager — datos persistentes",
                "public": False,
                "files": {"README.md": {"content": "Creado automáticamente por Portfolio Manager."}},
            },
            timeout=15,
        )
        if r.status_code == 201:
            gid = r.json().get("id", "")
            _write_credential(_GITHUB_GIST_ID_FILE, gid)
            return gid
    except Exception:
        pass
    return ""


def _gist_find_existing() -> str:
    """
    Busca en los Gists del usuario uno con descripción de Portfolio Manager.
    Útil al reiniciar cuando se pierde el GIST_ID del filesystem efímero.
    Devuelve el ID si lo encuentra, o "" si no hay ninguno.
    """
    if not _load_github_token():
        return ""
    try:
        page = 1
        while page <= 5:  # máximo 5 páginas = 500 gists
            r = requests.get(
                "https://api.github.com/gists",
                headers=_gist_headers(),
                params={"per_page": 100, "page": page},
                timeout=15,
            )
            if r.status_code != 200:
                break
            items = r.json()
            if not items:
                break
            for gist in items:
                desc = gist.get("description", "")
                if "Portfolio Manager" in desc:
                    gid = gist["id"]
                    _write_credential(_GITHUB_GIST_ID_FILE, gid)
                    return gid
            if len(items) < 100:
                break
            page += 1
    except Exception:
        pass
    return ""


def _gist_get_files() -> dict:
    """Devuelve el dict de archivos del Gist o {} si falla."""
    gid = _load_gist_id()
    if not gid:
        return {}
    try:
        r = requests.get(
            f"https://api.github.com/gists/{gid}",
            headers=_gist_headers(),
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("files", {})
    except Exception:
        pass
    return {}


def _gist_patch(files_patch: dict) -> bool:
    """
    Actualiza archivos en el Gist.
    files_patch = {"nombre.json": {"content": "..."}}
    Para borrar: {"nombre.json": None}
    """
    gid = _load_gist_id()
    if not gid or not _load_github_token():
        return False
    try:
        r = requests.patch(
            f"https://api.github.com/gists/{gid}",
            headers=_gist_headers(),
            json={"files": files_patch},
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False


def _gist_save_portfolio(name: str, data: dict) -> bool:
    """Guarda un portafolio en el Gist."""
    content = json.dumps(data, indent=2, default=str)
    return _gist_patch({f"{_GIST_PF_PREFIX}{name}.json": {"content": content}})


def _gist_delete_portfolio(name: str) -> None:
    """Elimina un portafolio del Gist."""
    _gist_patch({f"{_GIST_PF_PREFIX}{name}.json": None})


def _gist_save_credentials() -> None:
    """
    Persiste en el Gist los tokens actuales (Anthropic, Banxico).
    Se llama automáticamente cuando el usuario guarda cualquier credencial.
    """
    if not _load_gist_id():
        return
    creds: dict = {}
    if ak := _load_anthropic_key():
        creds["anthropic_key"] = ak
    if bt := _load_banxico_token():
        creds["banxico_token"] = bt
    if creds:
        _gist_patch({_GIST_CREDS_FILE: {"content": json.dumps(creds)}})


def _gist_sync_once() -> None:
    """
    Al inicio de la sesión, restaura desde el Gist todo lo que
    no exista en el sistema de archivos local (efímero en Cloud).
    Solo se ejecuta UNA vez por sesión.
    """
    if st.session_state.get("_gist_synced"):
        return
    st.session_state["_gist_synced"] = True

    if not _load_github_token() or not _load_gist_id():
        return

    files = _gist_get_files()
    if not files:
        return

    synced = 0
    for fname, fdata in files.items():
        # ── Portafolios ──────────────────────────────────────
        if fname.startswith(_GIST_PF_PREFIX) and fname.endswith(".json") and fdata:
            pname = fname[len(_GIST_PF_PREFIX):-5]
            path  = _portfolio_path(pname)
            if not os.path.exists(path):
                try:
                    data = json.loads(fdata.get("content", "{}"))
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, default=str)
                    synced += 1
                except Exception:
                    pass

        # Nota: las API keys (Anthropic, Banxico) NO se guardan en el Gist
        # para evitar que GitHub las detecte y las bloquee.
        # Usar Streamlit Secrets para persistir esas credenciales.

    if synced:
        st.session_state["_gist_sync_count"] = synced


# ─────────────────────────────────────────────────────────────
# TASA LIBRE DE RIESGO — CETES 28D (BANXICO SIE API)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cetes_rate(token: str) -> tuple[float, str]:
    """
    Retorna (tasa_decimal, descripción) de CETES 28 días desde Banxico SIE.
    Serie SF43936 — tasa de rendimiento anual en porcentaje.
    Si falla, devuelve (0.09, 'fallback').
    """
    if not token:
        return 0.09, "fallback"
    try:
        url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43936/datos/oportuno"
        resp = requests.get(
            url,
            headers={"Bmx-Token": token},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            series = data.get("bmx", {}).get("series", [])
            if series:
                datos = series[0].get("datos", [])
                if datos:
                    last = datos[-1]
                    fecha = last.get("fecha", "")
                    valor = float(last.get("dato", "9").replace(",", "."))
                    return round(valor / 100, 6), fecha
    except Exception:
        pass
    return 0.09, "fallback"

# ─────────────────────────────────────────────────────────────
# ESTILOS CSS  (Aesthetic: terminal financiero refinado)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════════════════
   DESIGN TOKENS  —  Apple Dark Mode palette
   ══════════════════════════════════════════════════════════ */
:root {
    --bg:        #000000;
    --surface:   #0d0d0d;
    --card:      #1c1c1e;
    --card-hi:   #2c2c2e;
    --border:    rgba(255,255,255,0.08);
    --border-hi: rgba(255,255,255,0.16);
    --accent:    #0a84ff;
    --green:     #30d158;
    --red:       #ff453a;
    --amber:     #ffd60a;
    --purple:    #bf5af2;
    --muted:     #8e8e93;
    --muted-2:   #48484a;
    --text:      #ffffff;
    --text-2:    rgba(255,255,255,0.55);
    --text-dim:  rgba(255,255,255,0.35);
    --mono:      'DM Mono', monospace;
    --radius-sm: 10px;
    --radius:    16px;
    --radius-lg: 22px;
    --shadow:    0 4px 24px rgba(0,0,0,0.5);
    --shadow-lg: 0 12px 48px rgba(0,0,0,0.6);
}

/* ── Base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg); }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.82rem;
    line-height: 1.65;
    color: var(--text-2);
}

/* ── Tabs — iOS Segment Control ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 4px;
    gap: 2px;
    margin-bottom: 28px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 8px 18px;
    background: transparent;
    border: none;
    color: var(--muted);
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: -0.1px;
    transition: color 0.18s ease, background 0.18s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.11) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: rgba(255,255,255,0.7) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Cards ────────────────────────────────────────────────── */
.pm-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px 24px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.pm-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    border-color: var(--border-hi);
}
.pm-card-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--muted);
    margin-bottom: 8px;
    font-family: var(--mono);
}
.pm-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -1.5px;
    font-family: var(--mono);
    line-height: 1;
}
.pm-card-sub {
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 6px;
}

/* ── KPI card gradient variants ──────────────────────────── */
.pm-card-green {
    background: linear-gradient(145deg, rgba(48,209,88,.09) 0%, var(--card) 60%) !important;
    border-color: rgba(48,209,88,.2) !important;
}
.pm-card-red {
    background: linear-gradient(145deg, rgba(255,69,58,.09) 0%, var(--card) 60%) !important;
    border-color: rgba(255,69,58,.2) !important;
}
.pm-card-blue {
    background: linear-gradient(145deg, rgba(10,132,255,.09) 0%, var(--card) 60%) !important;
    border-color: rgba(10,132,255,.2) !important;
}
.pm-card-amber {
    background: linear-gradient(145deg, rgba(255,214,10,.07) 0%, var(--card) 60%) !important;
    border-color: rgba(255,214,10,.18) !important;
}

/* ── Badges ───────────────────────────────────────────────── */
.badge-pos {
    color: var(--green); background: rgba(48,209,88,0.12);
    border: 1px solid rgba(48,209,88,0.22); border-radius: 20px;
    padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    font-family: var(--mono);
}
.badge-neg {
    color: var(--red); background: rgba(255,69,58,0.12);
    border: 1px solid rgba(255,69,58,0.22); border-radius: 20px;
    padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    font-family: var(--mono);
}
.badge-neu {
    color: var(--amber); background: rgba(255,214,10,0.1);
    border: 1px solid rgba(255,214,10,0.2); border-radius: 20px;
    padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    font-family: var(--mono);
}

/* ── Section label ────────────────────────────────────────── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: var(--muted);
    font-family: var(--mono);
    margin: 32px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.section-label::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 24px;
    height: 1.5px;
    background: var(--accent);
    border-radius: 2px;
}

/* ── Trade rows ───────────────────────────────────────────── */
.trade-buy  { color: var(--green);  font-weight: 600; font-family: var(--mono); }
.trade-sell { color: var(--red);    font-weight: 600; font-family: var(--mono); }
.trade-hold { color: var(--muted);  font-family: var(--mono); }

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.86rem !important;
    letter-spacing: -0.1px !important;
    border: none !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    box-shadow: 0 4px 16px rgba(10,132,255,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(10,132,255,0.38) !important;
}
.stButton > button:not([kind="primary"]) {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: var(--card-hi) !important;
    border-color: var(--border-hi) !important;
    color: var(--text) !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: rgba(10,132,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(10,132,255,0.12) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(10,132,255,0.55) !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

/* ── Plotly ───────────────────────────────────────────────── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Alerts ───────────────────────────────────────────────── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Streamlit metrics ────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -1px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* ── Live pulse dot ───────────────────────────────────────── */
@keyframes _lp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.65)} }
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    display: inline-block; vertical-align: middle; margin-right: 7px;
    animation: _lp 2.4s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(48,209,88,.55);
}

/* ── Glow utilities ───────────────────────────────────────── */
.glow-green { text-shadow: 0 0 22px rgba(48,209,88,.42); }
.glow-red   { text-shadow: 0 0 22px rgba(255,69,58,.42); }
.glow-blue  { text-shadow: 0 0 22px rgba(10,132,255,.42); }

/* ── Market Pulse card ────────────────────────────────────── */
@keyframes _fi2 { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
.mp-card {
    transition: transform .2s ease, box-shadow .2s ease;
    animation: _fi2 .28s ease-out both;
}
.mp-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 52px rgba(0,0,0,.45);
}

/* ── Ticker chip ──────────────────────────────────────────── */
.ticker-chip {
    display: inline-flex; align-items: center;
    background: rgba(10,132,255,.12);
    border: 1px solid rgba(10,132,255,.22);
    border-radius: 7px; padding: 2px 8px;
    font-family: var(--mono); font-size: .7rem; font-weight: 700;
    color: var(--accent); letter-spacing: .4px;
}

/* ── Hero typography ──────────────────────────────────────── */
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text);
    line-height: 1;
    margin-bottom: 4px;
}
.hero-sub {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Plotly chart containers — card con fondo propio ─────── */
[data-testid="stPlotlyChart"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: var(--border-hi) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,.45) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTES DE UNIVERSO
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_market_universe() -> list[str]:
    """
    Obtiene el universo completo: S&P 500 desde Wikipedia + activos base.
    Caché en disco por 24h — evita request a Wikipedia en cada arranque.
    """
    import json as _json, os as _os, time as _time
    _cache_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                ".universe_cache.json")
    _ttl_sec    = 86400  # 24 horas

    # Leer caché local si existe y es reciente
    try:
        if _os.path.exists(_cache_file):
            with open(_cache_file, "r") as _f:
                _cached = _json.load(_f)
            if _time.time() - _cached.get("ts", 0) < _ttl_sec:
                return _cached["tickers"]
    except Exception:
        pass

    base_assets = [
        # Sector ETFs del S&P 500
        "XLK","XLF","XLV","XLY","XLI","XLE","XLU","XLRE","XLB","XLC","XLP",
        "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","VXX",
        # Commodities ETFs (físicos y de futuros)
        "PSLV","PHYS","SGOL","IAU",        # oro/plata físicos Sprott/iShares
        "USO","UCO","DBO",                  # petróleo
        "DBC","PDBC","GSG",                 # commodities diversificados
        "COPX","CPER","JJC",                # cobre
        "WEAT","CORN","SOYB",               # agrícolas
        "BTC-USD","ETH-USD","SOL-USD",
        "AAPL","MSFT","AMZN","GOOGL","GOOG","META","TSLA","NVDA",
        "AMD","INTC","NFLX","ADBE","CRM","ORCL","CSCO","ACN",
        "PLTR","SOFI","NU","COIN","HOOD","DKNG","ROKU", "KR",
        "JPM","BAC","WFC","C","GS","MS","V","MA","AXP",
        "JNJ","UNH","LLY","PFE","MRK","ABBV","TMO",
        "PG","KO","PEP","COST","WMT","TGT","HD","MCD",
        "XOM","CVX","COP","SLB","EOG","SHOP",
        "BA","CAT","GE","LMT","RTX","HON", "SPCX",
        "DIS","CMCSA","TMUS","VZ","T","SNDK","TTWO","INCY","FIX","ALL","ALB","CBOE", 
    ]
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        sp500_df = pd.read_html(StringIO(resp.text))[0]
        sp500_tickers = [t.replace(".", "-") for t in sp500_df["Symbol"].tolist()]
        result = sorted(list(set(base_assets + sp500_tickers)))
        # Guardar en caché local
        try:
            with open(_cache_file, "w") as _f:
                _json.dump({"ts": _time.time(), "tickers": result}, _f)
        except Exception:
            pass
        return result
    except Exception:
        result = sorted(list(set(base_assets)))
        try:
            with open(_cache_file, "w") as _f:
                _json.dump({"ts": _time.time(), "tickers": result}, _f)
        except Exception:
            pass
        return result


UNIVERSE: list[str] = get_market_universe()


# ─────────────────────────────────────────────────────────────
# CAPA DE DATOS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices(tickers: list[str]) -> dict:
    """
    Precios en tiempo real vía batch yf.download (period='2d', interval='1d').
    Durante horario de mercado yfinance incluye el bar parcial de hoy como
    vals[-1] (precio actual) y el cierre de ayer como vals[-2].
    Cuando solo hay 1 bar (arranque del día o ticker muy nuevo) se usa
    fast_info.last_price para el precio y fast_info.previous_close para prev.
    """
    prices: dict = {}
    if not tickers:
        return prices
    clean = [t for t in tickers if t and str(t).strip() not in ("", "nan")]
    if not clean:
        return prices

    # Fecha de hoy en hora del mercado (Eastern)
    try:
        today_date = pd.Timestamp.now(tz="America/New_York").date()
    except Exception:
        today_date = pd.Timestamp.now().date()

    # ── Batch diario — igual que el código original que funcionaba ──
    close_df = pd.DataFrame()
    try:
        raw = yf.download(
            clean, period="2d", interval="1d",
            auto_adjust=True, prepost=False,
            progress=False, threads=True,
        )
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"]
            elif len(clean) == 1:
                close_df = raw[["Close"]].rename(columns={"Close": clean[0]})
    except Exception:
        pass

    # Tickers que necesitan fast_info solo para el PRECIO (prev_close viene del daily)
    needs_live_price: list[str] = []
    # Tickers que no tienen ningún dato daily
    needs_full_fallback: list[str] = []

    for t in clean:
        if not close_df.empty and t in close_df.columns:
            vals = close_df[t].dropna()
            if len(vals) == 0:
                needs_full_fallback.append(t)
                continue

            # ¿El último bar es de hoy? → yfinance ya incluyó el bar parcial actual
            try:
                last_idx = vals.index[-1]
                last_date = (last_idx.date() if hasattr(last_idx, "date")
                             else pd.Timestamp(last_idx).date())
            except Exception:
                last_date = None

            if last_date == today_date and len(vals) >= 2:
                # Bar de hoy incluido: vals[-1]=precio actual, vals[-2]=cierre ayer
                prices[t] = {
                    "price":      float(vals.iloc[-1]),
                    "prev_close": float(vals.iloc[-2]),
                }
            elif last_date == today_date and len(vals) == 1:
                # Solo el bar de hoy (ticker recién listado hoy mismo)
                prices[t] = {"price": float(vals.iloc[0]), "prev_close": float(vals.iloc[0])}
                needs_live_price.append(t)
            else:
                # Bar de hoy NO incluido (IPO reciente, primeros minutos del día, etc.)
                # vals[-1] ES el cierre de ayer → usarlo como prev_close
                # y pedir el precio actual a fast_info
                prices[t] = {
                    "price":      float(vals.iloc[-1]),   # placeholder
                    "prev_close": float(vals.iloc[-1]),   # cierre de ayer ✓
                }
                needs_live_price.append(t)
        else:
            needs_full_fallback.append(t)

    # ── fast_info en paralelo solo para lo que falta ─────────────────
    import concurrent.futures

    def _get_price_only(t: str):
        """Devuelve solo last_price; prev_close ya lo tenemos del daily."""
        try:
            lp = yf.Ticker(t).fast_info.last_price
            return t, float(lp) if lp is not None else 0.0
        except Exception:
            return t, 0.0

    def _get_full(t: str):
        try:
            fi = yf.Ticker(t).fast_info
            lp = fi.last_price
            pc = fi.previous_close
            return t, {
                "price":      float(lp if lp is not None else 0),
                "prev_close": float(pc if pc is not None else 0),
            }
        except Exception:
            return t, {"price": 0.0, "prev_close": 0.0}

    all_tasks = needs_live_price + needs_full_fallback
    if all_tasks:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(all_tasks), 12)
        ) as ex:
            price_futs  = {ex.submit(_get_price_only, t): t for t in needs_live_price}
            full_futs   = {ex.submit(_get_full, t):       t for t in needs_full_fallback}

            for fut, t in price_futs.items():
                lp = fut.result()
                if lp > 0:
                    prices[t]["price"] = lp   # prev_close ya está del daily

            for fut, t in full_futs.items():
                data = fut.result()
                if data["price"] > 0:
                    prices[t] = data

    return prices


@st.cache_data(ttl=60, show_spinner=False)
def fetch_pulse_data(tickers: list[str]) -> dict:
    """
    Datos en tiempo real para las tarjetas Market Pulse.

    Fuente de verdad:
      prev_close      — último cierre diario (sin pre/post market)
      day_open        — primer bar de la sesión regular de hoy (prepost=False)
      price           — fast_info.last_price (tick más reciente)
      change_vs_prev  — (price - prev_close) / prev_close  → +9.64% como Yahoo
      change_vs_open  — (price - day_open)   / day_open    → movimiento intradiario

    Clave: prepost=False en todos los downloads para excluir pre/after market.
    """
    if not tickers:
        return {}

    result = {}

    # ── 1. Intraday sesión regular hoy (1m, sin pre/post) ─────
    # prepost=False es crítico: excluye las barras antes de 9:30am
    # que distorsionan el "open" y el sparkline
    try:
        intra_raw = yf.download(
            tickers, period="1d", interval="1m",
            auto_adjust=True, prepost=False,
            progress=False, threads=True,
        )
        if isinstance(intra_raw.columns, pd.MultiIndex):
            intra_close = intra_raw["Close"]
        else:
            intra_close = intra_raw[["Close"]].rename(
                columns={"Close": tickers[0]}) if len(tickers) == 1 else intra_raw
    except Exception:
        intra_close = pd.DataFrame()

    # ── 2. 2 días diarios → prev_close confiable ──────────────
    # Los cierres diarios ajustados son la referencia más fiable
    try:
        daily_raw = yf.download(
            tickers, period="5d", interval="1d",
            auto_adjust=True, prepost=False,
            progress=False, threads=True,
        )
        if isinstance(daily_raw.columns, pd.MultiIndex):
            daily_close = daily_raw["Close"]
        else:
            daily_close = daily_raw[["Close"]].rename(
                columns={"Close": tickers[0]}) if len(tickers) == 1 else daily_raw
    except Exception:
        daily_close = pd.DataFrame()

    # ── 3. Histórico mensual → RSI / SMA20 / retorno 1S ───────
    try:
        hist_1m = yf.download(
            tickers, period="1mo", interval="1d",
            auto_adjust=True, prepost=False,
            progress=False, threads=True,
        )
        if isinstance(hist_1m.columns, pd.MultiIndex):
            close_1m = hist_1m["Close"]
            vol_1m   = hist_1m["Volume"]
        else:
            close_1m = hist_1m[["Close"]].rename(
                columns={"Close": tickers[0]}) if len(tickers) == 1 else hist_1m
            vol_1m   = hist_1m[["Volume"]].rename(
                columns={"Volume": tickers[0]}) if len(tickers) == 1 else hist_1m
    except Exception:
        close_1m = pd.DataFrame()
        vol_1m   = pd.DataFrame()

    # ── 4. Precios live via batch (reutiliza fetch_live_prices) ──
    live_prices = fetch_live_prices(tuple(sorted(tickers)))

    for t in tickers:
        # Precio y prev_close desde el batch ya cacheado
        price      = live_prices.get(t, {}).get("price",      0.0)
        prev_close = live_prices.get(t, {}).get("prev_close", 0.0)

        # Fallback: si el batch diario tiene mejor prev_close, usarlo
        if prev_close <= 0 and not daily_close.empty and t in daily_close.columns:
            dc = daily_close[t].dropna()
            if len(dc) >= 2:
                prev_close = float(dc.iloc[-2])

        # Apertura de hoy: primer bar de la sesión regular
        day_open = 0.0
        spark_p, spark_t = [], []
        if not intra_close.empty and t in intra_close.columns:
            s = intra_close[t].dropna()
            if not s.empty:
                day_open = float(s.iloc[0])       # primer bar = apertura oficial
                spark_p  = [round(float(v), 2) for v in s.values]
                spark_t  = [str(i.strftime("%H:%M")) for i in s.index]

        # Variaciones
        change_vs_prev = (price - prev_close) / prev_close if prev_close > 0 else 0.0
        change_vs_open = (price - day_open)   / day_open   if day_open   > 0 else 0.0

        # RSI / SMA20 / retorno semanal
        chg_1w = None
        rsi14  = None
        above_sma20 = None
        vol_ratio   = None

        if not close_1m.empty and t in close_1m.columns:
            cl = close_1m[t].dropna()
            if len(cl) >= 5:
                chg_1w = float(cl.iloc[-1] / cl.iloc[-5] - 1)
            if len(cl) >= 14:
                delta = cl.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain / loss.replace(0, np.nan)
                rsi_s = 100 - 100 / (1 + rs)
                v     = rsi_s.iloc[-1]
                rsi14 = float(v) if not (isinstance(v, float) and np.isnan(v)) else None
            if len(cl) >= 20:
                sma20       = cl.rolling(20).mean().iloc[-1]
                above_sma20 = bool(cl.iloc[-1] > sma20)

        if not vol_1m.empty and t in vol_1m.columns:
            vl = vol_1m[t].dropna()
            if len(vl) >= 5:
                avg_v     = vl.iloc[:-1].mean()
                vol_ratio = float(vl.iloc[-1] / avg_v) if avg_v > 0 else None

        result[t] = {
            "price":          price,
            "prev_close":     prev_close,
            "day_open":       day_open,
            "change_vs_prev": change_vs_prev,
            "change_vs_open": change_vs_open,
            "change_1d":      change_vs_prev,   # alias
            "change_1w":      chg_1w,
            "rsi14":          rsi14,
            "above_sma20":    above_sma20,
            "vol_ratio":      vol_ratio,
            "spark_prices":   spark_p,
            "spark_times":    spark_t,
        }

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_range_data(tickers: tuple, period: str) -> pd.DataFrame:
    """
    Cierres diarios ajustados para el período solicitado.
    period: '1M' | '6M' | 'YTD' | '1A'
    Devuelve DataFrame con columnas = tickers, índice = fecha.
    """
    _map = {"1M": "1mo", "6M": "6mo", "YTD": "ytd", "1A": "1y"}
    yf_period = _map.get(period, "1mo")
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers), period=yf_period, interval="1d",
            auto_adjust=True, prepost=False, progress=False, threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"]
        else:
            df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        df = df.dropna(how="all")
        # Aseguramos que el índice sea DatetimeIndex sin timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_events_data(tickers: tuple) -> dict:
    """
    Obtiene earnings dates y datos de dividendos para cada ticker.
    Retorna dict: {ticker: {next_earnings, div_yield, annual_div, next_ex_date, last_div}}
    TTL = 2h (los eventos no cambian con frecuencia).
    """
    result = {}
    for t in tickers:
        entry = {"next_earnings": None, "div_yield": 0.0,
                 "annual_div": 0.0, "next_ex_date": None, "last_div": 0.0}
        try:
            tk = yf.Ticker(t)
            # ── Earnings date ──────────────────────────────────────
            try:
                cal = tk.calendar
                if cal is not None:
                    ed = None
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if hasattr(ed, "__iter__") and not isinstance(ed, str):
                            ed = list(ed)[0] if ed else None
                    elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                        ed = cal["Earnings Date"].iloc[0] if not cal.empty else None
                    elif hasattr(cal, "index") and "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"]
                        ed = val.iloc[0] if hasattr(val, "iloc") else val
                    if ed is not None:
                        entry["next_earnings"] = pd.Timestamp(ed)
            except Exception:
                pass
            # ── Dividend info ──────────────────────────────────────
            try:
                info = tk.info or {}
                entry["div_yield"]  = float(info.get("dividendYield") or 0)
                entry["annual_div"] = float(info.get("dividendRate")  or 0)
                ex_ts = info.get("exDividendDate")
                if ex_ts:
                    entry["next_ex_date"] = datetime.fromtimestamp(int(ex_ts))
            except Exception:
                pass
            try:
                divs = tk.dividends
                if not divs.empty:
                    entry["last_div"] = float(divs.iloc[-1])
            except Exception:
                pass
        except Exception:
            pass
        result[t] = entry
    return result


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Descarga histórico de precios de cierre (ajustados)."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"].copy()
        else:
            df = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
            if len(tickers) == 1:
                df.columns = [tickers[0]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


def get_usd_mxn() -> float:
    try:
        p = yf.Ticker("USDMXN=X").fast_info.last_price
        return float(p) if p and p > 0 else 17.5
    except Exception:
        return 17.5


# ─────────────────────────────────────────────────────────────
# PERSISTENCIA (JSON)
# ─────────────────────────────────────────────────────────────

def _portfolio_path(name: str) -> str:
    return os.path.join(PORTFOLIO_DIR, f"{name}.json")


def save_portfolio(name: str, transactions: list[dict], target_weights: dict, benchmark: str = "SPY") -> None:
    data = {
        "name": name,
        "updated_at": datetime.now().isoformat(),
        "benchmark": benchmark,
        "target_weights": target_weights,
        "transactions": transactions,
        "thesis":   st.session_state.get("thesis", {}),
        "alerts":   st.session_state.get("alerts", []),
        "extra_benchmarks":  st.session_state.get("extra_benchmarks", []),
        "perf_custom_start": st.session_state.get("perf_custom_start"),
        "perf_use_custom":   st.session_state.get("perf_use_custom", False),
        "watchlist":         st.session_state.get("watchlist", []),
    }
    # 1. Guardar localmente (siempre)
    with open(_portfolio_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    # 2. Sincronizar al Gist (si está configurado)
    _gist_save_portfolio(name, data)


def load_portfolio(name: str) -> dict | None:
    path = _portfolio_path(name)
    data = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Intentar desde el Gist (el archivo local se borró al reiniciar Cloud)
        files = _gist_get_files()
        fname = f"{_GIST_PF_PREFIX}{name}.json"
        if fname in files and files[fname]:
            try:
                data = json.loads(files[fname].get("content", "{}"))
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception:
                data = None
    if data is None:
        return None
    # Backward-compat: archivos viejos usaban "holdings" como posiciones agregadas
    if "transactions" not in data and "holdings" in data:
        migrated = []
        for h in data.get("holdings", []):
            migrated.append({
                "Ticker": h.get("Ticker", ""),
                "Type":   "BUY",
                "Shares": h.get("Shares", 0),
                "Price":  h.get("AvgCost", 0),
                "Date":   h.get("Date", str(date.today())),
                "Notes":  "Importado",
            })
        data["transactions"] = migrated
    return data


def list_portfolios() -> list[str]:
    # Sincronizar desde Gist una vez por sesión (restaura archivos tras reinicio)
    _gist_sync_once()
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(PORTFOLIO_DIR)
        if f.endswith(".json")
    )


def delete_portfolio(name: str) -> None:
    path = _portfolio_path(name)
    if os.path.exists(path):
        os.remove(path)
    # Eliminar también del Gist
    _gist_delete_portfolio(name)


# ─────────────────────────────────────────────────────────────
# MOTOR DE REBALANCEO  (la lógica correcta y transparente)
# ─────────────────────────────────────────────────────────────

def calculate_rebalance(
    holdings: pd.DataFrame,           # cols: Ticker, Shares, AvgCost
    target_weights: dict,             # {ticker: weight}  suma = 1.0
    prices: dict,                     # {ticker: {"price": x}}
    new_capital: float = 0.0,
    mode: str = "active",             # "active" | "passive"
    min_trade_usd: float = 5.0,       # ignorar trades menores a este monto
    commission_per_trade: float = 0.0,
) -> dict:
    """
    Calcula el rebalanceo de forma matemáticamente correcta y transparente.

    Retorna un dict con todos los datos necesarios para mostrar y ejecutar.
    """
    result = {
        "ok": False,
        "error": None,
        "nav": 0.0,
        "total_capital": 0.0,
        "current_weights": {},
        "target_weights": target_weights,
        "trades": [],
        "buys_total": 0.0,
        "sells_total": 0.0,
        "turnover": 0.0,
        "estimated_commissions": 0.0,
    }

    # ── 1. Valores actuales por posición ──────────────────────────────
    position_values: dict[str, float] = {}
    for _, row in holdings.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        shares = float(row.get("Shares", 0))
        price_info = prices.get(ticker, {})
        price = float(price_info.get("price", 0))
        if ticker and price > 0:
            position_values[ticker] = shares * price

    nav = sum(position_values.values())
    total_capital = nav + new_capital
    result["nav"] = nav
    result["total_capital"] = total_capital

    if total_capital <= 0:
        result["error"] = "Capital total es cero. Agrega posiciones o capital nuevo."
        return result

    # ── 2. Pesos actuales ─────────────────────────────────────────────
    current_weights = {t: v / nav for t, v in position_values.items()} if nav > 0 else {}
    result["current_weights"] = current_weights

    # ── 3. Validar pesos objetivo ─────────────────────────────────────
    tw_sum = sum(target_weights.values())
    if abs(tw_sum - 1.0) > 0.02:
        result["error"] = f"Los pesos objetivo suman {tw_sum:.2%}, deben sumar 100%."
        return result

    # Normalizar por si hay drift mínimo
    target_weights = {t: w / tw_sum for t, w in target_weights.items()}

    # ── 4. Valores objetivo y delta ───────────────────────────────────
    all_tickers = set(list(position_values.keys()) + list(target_weights.keys()))
    trades = []

    for ticker in sorted(all_tickers):
        current_val = position_values.get(ticker, 0.0)
        target_weight = target_weights.get(ticker, 0.0)
        target_val = target_weight * total_capital
        delta_val = target_val - current_val

        price_info = prices.get(ticker, {})
        price = float(price_info.get("price", 0))

        # En modo pasivo: no vender
        if mode == "passive" and delta_val < 0:
            delta_val = 0.0
            target_val = current_val
            target_weight = current_val / total_capital if total_capital > 0 else 0

        shares_to_trade = delta_val / price if price > 0 else 0.0
        action = "BUY" if delta_val > min_trade_usd else ("SELL" if delta_val < -min_trade_usd else "HOLD")

        trades.append({
            "Ticker": ticker,
            "Precio": price,
            "Peso Actual": current_weights.get(ticker, 0.0),
            "Peso Objetivo": target_weight,
            "Drift": target_weight - current_weights.get(ticker, 0.0),
            "Val. Actual": current_val,
            "Val. Objetivo": target_val,
            "Delta USD": delta_val,
            "Acciones": shares_to_trade,
            "Acción": action,
        })

    # Ordenar: primero ventas, luego compras, luego holds
    order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    trades.sort(key=lambda x: (order.get(x["Acción"], 2), -abs(x["Delta USD"])))

    result["trades"] = trades

    # ── 5. Métricas de rebalanceo ─────────────────────────────────────
    buys  = sum(t["Delta USD"] for t in trades if t["Acción"] == "BUY")
    sells = abs(sum(t["Delta USD"] for t in trades if t["Acción"] == "SELL"))

    n_trades = sum(1 for t in trades if t["Acción"] != "HOLD")
    result["buys_total"]  = buys
    result["sells_total"] = sells
    result["turnover"]    = sells / nav if nav > 0 else 0.0
    result["estimated_commissions"] = n_trades * commission_per_trade
    result["n_trades"] = n_trades
    result["ok"] = True
    return result


# ─────────────────────────────────────────────────────────────
# MÉTRICAS DE PERFORMANCE
# ─────────────────────────────────────────────────────────────

def compute_performance_metrics(returns: pd.Series, rf_annual: float = 0.0) -> dict:
    """
    Calcula métricas de rendimiento de una serie de retornos diarios.

    Notas sobre anualización:
    - Con < 252 días de datos la anualización es indicativa, no confiable.
    - 'Retorno Período' es siempre el retorno real acumulado sin anualizar.
    - 'Retorno Anualizado' solo se presenta si hay ≥ 180 días de datos;
      de lo contrario se devuelve None para que la UI lo marque como
      estimado.
    """
    if returns.empty or len(returns) < 5:
        return {}

    n_days   = len(returns)
    rf_daily = rf_annual / TRADING_DAYS
    excess   = returns - rf_daily

    # ── Retorno real del período (sin anualizar) ──────────────────
    period_return = (1 + returns).prod() - 1

    # ── Anualización (solo si hay datos suficientes) ──────────────
    # Con < 180 días la anualización exagera enormemente cualquier
    # retorno positivo — se marca como estimado.
    if n_days >= TRADING_DAYS:
        # Período completo de ≥ 1 año → anualización fiable
        ann_return      = (1 + returns).prod() ** (TRADING_DAYS / n_days) - 1
        ann_return_est  = False
    else:
        # Período < 1 año → anualizar matemáticamente pero avisar
        ann_return      = (1 + returns).prod() ** (TRADING_DAYS / n_days) - 1
        ann_return_est  = True   # flag: "este dato es una proyección, no histórico"

    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    # Sharpe — se calcula sobre retornos del período, no anualizado,
    # para evitar inflar el ratio cuando n_days es pequeño.
    # Usamos la versión estándar (anualizada) pero con el vol correcto.
    sharpe = (excess.mean() / returns.std() * np.sqrt(TRADING_DAYS)
              if returns.std() > 0 else 0.0)

    # ── Drawdown ──────────────────────────────────────────────────
    cum    = (1 + returns).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    max_dd = dd.min()

    # ── Calmar — solo fiable con ≥ 1 año ─────────────────────────
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # ── Sortino ───────────────────────────────────────────────────
    downside    = returns[returns < rf_daily]
    sortino_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else ann_vol
    sortino     = (ann_return - rf_annual) / sortino_vol if sortino_vol > 0 else 0.0

    # ── Win rate, VaR, CVaR ───────────────────────────────────────
    win_rate = (returns > 0).mean()
    var_95   = returns.quantile(0.05)
    cvar_95  = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    return {
        "Retorno Período":    period_return,    # siempre real, sin anualizar
        "Retorno Anualizado": ann_return,        # puede ser proyección si < 1 año
        "Retorno Anual Est":  ann_return_est,    # True = proyección (<1 año de datos)
        "N Días":             n_days,
        "Volatilidad Anual":  ann_vol,
        "Sharpe Ratio":       sharpe,
        "Sortino Ratio":      sortino,
        "Max Drawdown":       max_dd,
        "Calmar Ratio":       calmar,
        "Win Rate":           win_rate,
        "VaR 95%":            var_95,
        "CVaR 95%":           cvar_95,
    }


def build_portfolio_equity(holdings: pd.DataFrame, prices_df: pd.DataFrame) -> pd.Series:
    """Construye la curva de valor del portafolio a lo largo del tiempo.
    NOTA: usa posiciones actuales para TODO el historial — solo usar como fallback.
    """
    if prices_df.empty or holdings.empty:
        return pd.Series(dtype=float)

    equity = pd.Series(0.0, index=prices_df.index)
    for _, row in holdings.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        shares = float(row.get("Shares", 0))
        if ticker in prices_df.columns and shares > 0:
            equity = equity + prices_df[ticker] * shares

    return equity.dropna()


def build_portfolio_equity_from_transactions(
    txns: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.Series:
    """
    Curva de valor REAL del portafolio reconstruida desde el historial
    de transacciones.

    Por cada día de trading se calcula exactamente qué acciones se tenían
    en ese momento (acumulando BUYs y SELLs en orden cronológico) y se
    multiplica por el precio histórico de ese día.

    Esto corrige el bug de usar las posiciones actuales para todo el historial:
    si compraste NVDA ayer, NO aparece en el historial de hace 6 meses.
    """
    if txns.empty or prices_df.empty:
        return pd.Series(dtype=float)

    txns = txns.copy()
    txns["Date"] = pd.to_datetime(txns["Date"])
    txns = txns.sort_values("Date").reset_index(drop=True)

    trading_days = prices_df.index          # DatetimeIndex
    positions: dict[str, float] = {}        # {ticker: shares_held}
    nav_dict: dict = {}
    txn_ptr = 0
    n_txns  = len(txns)

    for day in trading_days:
        # Aplicar todas las transacciones cuya fecha ≤ este día de trading
        while txn_ptr < n_txns and txns.at[txn_ptr, "Date"] <= day:
            t  = str(txns.at[txn_ptr, "Ticker"])
            sh = float(txns.at[txn_ptr, "Shares"])
            tp = str(txns.at[txn_ptr, "Type"]).upper()
            positions.setdefault(t, 0.0)
            if tp == "BUY":
                positions[t] += sh
            elif tp == "SELL":
                positions[t] = max(0.0, positions[t] - sh)
            txn_ptr += 1

        if not positions:
            continue   # aún no hay ninguna compra

        # NAV = Σ acciones_i × precio_i (solo tickers con precio disponible ese día)
        nav = 0.0
        for t, sh in positions.items():
            if sh < 1e-9 or t not in prices_df.columns:
                continue
            p = prices_df.at[day, t] if day in prices_df.index else float("nan")
            if pd.notna(p) and p > 0:
                nav += sh * p

        if nav > 0:
            nav_dict[day] = nav

    return pd.Series(nav_dict)


def build_twr_series(txns: pd.DataFrame, prices_df: pd.DataFrame) -> pd.Series:
    """
    Time-Weighted Return (TWR) — retorno puro sin efecto de aportaciones.

    Cada vez que el usuario deposita capital nuevo (BUY sin SELL compensador),
    ese monto se descuenta del "retorno" del día usando Modified Dietz:

        CF(d)  = Σ compras_d  − Σ ventas_d     (neto de capital externo ese día)
        r(d)   = (NAV(d) − NAV(d−1) − CF(d)) / (NAV(d−1) + CF(d))
        TWR(d) = TWR(d−1) × (1 + r(d))

    Resultado: serie normalizada a base 100 que sube/baja solo por
    movimientos de precios, ignorando depósitos periódicos.
    """
    if txns.empty or prices_df.empty:
        return pd.Series(dtype=float)

    txns = txns.copy()
    txns["Date"] = pd.to_datetime(txns["Date"])
    txns = txns.sort_values("Date").reset_index(drop=True)

    # ── Flujos de capital netos por día ──────────────────────────
    # CF > 0 → capital entrando  |  CF < 0 → capital saliendo
    daily_cf: dict = {}
    for _, row in txns.iterrows():
        d      = pd.Timestamp(row["Date"]).normalize()
        amount = float(row["Shares"]) * float(row["Price"])
        tp     = str(row["Type"]).upper()
        if tp == "BUY":
            daily_cf[d] = daily_cf.get(d, 0.0) + amount
        elif tp == "SELL":
            daily_cf[d] = daily_cf.get(d, 0.0) - amount

    # ── Reconstruir NAV diario desde transacciones ───────────────
    trading_days = prices_df.index
    positions: dict[str, float] = {}
    nav_dict: dict = {}
    txn_ptr = 0
    n_txns  = len(txns)

    for day in trading_days:
        while txn_ptr < n_txns and txns.at[txn_ptr, "Date"] <= day:
            t  = str(txns.at[txn_ptr, "Ticker"])
            sh = float(txns.at[txn_ptr, "Shares"])
            tp = str(txns.at[txn_ptr, "Type"]).upper()
            positions.setdefault(t, 0.0)
            if tp == "BUY":
                positions[t] += sh
            elif tp == "SELL":
                positions[t] = max(0.0, positions[t] - sh)
            txn_ptr += 1

        if not positions:
            continue

        nav = 0.0
        for t, sh in positions.items():
            if sh < 1e-9 or t not in prices_df.columns:
                continue
            p = prices_df.at[day, t] if day in prices_df.index else float("nan")
            if pd.notna(p) and p > 0:
                nav += sh * p
        if nav > 0:
            nav_dict[day] = nav

    if len(nav_dict) < 2:
        return pd.Series(dtype=float)

    nav_series = pd.Series(nav_dict)

    # ── Calcular TWR acumulado ────────────────────────────────────
    twr_values = [100.0]
    dates      = [nav_series.index[0]]

    for i in range(1, len(nav_series)):
        day      = nav_series.index[i]
        prev_nav = nav_series.iloc[i - 1]
        curr_nav = nav_series.iloc[i]
        cf       = daily_cf.get(day, 0.0)

        denom = prev_nav + cf
        if denom > 1.0:
            r = (curr_nav - prev_nav - cf) / denom
        else:
            r = 0.0

        # Acotar retornos diarios extremos para evitar errores de datos
        r = max(-0.50, min(0.50, r))

        twr_values.append(twr_values[-1] * (1.0 + r))
        dates.append(day)

    return pd.Series(twr_values, index=dates)


# ─────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "portfolio_name": "",
        "transactions":   [],
        "target_weights": {},
        "benchmark":      "SPY",
        "rf_rate":        0.09,
        # thesis: {ticker: {thesis, target_price, exit_date, catalyst, added_date}}
        "thesis":         {},
        # alerts: [{type, ticker, condition, threshold, active}]
        "alerts":         [],
        # extra benchmarks for multi-benchmark comparison
        "extra_benchmarks": [],
        # custom performance start date (persisted per portfolio)
        "perf_custom_start": None,
        "perf_use_custom":   False,
        "watchlist":         [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-cargar credenciales persistidas
    if not st.session_state.get("anthropic_api_key"):
        loaded = _load_anthropic_key()
        if loaded:
            st.session_state["anthropic_api_key"] = loaded

    if not st.session_state.get("banxico_token"):
        loaded = _load_banxico_token()
        if loaded:
            st.session_state["banxico_token"] = loaded


def transactions_df() -> pd.DataFrame:
    """Retorna el log completo de transacciones como DataFrame."""
    txns = st.session_state.get("transactions", [])
    if not txns:
        return pd.DataFrame(columns=["Ticker","Type","Shares","Price","Date","Notes"])
    df = pd.DataFrame(txns)
    for col in ["Ticker","Type","Notes","Date"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["Shares","Price"]:
        if col not in df.columns:
            df[col] = 0.0
    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)
    df["Price"]  = pd.to_numeric(df["Price"],  errors="coerce").fillna(0)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Type"]   = df["Type"].astype(str).str.upper()
    return df[df["Ticker"] != ""].reset_index(drop=True)


def compute_positions(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega el log de transacciones a posiciones actuales.
    Método: Costo Promedio Ponderado.
    Retorna DataFrame con columnas: Ticker, Shares, AvgCost
    """
    if txns.empty:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost"])

    positions: dict[str, dict] = {}
    for _, row in txns.sort_values("Date").iterrows():
        t      = str(row["Ticker"])
        shares = float(row["Shares"])
        price  = float(row["Price"])
        ttype  = str(row["Type"]).upper()

        if t not in positions:
            positions[t] = {"shares": 0.0, "avg_cost": 0.0}

        if ttype == "BUY":
            old_sh   = positions[t]["shares"]
            old_cost = positions[t]["avg_cost"]
            new_sh   = old_sh + shares
            if new_sh > 0:
                positions[t]["avg_cost"] = (old_cost * old_sh + price * shares) / new_sh
            positions[t]["shares"] = new_sh

        elif ttype == "SELL":
            positions[t]["shares"] = max(0.0, positions[t]["shares"] - shares)
            # El costo promedio NO cambia en ventas (método promedio ponderado)

    rows = [
        {"Ticker": t, "Shares": round(p["shares"], 8), "AvgCost": p["avg_cost"]}
        for t, p in positions.items()
        if p["shares"] > 1e-8
    ]
    if not rows:
        return pd.DataFrame(columns=["Ticker","Shares","AvgCost"])
    return pd.DataFrame(rows)


def compute_realized_pnl(txns: pd.DataFrame) -> dict:
    """
    Calcula P&L realizado completo a partir del historial de transacciones.
    Método: Costo Promedio Ponderado.

    En cada SELL registra:
        P&L realizado = (precio_venta - avg_cost_en_ese_momento) × acciones_vendidas

    Returns:
        {
          'by_ticker': {ticker: {
              'realized_pnl':  float,   # ganancia/pérdida de acciones ya vendidas
              'shares_sold':   float,   # total acciones vendidas en toda la historia
              'proceeds':      float,   # ingresos brutos de todas las ventas
              'cost_sold':     float,   # costo base de lo vendido
              'shares_open':   float,   # acciones aún abiertas
              'avg_cost_open': float,   # costo promedio posición abierta actual
              'is_closed':     bool,    # True si la posición está totalmente cerrada
          }},
          'total_realized': float,      # suma de P&L realizado de todos los tickers
        }
    """
    if txns.empty:
        return {"by_ticker": {}, "total_realized": 0.0}

    state: dict[str, dict] = {}   # {ticker: {shares, avg_cost, realized_pnl, ...}}

    for _, row in txns.sort_values("Date").iterrows():
        t      = str(row["Ticker"])
        shares = float(row["Shares"])
        price  = float(row["Price"])
        ttype  = str(row["Type"]).upper()

        if t not in state:
            state[t] = {
                "shares":       0.0,
                "avg_cost":     0.0,
                "realized_pnl": 0.0,
                "shares_sold":  0.0,
                "proceeds":     0.0,
                "cost_sold":    0.0,
            }

        s = state[t]

        if ttype == "BUY":
            old_sh = s["shares"]
            new_sh = old_sh + shares
            if new_sh > 0:
                s["avg_cost"] = (s["avg_cost"] * old_sh + price * shares) / new_sh
            s["shares"] = new_sh

        elif ttype == "SELL":
            sold = min(shares, s["shares"])   # no vender más de lo que hay
            if sold > 0 and s["avg_cost"] > 0:
                pnl_this = (price - s["avg_cost"]) * sold
                s["realized_pnl"] += pnl_this
                s["proceeds"]     += price * sold
                s["cost_sold"]    += s["avg_cost"] * sold
                s["shares_sold"]  += sold
            s["shares"] = max(0.0, s["shares"] - sold)

    by_ticker = {}
    total_realized = 0.0
    for t, s in state.items():
        by_ticker[t] = {
            "realized_pnl":  round(s["realized_pnl"], 4),
            "shares_sold":   round(s["shares_sold"],  8),
            "proceeds":      round(s["proceeds"],     4),
            "cost_sold":     round(s["cost_sold"],    4),
            "shares_open":   round(s["shares"],       8),
            "avg_cost_open": round(s["avg_cost"],     4),
            "is_closed":     s["shares"] < 1e-8,
        }
        total_realized += s["realized_pnl"]

    return {"by_ticker": by_ticker, "total_realized": round(total_realized, 4)}


def holdings_df() -> pd.DataFrame:
    """
    Posiciones actuales derivadas del log de transacciones.
    Memoizado en session_state usando un hash de las transacciones.
    Evita recalcular en cada re-render de Streamlit.
    """
    txns = st.session_state.get("transactions", [])
    # Build a lightweight cache key from transaction count + last entry
    n = len(txns)
    last = str(txns[-1]) if n > 0 else ""
    cache_key = f"_hdf_cache_{n}_{hash(last)}"

    if cache_key not in st.session_state:
        # Invalidate any old cache entry
        for k in list(st.session_state.keys()):
            if k.startswith("_hdf_cache_"):
                del st.session_state[k]
        st.session_state[cache_key] = compute_positions(transactions_df())

    return st.session_state[cache_key]


def all_tickers() -> list[str]:
    hdf = holdings_df()
    tw  = st.session_state.get("target_weights", {})
    tickers = list(set(
        hdf["Ticker"].astype(str).tolist() +
        list(tw.keys()) +
        [st.session_state.get("benchmark", "SPY")]
    ))
    return [t for t in tickers if t.strip()]


# ─────────────────────────────────────────────────────────────
# COMPONENTES UI
# ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="DM Mono, monospace",
    font_color="#8e8e93",
    margin=dict(t=40, b=40, l=50, r=20),
    legend=dict(
        bgcolor="rgba(28,28,30,0.85)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(color="#aeaeb2", size=11),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, zerolinecolor="rgba(255,255,255,0.08)"),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(28,28,30,0.97)",
        font_color="#ffffff",
        font_family="DM Mono, monospace",
        font_size=12,
        bordercolor="rgba(255,255,255,0.12)",
    ),
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> None:
    accent_style = f"color: {accent};" if accent else ""
    _cls = ""
    if accent == "#30d158":   _cls = "pm-card-green"
    elif accent == "#ff453a": _cls = "pm-card-red"
    elif accent == "#0a84ff": _cls = "pm-card-blue"
    elif accent == "#ffd60a": _cls = "pm-card-amber"
    st.markdown(f"""
    <div class="pm-card {_cls}">
        <div class="pm-card-label">{label}</div>
        <div class="pm-card-value" style="{accent_style}">{value}</div>
        {"<div class='pm-card-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


def badge(value: float, fmt: str = ".2%", threshold: float = 0.0) -> str:
    s = f"{value:{fmt}}"
    if value > threshold:
        return f'<span class="badge-pos">▲ {s}</span>'
    elif value < threshold:
        return f'<span class="badge-neg">▼ {s}</span>'
    else:
        return f'<span class="badge-neu">— {s}</span>'


# ─────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD EN VIVO
# ─────────────────────────────────────────────────────────────


# Keys of PLOTLY_LAYOUT that conflict when we pass custom versions
_LAYOUT_CONFLICT_KEYS = frozenset({
    "template","paper_bgcolor","plot_bgcolor","font_family","font_color",
    "margin","legend","xaxis","yaxis","hovermode","hoverlabel",
})

def _pl(*exclude):
    """Return PLOTLY_LAYOUT minus any keys we want to override."""
    skip = _LAYOUT_CONFLICT_KEYS | set(exclude)
    return {k: v for k, v in PLOTLY_LAYOUT.items() if k not in skip}


def _build_donut_html(df_live: pd.DataFrame, nav: float,
                      chart_colors: list) -> str:
    """
    Donut SVG + leyenda con logos como un único bloque HTML unificado.
    Sin Plotly — control total sobre el layout.
    """
    import math

    total = df_live["Valor"].sum()
    if total <= 0:
        return ""

    sorted_df = df_live.sort_values("Valor", ascending=False).reset_index(drop=True)
    cx, cy, R, ri = 100, 100, 88, 58   # centro, radio externo, radio interno
    angle = -90.0                        # empieza desde arriba

    svg_paths = ""
    legend_rows = ""

    for i, row in sorted_df.iterrows():
        pct   = row["Valor"] / total
        sweep = pct * 360
        clr   = chart_colors[i % len(chart_colors)]
        tick  = str(row["Emisora"])

        # Ángulos con pequeño gap entre sectores
        a1 = math.radians(angle + 0.7)
        a2 = math.radians(angle + sweep - 0.7)
        laf = 1 if sweep > 180 else 0

        x1o, y1o = cx + R  * math.cos(a1), cy + R  * math.sin(a1)
        x2o, y2o = cx + R  * math.cos(a2), cy + R  * math.sin(a2)
        x1i, y1i = cx + ri * math.cos(a1), cy + ri * math.sin(a1)
        x2i, y2i = cx + ri * math.cos(a2), cy + ri * math.sin(a2)

        pd_ = (f"M{x1o:.1f} {y1o:.1f}"
               f"A{R} {R} 0 {laf} 1 {x2o:.1f} {y2o:.1f}"
               f"L{x2i:.1f} {y2i:.1f}"
               f"A{ri} {ri} 0 {laf} 0 {x1i:.1f} {y1i:.1f}Z")
        tooltip = f"{tick} · {pct:.1%} · ${row['Valor']:,.0f}"
        svg_paths += (
            f'<path d="{pd_}" fill="{clr}" '
            f'style="cursor:default;transition:opacity .15s;" '
            f'onmouseover="this.style.opacity=\'0.7\'" '
            f'onmouseout="this.style.opacity=\'1\'">'
            f'<title>{tooltip}</title></path>\n'
        )

        logo = f"https://assets.parqet.com/logos/symbol/{tick.upper()}"
        legend_rows += (
            f'<div style="display:flex;align-items:center;gap:7px;padding:4px 0;">'
            f'<div style="width:8px;height:8px;border-radius:50%;'
            f'background:{clr};flex-shrink:0;"></div>'
            f'<img src="{logo}" width="22" height="22" '
            f'style="border-radius:5px;object-fit:contain;'
            f'background:#1c1c1e;padding:1px;flex-shrink:0;" '
            f'onerror="this.style.display=\'none\'">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
            f'color:#aeaeb2;flex:1;">{tick}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.70rem;'
            f'color:#636366;flex-shrink:0;">{pct:.1%}</span>'
            f'</div>'
        )
        angle += sweep

    nav_str = f"${nav:,.0f}" if nav < 1_000_000 else f"${nav/1_000:.0f}K"
    n = len(sorted_df)

    return (
        f'<div style="background:#000;border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:18px;padding:14px 18px 14px 10px;'
        f'display:flex;align-items:center;gap:4px;">'
        # ── SVG donut ──
        f'<div style="flex-shrink:0;">'
        f'<svg width="190" height="190" viewBox="0 0 200 200">'
        f'{svg_paths}'
        f'<text x="100" y="95" text-anchor="middle" dominant-baseline="middle" '
        f'style="fill:#fff;font-size:16px;font-weight:700;'
        f'font-family:\'DM Mono\',monospace;">{nav_str}</text>'
        f'<text x="100" y="114" text-anchor="middle" '
        f'style="fill:#636366;font-size:10px;font-family:\'DM Mono\',monospace;">'
        f'{n} activos</text>'
        f'</svg></div>'
        # ── Leyenda ──
        f'<div style="flex:1;min-width:0;padding-left:6px;">'
        f'<div style="font-size:0.58rem;color:#636366;font-family:\'DM Mono\',monospace;'
        f'text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Composición</div>'
        f'{legend_rows}'
        f'</div></div>'
    )


def tab_dashboard() -> None:
    hdf = holdings_df()
    if hdf.empty:
        st.info("📂 Agrega posiciones en **Portfolio Editor** para ver el dashboard.")
        return

    tickers = hdf["Ticker"].unique().tolist()
    bench   = st.session_state.get("benchmark", "SPY")

    # ── Cargar todos los datos en UN solo bloque ─────────────
    # fetch_pulse_data y fetch_live_prices están cacheados;
    # usar siempre tuple(sorted(...)) para que el caché se comparta
    # entre la sección de alertas y la de Market Pulse.
    _tickers_key = tuple(sorted(tickers))
    with st.spinner("⚡ Cargando datos de mercado…"):
        prices   = fetch_live_prices(tuple(sorted(tickers + [bench])))
        fx       = get_usd_mxn()
        try:
            pulse_shared = fetch_pulse_data(_tickers_key)
        except Exception:
            pulse_shared = {}
    triggered = check_alerts(hdf, prices, pulse_shared)
    render_alerts_banner(triggered)

    # ── Calcular posiciones ────────────────────────────────────
    rows = []
    nav = cost_total = 0.0
    for _, row in hdf.iterrows():
        t        = str(row["Ticker"])
        shares   = float(row["Shares"])
        avg_cost = float(row["AvgCost"])
        info     = prices.get(t, {})
        price    = info.get("price",      0.0)
        prev     = info.get("prev_close", 0.0)
        imp      = shares * avg_cost
        val      = shares * price
        nav        += val
        cost_total += imp
        rows.append({
            "Emisora": t, "Títulos": shares,
            "Cto. Prom.": avg_cost, "Invertido": imp,
            "Precio": price, "Valor": val,
            "P&L $": val - imp,
            "P&L %": (val - imp) / imp if imp > 0 else 0.0,
            "Var. Día %": (price - prev) / prev if prev > 0 else 0.0,
            "Var. Día $": (price - prev) * shares,
        })

    df_live = pd.DataFrame(rows)
    if df_live.empty:
        st.warning("Sin datos de precios disponibles.")
        return

    df_live["Peso"]   = df_live["Valor"] / nav if nav > 0 else 0.0
    df_live = df_live.sort_values("Valor", ascending=False).reset_index(drop=True)
    tw = st.session_state.get("target_weights", {})
    df_live["Target"] = df_live["Emisora"].map(lambda t: tw.get(t, 0.0))
    df_live["Drift"]  = df_live["Peso"] - df_live["Target"]

    # ── P&L realizado (posiciones cerradas o ventas parciales) ──
    _pnl_data     = compute_realized_pnl(transactions_df())
    _realized_map = _pnl_data["by_ticker"]
    total_realized_pnl = _pnl_data["total_realized"]

    bench_i      = prices.get(bench, {})
    bench_today  = (bench_i.get("price",0) - bench_i.get("prev_close",0)) / bench_i.get("prev_close",1) if bench_i.get("prev_close",0) > 0 else 0.0
    daily_usd    = df_live["Var. Día $"].sum()
    daily_pct    = daily_usd / (nav - daily_usd) if (nav - daily_usd) > 0 else 0.0
    unrealized_pnl = nav - cost_total
    total_pnl      = unrealized_pnl + total_realized_pnl
    total_ret    = total_pnl / cost_total if cost_total > 0 else 0.0
    alpha_today  = daily_pct - bench_today

    # ── Métricas de riesgo (cached) ─────────────────────────
    @st.cache_data(ttl=900, show_spinner=False)
    def _risk(tickers_t, weights_t, bench_t):
        end = datetime.today(); start = end - timedelta(days=365)
        h = fetch_history(list(tickers_t) + [bench_t],
                          start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if h.empty: return {}
        tl = [t for t in tickers_t if t in h.columns]
        if not tl: return {}
        wd = dict(zip(tickers_t, weights_t))
        w  = np.array([wd.get(t,0) for t in tl]); w /= max(w.sum(),1e-9)
        pr = h[tl].pct_change().dropna(how="all").fillna(0).values @ w
        ar = float((1+pr).prod()**(252/max(len(pr),1)) - 1)
        av = float(pr.std() * np.sqrt(252))
        sh = (ar - 0.045) / av if av > 0 else 0
        v95  = float(np.percentile(pr, 5))
        cv95 = float(pr[pr<=v95].mean()) if len(pr[pr<=v95]) > 0 else v95
        cum  = (1+pr).cumprod()
        dd   = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        mdd  = float(dd.min())
        beta = None
        if bench_t in h.columns:
            br = h[bench_t].pct_change().dropna()
            idx = h[tl].pct_change().dropna().index
            br2 = br.reindex(idx).dropna()
            if len(br2) > 30:
                cv = np.cov(pr[:len(br2)], br2.values)
                bv = np.var(br2.values, ddof=1)
                beta = round(cv[0,1]/bv, 2) if bv > 0 else None
        return {"ar":ar,"av":av,"sh":sh,"v95":v95,"cv95":cv95,"mdd":mdd,"beta":beta}

    risk = _risk(tuple(df_live["Emisora"]), tuple(df_live["Peso"]), bench)

    # ── Colores ───────────────────────────────────────────────
    gc = lambda v, rev=False: ("#30d158" if (v>=0)^rev else "#ff453a")
    ic = lambda v: "▲" if v >= 0 else "▼"

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 1: HERO HEADER (HTML auto-contenido)
    # ═══════════════════════════════════════════════════════════
    def _build_monthly_hero(data: dict) -> str:
        if not data:
            return ""
        _mn = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        _years = sorted(data.keys())
        rows_html = ""
        for _yr in _years:
            cells = ""
            for _mo in range(1, 13):
                _rv = data[_yr].get(_mo)
                if _rv is None:
                    cells += (
                        f'<td style="padding:5px 0;text-align:center;'
                        f'font-family:DM Mono,monospace;font-size:0.65rem;'
                        f'color:#2c2c2e;border-radius:5px;">—</td>'
                    )
                else:
                    # Color: verde → rojo lineal, máximo visual en ±10%
                    _pct = max(-1, min(1, _rv / 0.10))
                    if _rv >= 0:
                        _a   = min(0.85, 0.15 + _pct * 0.70)
                        _bg  = f"rgba(48,209,88,{_a:.2f})"
                        _txt = "#000000" if _a > 0.45 else "#30d158"
                    else:
                        _a   = min(0.85, 0.15 + abs(_pct) * 0.70)
                        _bg  = f"rgba(255,69,58,{_a:.2f})"
                        _txt = "#000000" if _a > 0.45 else "#ff453a"
                    cells += (
                        f'<td style="padding:5px 4px;text-align:center;'
                        f'background:{_bg};border-radius:5px;'
                        f'font-family:DM Mono,monospace;font-size:0.68rem;'
                        f'font-weight:600;color:{_txt};white-space:nowrap;">'
                        f'{_rv:+.1%}</td>'
                    )
            rows_html += (
                f'<tr>'
                f'<td style="padding:5px 10px 5px 0;font-family:DM Mono,monospace;'
                f'font-size:0.65rem;color:#636366;white-space:nowrap;">{_yr}</td>'
                f'{cells}</tr>'
            )
        header_cells = "".join(
            f'<th style="padding:0 4px 6px;text-align:center;font-family:DM Mono,monospace;'
            f'font-size:0.6rem;font-weight:600;color:#48484a;letter-spacing:0.5px;">{m}</th>'
            for m in _mn
        )
        return (
            f'<div style="background:#111113;border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:18px;padding:18px 24px 16px;margin-bottom:10px;">'
            f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:1.5px;'
            f'color:#48484a;text-transform:uppercase;font-family:DM Mono,monospace;'
            f'margin-bottom:10px;">Retornos Mensuales</div>'
            f'<table style="border-collapse:separate;border-spacing:3px 0;width:100%;">'
            f'<thead><tr><th style="width:36px"></th>{header_cells}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>'
        )

    def _hero_stat(label, value, color, sub=""):
        return (
            f'<div style="text-align:right;">'
            f'<div style="font-size:0.6rem;color:#636366;text-transform:uppercase;'
            f'letter-spacing:1.5px;font-family:DM Mono,monospace;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:1.9rem;font-weight:700;color:{color};'
            f'font-family:DM Mono,monospace;line-height:1;letter-spacing:-0.5px;">{value}</div>'
            + (f'<div style="font-size:0.8rem;color:{color};opacity:0.7;'
               f'font-family:DM Mono,monospace;margin-top:2px;">{sub}</div>' if sub else "")
            + '</div>'
        )

    def _hero_chip(label, value, color):
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:12px;padding:8px 14px;min-width:80px;">'
            f'<span style="font-size:0.58rem;color:#636366;text-transform:uppercase;'
            f'letter-spacing:1px;font-family:DM Mono,monospace;margin-bottom:3px;">{label}</span>'
            f'<span style="font-size:0.9rem;font-weight:600;color:{color};'
            f'font-family:DM Mono,monospace;">{value}</span></div>'
        )

    sh_c  = "#30d158" if risk.get("sh",0)>1 else ("#ffd60a" if risk.get("sh",0)>0 else "#ff453a")
    chips = "".join([
        _hero_chip("Sharpe 1A",  f"{risk.get('sh',0):.2f}",  sh_c),
        _hero_chip("Vol. 1A",    f"{risk.get('av',0):.1%}",  "#ffd60a"),
        _hero_chip("Max DD 1A",  f"{risk.get('mdd',0):.1%}", "#ff453a"),
        _hero_chip("VaR 95%",    f"{risk.get('v95',0):.2%}", "#ff453a"),
        _hero_chip("Beta",       str(risk.get("beta","—")),   "#bf5af2"),
        _hero_chip("Posiciones", str(len(df_live)),            "#0a84ff"),
    ])

    _dc = gc(daily_usd); _tc = gc(total_pnl); _ac = gc(alpha_today)

    st.markdown(f"""
<style>
@keyframes _fi{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.pm-hero{{animation:_fi .4s ease-out both}}
</style>
<div class="pm-hero" style="
  background:#111113;
  border:1px solid rgba(255,255,255,0.08);border-radius:22px;
  padding:28px 32px 22px;margin-bottom:12px;position:relative;overflow:hidden;
  box-shadow:0 1px 0 rgba(255,255,255,0.04) inset,0 16px 48px rgba(0,0,0,.5);">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              flex-wrap:wrap;gap:20px;margin-bottom:22px;">
    <!-- NAV principal — estilo Apple Stocks -->
    <div>
      <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px;">
        <span class="live-dot"></span>
        <span style="font-size:0.62rem;font-weight:600;letter-spacing:1.8px;color:#636366;
                     text-transform:uppercase;font-family:DM Mono,monospace;">VALOR DE PORTAFOLIO</span>
      </div>
      <div style="font-size:3.4rem;font-weight:700;color:#ffffff;line-height:1;
                  letter-spacing:-2px;font-family:DM Sans,sans-serif;">${nav:,.2f}</div>
      <div style="font-size:0.8rem;color:#48484a;margin-top:8px;font-family:DM Mono,monospace;">
        <span style="color:#636366;">≈ ${nav*fx:,.0f} MXN</span>
        &ensp;·&ensp;
        <span>Invertido ${cost_total:,.0f}</span>
      </div>
    </div>
    <!-- Tres métricas clave -->
    <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;padding-top:4px;">
      {_hero_stat("Hoy", f"{ic(daily_usd)} ${abs(daily_usd):,.2f}", _dc, f"{daily_pct:+.2%}")}
      {_hero_stat("P&amp;L Total", f"{ic(total_pnl)} ${abs(total_pnl):,.2f}", _tc, f"{total_ret:+.2%}")}
      {_hero_stat(f"α vs {bench}", f"{alpha_today:+.2%}", _ac, f"{bench}: {bench_today:+.2%}")}
    </div>
  </div>

  <!-- Divisor -->
  <div style="height:1px;background:rgba(255,255,255,0.07);margin-bottom:16px;"></div>

  <!-- Chips de riesgo -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;">{chips}</div>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 2: GRÁFICA RENDIMIENTO + DONUT
    # Usa transacciones reales para construir el historial correcto
    # ═══════════════════════════════════════════════════════════
    _dash_txns = transactions_df()
    _end_dash  = datetime.today()

    # Ventana: desde primera transacción o 120 días, lo que sea más largo
    _first_txn_dash = None
    if not _dash_txns.empty:
        try:
            _first_txn_dash = pd.to_datetime(_dash_txns["Date"]).min()
        except Exception:
            _first_txn_dash = None
    _start_dash = _first_txn_dash if _first_txn_dash else (_end_dash - timedelta(days=120))
    # No más de 18 meses para que el dashboard siga siendo rápido
    if (_end_dash - _start_dash).days > 540:
        _start_dash = _end_dash - timedelta(days=540)

    # Todos los tickers que hayan aparecido en transacciones (incluidos los vendidos)
    # son necesarios para calcular el NAV histórico correcto.
    _all_txn_t = (
        _dash_txns["Ticker"].unique().tolist()
        if not _dash_txns.empty else []
    )
    _all_dash_t = list(dict.fromkeys(_all_txn_t + tickers + [bench]))
    _h_dash = fetch_history(
        _all_dash_t,
        _start_dash.strftime("%Y-%m-%d"),
        _end_dash.strftime("%Y-%m-%d"),
    )

    port_c, bench_c = None, None
    if not _h_dash.empty:
        # TWR — mismo cálculo que tab Performance, sin efecto de aportaciones
        _twr_dash = build_twr_series(_dash_txns, _h_dash)
        if not _twr_dash.empty:
            port_c = _twr_dash
        else:
            # Fallback si no hay transacciones con fecha válida
            _eq_dash = build_portfolio_equity_from_transactions(_dash_txns, _h_dash)
            if _eq_dash.empty:
                _tl = [t for t in tickers if t in _h_dash.columns]
                _wd = dict(zip(df_live["Emisora"], df_live["Peso"]))
                _w  = np.array([_wd.get(t, 0) for t in _tl])
                _w /= max(_w.sum(), 1e-9)
                _pr = _h_dash[_tl].pct_change().dropna(how="all").fillna(0)
                _eq_dash = pd.Series((_pr.values @ _w + 1).cumprod() * 100,
                                     index=_pr.index)
            if not _eq_dash.empty:
                port_c = _eq_dash / _eq_dash.iloc[0] * 100

        if bench in _h_dash.columns and port_c is not None:
            _bs = _h_dash[bench].reindex(port_c.index).ffill().bfill()
            if not _bs.empty and _bs.iloc[0] > 0:
                bench_c = _bs / _bs.iloc[0] * 100

    # ── Retornos mensuales ────────────────────────────────────
    _monthly_hero: dict[int, dict[int, float]] = {}  # {year: {month: ret}}
    if port_c is not None and len(port_c) > 5:
        _dr_hero = port_c.pct_change().dropna()
        _mr_hero = ((_dr_hero + 1).resample("ME").prod() - 1)
        for _ts, _rv in _mr_hero.items():
            _y, _m = _ts.year, _ts.month
            _monthly_hero.setdefault(_y, {})[_m] = float(_rv)

    # Renderizar mini heatmap como extensión del hero (misma estética)
    if _monthly_hero:
        st.markdown(_build_monthly_hero(_monthly_hero), unsafe_allow_html=True)

    col_line, col_donut = st.columns([3, 2], gap="small")

    with col_line:
        if port_c is not None and len(port_c) > 0:
            fig_l = go.Figure()

            # Rango Y ajustado a los datos reales (sin incluir el 0)
            _all_vals = list(port_c.values)
            if bench_c is not None:
                _all_vals += list(bench_c.values)
            _y_min = min(_all_vals)
            _y_max = max(_all_vals)
            _y_pad = max((_y_max - _y_min) * 0.10, 2)
            _y_lo, _y_hi = _y_min - _y_pad, _y_max + _y_pad

            # Benchmark primero (atrás)
            if bench_c is not None:
                fig_l.add_trace(go.Scatter(
                    x=bench_c.index, y=bench_c.values, name=bench,
                    mode="lines",
                    line=dict(color="rgba(142,142,147,0.5)", width=1.2, dash="dot"),
                    hovertemplate=f"<b>{bench}</b>: %{{y:.1f}}<extra></extra>",
                ))

            # Fill que imita el gradiente de Apple Stocks sobre fondo negro
            _line_clr = "#0a84ff" if float(port_c.iloc[-1]) >= 100 else "#ff453a"
            _fill_clr = "rgba(10,132,255,0.18)" if _line_clr == "#0a84ff" else "rgba(255,69,58,0.18)"
            fig_l.add_trace(go.Scatter(
                x=port_c.index, y=port_c.values, name="Portafolio (TWR)",
                mode="lines",
                line=dict(color=_line_clr, width=2),
                fill="tozeroy", fillcolor=_fill_clr,
                hovertemplate="TWR: <b>%{y:.1f}</b><extra></extra>",
            ))

            # Punto final destacado
            fig_l.add_trace(go.Scatter(
                x=[port_c.index[-1]], y=[float(port_c.iloc[-1])],
                mode="markers", showlegend=False,
                marker=dict(color=_line_clr, size=7,
                            line=dict(color="#ffffff", width=1.5)),
                hoverinfo="skip",
            ))

            # Anotación del delta final
            _final_port = float(port_c.iloc[-1])
            _anno_clr   = "#30d158" if _final_port >= 100 else "#ff453a"
            fig_l.add_annotation(
                x=port_c.index[-1], y=_final_port,
                text=f"<b>{_final_port - 100:+.1f}%</b>",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=12, color=_anno_clr, family="DM Mono"),
                xshift=8,
            )

            # Línea base 100
            fig_l.add_hline(y=100, line_width=1, line_dash="dot",
                            line_color="rgba(255,255,255,0.12)")

            fig_l.update_layout(
                **_pl(),
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                title=dict(text="Rendimiento TWR (base 100)",
                           font=dict(size=11, color="#636366", family="DM Mono"), x=0),
                height=308, margin=dict(t=40, b=32, l=8, r=52),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0,
                            font=dict(size=10, color="#8e8e93"),
                            bgcolor="rgba(0,0,0,0)"),
            )
            fig_l.update_xaxes(showgrid=False, zeroline=False,
                               tickfont=dict(size=9, color="#636366", family="DM Mono"),
                               tickcolor="rgba(255,255,255,0.06)")
            fig_l.update_yaxes(range=[_y_lo, _y_hi],
                               showgrid=True,
                               gridcolor="rgba(255,255,255,0.04)",
                               zeroline=False,
                               tickfont=dict(size=9, color="#636366", family="DM Mono"))
            st.plotly_chart(fig_l, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Sin datos históricos aún. Agrega transacciones con fecha para ver la curva.")

    CHART_COLORS = ["#0a84ff","#30d158","#ffd60a","#a78bfa",
                    "#fb923c","#f472b6","#38bdf8","#4ade80",
                    "#facc15","#c084fc","#ff453a","#67e8f9"]

    with col_donut:
        _donut_html = _build_donut_html(df_live, nav, CHART_COLORS)
        if _donut_html:
            st.markdown(_donut_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 3: P&L BARRAS + DRIFT
    # ═══════════════════════════════════════════════════════════
    col_bar, col_drift_c = st.columns([3, 2], gap="small")

    with col_bar:
        # ── Combinar P&L abierto + realizado para cada ticker ────
        # Posiciones abiertas: P&L no realizado actual
        pnl_rows = []
        for _, r in df_live.iterrows():
            t = r["Emisora"]
            unreal = r["P&L $"]
            real   = _realized_map.get(t, {}).get("realized_pnl", 0.0)
            pnl_rows.append({
                "Ticker": t, "No realizado": unreal,
                "Realizado": real, "Total": unreal + real,
            })
        # Posiciones totalmente cerradas (no están en df_live)
        open_set = set(df_live["Emisora"])
        for t, info in _realized_map.items():
            if t not in open_set and info["is_closed"] and abs(info["realized_pnl"]) > 0.01:
                pnl_rows.append({
                    "Ticker": t, "No realizado": 0.0,
                    "Realizado": info["realized_pnl"],
                    "Total": info["realized_pnl"],
                })

        df_pnl = pd.DataFrame(pnl_rows).sort_values("Total")

        fig_b = go.Figure()
        # P&L no realizado — barra principal, colores sólidos Apple
        _unr_colors = ["#30d158" if v >= 0 else "#ff453a" for v in df_pnl["No realizado"]]
        fig_b.add_trace(go.Bar(
            name="No realizado",
            x=df_pnl["Ticker"], y=df_pnl["No realizado"],
            marker=dict(
                color=_unr_colors,
                opacity=0.92,
                line=dict(width=0),
                cornerradius=4,
            ),
            hovertemplate="<b>%{x}</b><br>No realizado: <b>$%{y:+,.2f}</b><extra></extra>",
        ))
        # P&L realizado — semitransparente encima
        if df_pnl["Realizado"].abs().sum() > 0:
            _rel_colors = ["rgba(48,209,88,0.55)" if v >= 0 else "rgba(255,69,58,0.55)"
                           for v in df_pnl["Realizado"]]
            fig_b.add_trace(go.Bar(
                name="Realizado (ventas)",
                x=df_pnl["Ticker"], y=df_pnl["Realizado"],
                marker=dict(color=_rel_colors, line=dict(width=0), cornerradius=4),
                hovertemplate="<b>%{x}</b><br>Realizado: <b>$%{y:+,.2f}</b><extra></extra>",
            ))

        # Valor total encima de cada barra
        for _, _row in df_pnl.iterrows():
            _tot = _row["Total"]
            if abs(_tot) > 0.5:
                fig_b.add_annotation(
                    x=_row["Ticker"], y=_tot,
                    text=f"${_tot:+.0f}",
                    showarrow=False, yanchor="bottom" if _tot >= 0 else "top",
                    yshift=4 if _tot >= 0 else -4,
                    font=dict(size=9, color="#8e8e93", family="DM Mono"),
                )

        fig_b.update_layout(
            **_pl(),
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            barmode="stack",
            bargap=0.32,
            title=dict(text="P&L por posición",
                       font=dict(size=11, color="#636366", family="DM Mono"), x=0),
            height=308, margin=dict(t=40, b=20, l=8, r=10),
            legend=dict(orientation="h", y=1.08, x=0,
                        font=dict(size=10, color="#8e8e93"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig_b.update_xaxes(showgrid=False, zeroline=False,
                           tickfont=dict(size=10, color="#8e8e93", family="DM Mono"),
                           tickcolor="rgba(255,255,255,0.05)")
        fig_b.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,0.12)",
                           tickfont=dict(size=9, color="#636366", family="DM Mono"),
                           tickprefix="$")
        st.plotly_chart(fig_b, use_container_width=True, config=PLOTLY_CONFIG)

    with col_drift_c:
        df_dr = df_live[df_live["Drift"].abs() > 0.001]
        if tw and not df_dr.empty:
            df_dr2 = df_dr.sort_values("Drift")
            _dr_colors = ["#30d158" if v > 0 else "#ff453a" for v in df_dr2["Drift"]]
            fig_dr = go.Figure(go.Bar(
                x=df_dr2["Drift"] * 100, y=df_dr2["Emisora"],
                orientation="h",
                marker=dict(color=_dr_colors, opacity=0.88,
                            line=dict(width=0), cornerradius=4),
                hovertemplate="<b>%{y}</b>: %{x:+.1f}pp<extra></extra>",
                text=[f"{v:+.1f}pp" for v in df_dr2["Drift"] * 100],
                textposition="outside",
                textfont=dict(size=10, color="#8e8e93", family="DM Mono"),
            ))
            fig_dr.update_layout(
                **_pl(),
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                title=dict(text="Drift vs target",
                           font=dict(size=11, color="#636366", family="DM Mono"), x=0),
                height=308, margin=dict(t=40, b=20, l=10, r=52),
                showlegend=False,
                hovermode="y unified",
            )
            fig_dr.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                                zeroline=True, zerolinecolor="rgba(255,255,255,0.14)",
                                tickfont=dict(size=9, color="#636366", family="DM Mono"),
                                ticksuffix="pp")
            fig_dr.update_yaxes(showgrid=False, zeroline=False,
                                tickfont=dict(size=10, color="#8e8e93", family="DM Mono"))
            st.plotly_chart(fig_dr, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Configura target weights para ver el drift.")

    # ═══════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 3.5: MARKET PULSE — Movimiento diario de cada acción
    # ═══════════════════════════════════════════════════════════
    # ── Refresh button + header + toggle ─────────────────────
    hdr_col, tog_col, btn_col = st.columns([4, 1, 1])
    with hdr_col:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:1.2px;"
            "color:var(--accent, #0a84ff);text-transform:uppercase;"
            "font-family:DM Mono,monospace;margin:18px 0 10px;position:relative;"
            "padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07);'>"
            "<span class='live-dot'></span>MARKET PULSE — MOVIMIENTO DIARIO"
            "<div style='position:absolute;bottom:-1px;left:0;width:36px;height:2px;"
            "background:linear-gradient(90deg,#0a84ff,rgba(10,132,255,0));border-radius:2px;'>"
            "</div></div>",
            unsafe_allow_html=True)
    with tog_col:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        _pulse_table_mode = st.toggle("Tabla", key="pulse_table_mode",
                                      help="Alternar entre tarjetas y tabla compacta")
    with btn_col:
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="pulse_refresh",
                     help="Actualiza los precios del Market Pulse ahora mismo"):
            fetch_pulse_data.clear()
            fetch_live_prices.clear()
            st.rerun()

    # Reusar los datos ya descargados arriba (misma cache key)
    pulse = pulse_shared if pulse_shared else fetch_pulse_data(_tickers_key)

    def _rsi_label(rsi):
        if rsi is None: return "—", "#8e8e93"
        if rsi >= 70:   return f"RSI {rsi:.0f} ↑ OC", "#ff453a"
        if rsi <= 30:   return f"RSI {rsi:.0f} ↓ OS", "#30d158"
        if rsi >= 55:   return f"RSI {rsi:.0f} ▲",    "#86efac"
        return             f"RSI {rsi:.0f} ▼",         "#aeaeb2"

    def _vol_label(vr):
        if vr is None: return ""
        if vr >= 2.0:  return f"Vol ×{vr:.1f} 🔥"
        if vr >= 1.4:  return f"Vol ×{vr:.1f} ↑"
        if vr <= 0.5:  return f"Vol ×{vr:.1f} ↓"
        return             f"Vol ×{vr:.1f}"

    # Build one HTML card per stock, in a responsive grid
    cards_html = ""
    card_w = max(1, min(4, len(tickers)))  # 1-4 columns

    for _, row_data in df_live.iterrows():
        t      = str(row_data["Emisora"])
        pd_    = pulse.get(t, {})
        chg_prev = pd_.get("change_vs_prev", pd_.get("change_1d", 0.0)) or 0.0
        chg_open = pd_.get("change_vs_open", 0.0) or 0.0
        chg1w  = pd_.get("change_1w")
        rsi    = pd_.get("rsi14")
        sma    = pd_.get("above_sma20")
        vr     = pd_.get("vol_ratio")
        spark  = pd_.get("spark_prices", [])
        price  = pd_.get("price", float(row_data["Precio"]))

        # Card color based on intraday move (vs open), more accurate
        chg    = chg_open if pd_.get("day_open", 0) > 0 else chg_prev
        clr    = "#30d158" if chg >= 0 else "#ff453a"
        bg     = "rgba(48,209,88,0.04)" if chg >= 0 else "rgba(255,69,58,0.04)"
        brd    = "rgba(48,209,88,0.15)" if chg >= 0 else "rgba(255,69,58,0.15)"
        arr    = "▲" if chg >= 0 else "▼"

        rsi_txt, rsi_clr = _rsi_label(rsi)
        vol_txt = _vol_label(vr)

        sma_tag = ""
        if sma is True:
            sma_tag = "<span style='color:#86efac;font-size:0.68rem;'>↑ SMA20</span>"
        elif sma is False:
            sma_tag = "<span style='color:#ff453a;font-size:0.68rem;'>↓ SMA20</span>"

        chg1w_tag = ""
        if chg1w is not None:
            c1w = "#30d158" if chg1w >= 0 else "#ff453a"
            chg1w_tag = (f"<span style='color:{c1w};font-size:0.72rem;"
                         f"font-family:DM Mono,monospace;'>1S: {chg1w:+.1%}</span>")

        # ── Mini SVG sparkline (180×54 — 50% más grande que antes) ─
        spark_svg = ""
        if len(spark) >= 4:
            w_svg, h_svg = 180, 54
            lo, hi = min(spark), max(spark)
            rng = hi - lo if hi != lo else 1.0
            pts = " ".join(
                f"{i*(w_svg/(len(spark)-1)):.1f},{h_svg - (v-lo)/rng*(h_svg-4):.1f}"
                for i, v in enumerate(spark)
            )
            spark_clr = "#30d158" if chg >= 0 else "#ff453a"
            fill_id   = f"sfill_{t.replace('-','')}"
            fill_clr_a = "rgba(48,209,88,0.18)" if chg >= 0 else "rgba(255,69,58,0.12)"
            pts_fill = (f"0,{h_svg} " + pts + f" {w_svg},{h_svg}")
            # Add a horizontal baseline at last price
            last_y = f"{h_svg - (spark[-1]-lo)/rng*(h_svg-4):.1f}"
            spark_svg = (
                f'<svg width="100%" height="{h_svg}" viewBox="0 0 {w_svg} {h_svg}" '
                f'preserveAspectRatio="none" '
                f'style="display:block;margin:8px 0 4px;overflow:visible;">'
                f'<defs><linearGradient id="{fill_id}" x1="0" y1="0" x2="0" y2="1">'
                f'<stop offset="0%" stop-color="{spark_clr}" stop-opacity=".22"/>'
                f'<stop offset="100%" stop-color="{spark_clr}" stop-opacity="0"/>'
                f'</linearGradient></defs>'
                f'<polyline points="{pts_fill}" fill="url(#{fill_id})" stroke="none"/>'
                f'<polyline points="{pts}" fill="none" stroke="{spark_clr}" '
                f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
                f'<circle cx="{w_svg}" cy="{last_y}" r="3" fill="{spark_clr}" opacity=".9"/>'
                f'</svg>'
            )

        # ── Colores según dirección ─────────────────────────────
        prev_clr = "#30d158" if chg_prev >= 0 else "#ff453a"
        prev_arr = "▲" if chg_prev >= 0 else "▼"
        open_clr = "#30d158" if chg_open >= 0 else "#ff453a"
        open_arr = "▲" if chg_open >= 0 else "▼"
        has_open = pd_.get("day_open", 0) > 0
        pnl_val  = float(row_data["P&L $"])
        pnl_clr  = "#30d158" if pnl_val >= 0 else "#ff453a"
        weight_w = min(100, int(float(row_data["Peso"]) * 100 * 4))  # scale for bar

        # RSI badge with background
        rsi_badge = (
            f"<span style='background:rgba({('48,209,88' if rsi_clr=='#30d158' or rsi_clr=='#86efac' else ('255,69,58' if rsi_clr=='#ff453a' else '142,142,147'))},.12);"
            f"color:{rsi_clr};border:1px solid {rsi_clr}22;border-radius:5px;"
            f"padding:1px 7px;font-size:0.68rem;font-family:DM Mono,monospace;"
            f"font-weight:700;'>{rsi_txt}</span>"
            if rsi is not None else ""
        )
        vol_badge = (
            f"<span style='color:#8e8e93;font-size:0.68rem;font-family:DM Mono,monospace;"
            f"background:rgba(255,255,255,0.04);border-radius:5px;padding:1px 6px;'>{vol_txt}</span>"
            if vol_txt else ""
        )

        open_row = (
            f"<div style='font-size:0.68rem;color:{open_clr};font-family:DM Mono,monospace;"
            f"margin-top:1px;opacity:.85;'>{open_arr} {abs(chg_open):.2%} "
            f"<span style='color:#636366;font-size:.6rem;'>apertura</span></div>"
            if has_open else ""
        )

        cards_html += f"""
<div class="mp-card" style="background:{bg};border:1px solid {brd};border-radius:18px;
            padding:16px 18px 14px;display:flex;flex-direction:column;gap:0;min-width:0;">

  <!-- Fila superior: ticker + cambio vs ayer -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:2px;">
    <div>
      <div style="font-family:DM Mono,monospace;font-weight:900;font-size:0.95rem;
                  color:#ffffff;letter-spacing:0.2px;line-height:1;">{t}</div>
      <div style="font-family:DM Mono,monospace;font-size:1.5rem;font-weight:700;
                  color:#fff;line-height:1.1;margin-top:3px;letter-spacing:-1px;">
        ${price:,.2f}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:DM Mono,monospace;font-size:1.3rem;font-weight:700;
                  color:{prev_clr};line-height:1;">{prev_arr} {abs(chg_prev):.2%}</div>
      <div style="font-size:0.6rem;color:#636366;margin-top:1px;font-family:DM Mono,monospace;">
        vs cierre anterior</div>
      {open_row}
    </div>
  </div>

  <!-- Sparkline -->
  {spark_svg}

  <!-- Separador -->
  <div style="height:1px;background:rgba(255,255,255,0.05);margin:4px 0;"></div>

  <!-- Indicadores técnicos -->
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:4px;">
    {rsi_badge}
    {chg1w_tag}
    {sma_tag}
    {vol_badge}
  </div>

  <!-- P&L + peso en cartera -->
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-top:8px;">
    <div>
      <div style="font-size:0.62rem;color:#636366;font-family:DM Mono,monospace;
                  text-transform:uppercase;letter-spacing:.5px;">P&amp;L</div>
      <div style="font-family:DM Mono,monospace;font-weight:700;font-size:0.88rem;
                  color:{pnl_clr};">${pnl_val:+,.2f}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.62rem;color:#636366;font-family:DM Mono,monospace;
                  text-transform:uppercase;letter-spacing:.5px;">CARTERA</div>
      <div style="font-family:DM Mono,monospace;font-weight:700;font-size:0.88rem;
                  color:#aeaeb2;">{float(row_data["Peso"]):.1%}</div>
    </div>
  </div>

  <!-- Barra de peso en cartera -->
  <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:6px;overflow:hidden;">
    <div style="height:100%;width:{weight_w}%;background:linear-gradient(90deg,{clr},rgba(255,255,255,.1));
                border-radius:2px;transition:width .4s ease;"></div>
  </div>

</div>"""

    if _pulse_table_mode:
        # ── Modo tabla compacta ───────────────────────────────
        _tbl_rows = ""
        for _, _rd in df_live.iterrows():
            _tt   = str(_rd["Emisora"])
            _ppd  = pulse.get(_tt, {})
            _pc   = _ppd.get("change_vs_prev", _ppd.get("change_1d", 0.0)) or 0.0
            _pr   = _ppd.get("price", float(_rd["Precio"]))
            _rsi  = _ppd.get("rsi14")
            _pnl  = float(_rd["P&L $"])
            _pnlp = float(_rd["P&L %"])
            _peso = float(_rd["Peso"])
            _dc   = "#30d158" if _pc >= 0 else "#ff453a"
            _pc_arr = "▲" if _pc >= 0 else "▼"
            _pnl_c = "#30d158" if _pnl >= 0 else "#ff453a"
            _rsi_c = ("#ff453a" if (_rsi or 50) >= 70 else
                      "#30d158" if (_rsi or 50) <= 30 else "#8e8e93")
            _rsi_s = f"{_rsi:.0f}" if _rsi else "—"
            _tbl_rows += f"""
<tr onmouseover="this.style.background='rgba(255,255,255,0.025)'"
    onmouseout="this.style.background='transparent'"
    style="border-bottom:1px solid rgba(255,255,255,0.04);transition:background .15s;">
  <td style="padding:10px 16px;font-family:DM Mono,monospace;font-weight:800;
             font-size:0.88rem;color:#0a84ff;white-space:nowrap;">{_tt}</td>
  <td style="padding:10px 16px;font-family:DM Mono,monospace;font-weight:700;
             font-size:0.88rem;color:#ffffff;white-space:nowrap;">${_pr:,.2f}</td>
  <td style="padding:10px 12px;font-family:DM Mono,monospace;font-size:0.88rem;
             font-weight:700;color:{_dc};white-space:nowrap;">
    {_pc_arr} {abs(_pc):.2%}</td>
  <td style="padding:10px 12px;font-family:DM Mono,monospace;font-size:0.84rem;
             font-weight:700;color:{_pnl_c};white-space:nowrap;">
    {_pnlp*100:+.2f}%</td>
  <td style="padding:10px 12px;font-family:DM Mono,monospace;font-size:0.84rem;
             font-weight:700;color:{_rsi_c};white-space:nowrap;">RSI {_rsi_s}</td>
  <td style="padding:10px 16px;min-width:110px;">
    <div style="display:flex;align-items:center;gap:7px;">
      <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:3px;height:3px;">
        <div style="width:{min(100,_peso*100*4):.0f}%;height:100%;
                    background:#0a84ff;border-radius:3px;"></div>
      </div>
      <span style="font-size:0.74rem;color:#8e8e93;font-family:DM Mono,monospace;
                   white-space:nowrap;">{_peso:.1%}</span>
    </div>
  </td>
</tr>"""
        _th2 = ("padding:9px 16px;text-align:left;font-family:DM Mono,monospace;"
                "font-size:0.67rem;font-weight:700;color:#636366;text-transform:uppercase;"
                "letter-spacing:.9px;border-bottom:1px solid rgba(255,255,255,0.07);"
                "background:rgba(255,255,255,0.025);white-space:nowrap;")
        st.markdown(f"""
<div style="border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);
            margin-bottom:16px;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>
      <th style="{_th2}">Ticker</th>
      <th style="{_th2}">Precio</th>
      <th style="{_th2}">Hoy</th>
      <th style="{_th2}">P&amp;L</th>
      <th style="{_th2}">RSI</th>
      <th style="{_th2}">Peso</th>
    </tr></thead>
    <tbody>{_tbl_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)
    else:
        # ── Modo tarjetas (grid) ──────────────────────────────
        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
            gap:14px;margin-bottom:16px;">
  {cards_html}
</div>
""", unsafe_allow_html=True)

    # SECCIÓN 4: TABLA DE POSICIONES
    # ═══════════════════════════════════════════════════════════
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:1.2px;"
        "color:#0a84ff;text-transform:uppercase;font-family:DM Mono,monospace;"
        "margin:20px 0 10px;position:relative;padding-bottom:8px;"
        "border-bottom:1px solid rgba(255,255,255,0.07);'>"
        "MIS POSICIONES"
        "<div style='position:absolute;bottom:-1px;left:0;width:36px;height:2px;"
        "background:linear-gradient(90deg,#0a84ff,rgba(10,132,255,0));border-radius:2px;'>"
        "</div></div>",
        unsafe_allow_html=True)

    # ── HTML table with inline mini-sparklines ────────────────
    _rows_pos = ""
    for _, _rd in df_live.iterrows():
        _t       = str(_rd["Emisora"])
        _pd      = pulse.get(_t, {})
        _spark   = _pd.get("spark_prices", [])
        _price   = float(_rd["Precio"])
        _pnl_pct = float(_rd["P&L %"])
        _day_pct = float(_rd["Var. Día %"])
        _peso    = float(_rd["Peso"])
        _valor   = float(_rd["Valor"])
        _pnl_usd = float(_rd["P&L $"])
        _shares  = float(_rd["Títulos"])
        _avg     = float(_rd["Cto. Prom."])
        _drift   = float(_rd.get("Drift", 0) or 0)

        _pnl_rgb  = "48,209,88" if _pnl_pct >= 0 else "255,69,58"
        _pnl_clr  = "#30d158"  if _pnl_pct >= 0 else "#ff453a"
        _day_clr  = "#30d158"  if _day_pct >= 0 else "#ff453a"
        _day_arr  = "▲"        if _day_pct >= 0 else "▼"
        _drclr    = ("#30d158" if _drift > 0.005 else
                     "#ff453a" if _drift < -0.005 else "#8e8e93")

        _mini = ""
        if len(_spark) >= 4:
            _ws, _hs = 80, 28
            _lo, _hi = min(_spark), max(_spark)
            _rng = _hi - _lo if _hi != _lo else 1.0
            _sc  = "#30d158" if _day_pct >= 0 else "#ff453a"
            _fid = f"spf_{_t.replace('-','').replace('.','')}"
            _pts = " ".join(
                f"{i*(_ws/(len(_spark)-1)):.1f},{_hs-(_v-_lo)/_rng*(_hs-3):.1f}"
                for i, _v in enumerate(_spark))
            _pf  = f"0,{_hs} " + _pts + f" {_ws},{_hs}"
            _ly  = f"{_hs-(_spark[-1]-_lo)/_rng*(_hs-3):.1f}"
            _mini = (
                f'<svg width="{_ws}" height="{_hs}" viewBox="0 0 {_ws} {_hs}" '
                f'preserveAspectRatio="none">'
                f'<defs><linearGradient id="{_fid}" x1="0" y1="0" x2="0" y2="1">'
                f'<stop offset="0%" stop-color="{_sc}" stop-opacity=".18"/>'
                f'<stop offset="100%" stop-color="{_sc}" stop-opacity="0"/>'
                f'</linearGradient></defs>'
                f'<polygon points="{_pf}" fill="url(#{_fid})"/>'
                f'<polyline points="{_pts}" fill="none" stroke="{_sc}" '
                f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
                f'<circle cx="{_ws}" cy="{_ly}" r="2.5" fill="{_sc}"/>'
                f'</svg>'
            )

        _pw = min(100, _peso * 100 * 4)
        _rows_pos += f"""
<tr onmouseover="this.style.background='rgba(255,255,255,0.025)'"
    onmouseout="this.style.background='transparent'"
    style="border-bottom:1px solid rgba(255,255,255,0.04);transition:background .15s;">
  <td style="padding:12px 20px 12px 20px;white-space:nowrap;">
    <div style="font-family:DM Mono,monospace;font-weight:800;font-size:0.88rem;
                color:#0a84ff;background:rgba(10,132,255,.1);
                border:1px solid rgba(10,132,255,.2);border-radius:6px;
                padding:3px 10px;display:inline-block;">{_t}</div>
    <div style="font-size:0.67rem;color:#636366;font-family:DM Mono,monospace;
                margin-top:3px;">{_shares:.4f} × ${_avg:,.2f}</div>
  </td>
  <td style="padding:10px 16px;">
    {_mini if _mini else '<div style="width:80px;height:28px;background:rgba(255,255,255,0.025);border-radius:4px;"></div>'}
  </td>
  <td style="padding:10px 16px;font-family:DM Mono,monospace;font-weight:700;
             font-size:0.9rem;color:#ffffff;white-space:nowrap;">${_price:,.2f}</td>
  <td style="padding:10px 12px;white-space:nowrap;">
    <div style="background:rgba({_pnl_rgb},.1);color:{_pnl_clr};
                border:1px solid rgba({_pnl_rgb},.22);border-radius:20px;
                padding:3px 10px;font-family:DM Mono,monospace;
                font-size:0.82rem;font-weight:700;display:inline-block;">
      {_pnl_pct*100:+.2f}%</div>
    <div style="font-size:0.68rem;color:#636366;font-family:DM Mono,monospace;
                margin-top:2px;">${_pnl_usd:+,.2f}</div>
  </td>
  <td style="padding:10px 12px;font-family:DM Mono,monospace;font-size:0.86rem;
             font-weight:700;color:{_day_clr};white-space:nowrap;">
    {_day_arr} {abs(_day_pct*100):.2f}%</td>
  <td style="padding:10px 16px;font-family:DM Mono,monospace;font-size:0.86rem;
             color:#ffffff;font-weight:600;white-space:nowrap;">${_valor:,.2f}</td>
  <td style="padding:10px 20px 10px 16px;min-width:130px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:4px;
                  height:4px;overflow:hidden;min-width:60px;">
        <div style="width:{_pw:.0f}%;height:100%;background:#0a84ff;border-radius:4px;"></div>
      </div>
      <span style="font-family:DM Mono,monospace;font-size:0.76rem;color:#8e8e93;
                   white-space:nowrap;min-width:38px;text-align:right;">
        {_peso*100:.1f}%</span>
    </div>
  </td>
  <td style="padding:10px 20px 10px 12px;font-family:DM Mono,monospace;
             font-size:0.82rem;font-weight:600;color:{_drclr};white-space:nowrap;">
    {f'{_drift*100:+.1f}pp' if abs(_drift) > 0.001 else '—'}</td>
</tr>"""

    _th = ("padding:10px 16px;text-align:left;font-family:DM Mono,monospace;"
           "font-size:0.67rem;font-weight:700;color:#636366;text-transform:uppercase;"
           "letter-spacing:.9px;border-bottom:1px solid rgba(255,255,255,0.07);"
           "background:rgba(255,255,255,0.025);white-space:nowrap;")
    st.markdown(f"""
<div style="border-radius:20px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);
            margin-bottom:8px;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>
      <th style="{_th}padding-left:20px;">Emisora</th>
      <th style="{_th}">30D</th>
      <th style="{_th}">Precio</th>
      <th style="{_th}padding-left:12px;">P&amp;L</th>
      <th style="{_th}padding-left:12px;">Hoy</th>
      <th style="{_th}">Valor</th>
      <th style="{_th}">Peso</th>
      <th style="{_th}padding-right:20px;">Drift</th>
    </tr></thead>
    <tbody>{_rows_pos}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 4.5: EARNINGS CALENDAR + DIVIDENDOS
    # ═══════════════════════════════════════════════════════════
    section("PRÓXIMOS EVENTOS")
    _evts = fetch_events_data(tuple(sorted(tickers)))
    _today = datetime.today().date()
    _col_earn, _col_div = st.columns(2, gap="small")

    with _col_earn:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:#aeaeb2;"
            "letter-spacing:1px;text-transform:uppercase;font-family:DM Mono,monospace;"
            "margin-bottom:10px;'>📅 Earnings</div>",
            unsafe_allow_html=True,
        )
        _earn_rows = []
        for _t, _ev in _evts.items():
            _ne = _ev.get("next_earnings")
            if _ne is None:
                continue
            try:
                _nd = _ne.date() if hasattr(_ne, "date") else _ne
                _days = (_nd - _today).days
                _earn_rows.append((_t, _nd, _days))
            except Exception:
                pass
        _earn_rows.sort(key=lambda x: x[2])
        if _earn_rows:
            for _et, _ed, _dd in _earn_rows:
                if _dd < -7:
                    continue
                if _dd < 0:
                    _clr, _badge = "#636366", f"Hace {abs(_dd)}d"
                elif _dd == 0:
                    _clr, _badge = "#ff453a", "¡Hoy!"
                elif _dd <= 7:
                    _clr, _badge = "#ffd60a", f"En {_dd}d"
                else:
                    _clr, _badge = "#0a84ff", f"En {_dd}d"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;padding:8px 12px;"
                    f"background:rgba(255,255,255,0.03);border-radius:10px;"
                    f"border:1px solid rgba(255,255,255,0.06);margin-bottom:5px;'>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.82rem;"
                    f"color:#fff;font-weight:600;'>{_et}</span>"
                    f"<div style='text-align:right;'>"
                    f"<div style='font-family:DM Mono,monospace;font-size:0.78rem;"
                    f"color:{_clr};font-weight:700;'>{_badge}</div>"
                    f"<div style='font-size:0.62rem;color:#636366;'>"
                    f"{_ed.strftime('%d %b %Y')}</div></div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Sin earnings próximos disponibles.")

    with _col_div:
        st.markdown(
            "<div style='font-size:0.7rem;font-weight:700;color:#aeaeb2;"
            "letter-spacing:1px;text-transform:uppercase;font-family:DM Mono,monospace;"
            "margin-bottom:10px;'>💰 Dividendos</div>",
            unsafe_allow_html=True,
        )
        _div_rows = [(t, e) for t, e in _evts.items() if e.get("annual_div", 0) > 0]
        _total_annual_income = 0.0
        if _div_rows:
            for _dt, _de in _div_rows:
                _dy   = _de.get("div_yield", 0) * 100
                _ann  = _de.get("annual_div", 0)
                _ex   = _de.get("next_ex_date")
                _ex_s = _ex.strftime("%d %b") if _ex else "—"
                _sh   = float(hdf.loc[hdf["Ticker"] == _dt, "Shares"].sum()) if not hdf.empty else 0
                _inc  = _sh * _ann
                _total_annual_income += _inc
                st.markdown(
                    f"<div style='padding:8px 12px;background:rgba(255,255,255,0.03);"
                    f"border-radius:10px;border:1px solid rgba(255,255,255,0.06);"
                    f"margin-bottom:5px;'>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.82rem;"
                    f"color:#fff;font-weight:600;'>{_dt}</span>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.82rem;"
                    f"color:#30d158;font-weight:700;'>{_dy:.2f}%</span></div>"
                    f"<div style='display:flex;gap:14px;margin-top:3px;'>"
                    f"<span style='font-size:0.65rem;color:#636366;'>${_ann:.2f}/acción</span>"
                    f"<span style='font-size:0.65rem;color:#636366;'>Ex-div: {_ex_s}</span>"
                    + (f"<span style='font-size:0.65rem;color:#8e8e93;'>Ingreso: ${_inc:.2f}/año</span>" if _inc > 0 else "")
                    + f"</div></div>",
                    unsafe_allow_html=True,
                )
            if _total_annual_income > 0:
                st.markdown(
                    f"<div style='margin-top:6px;padding:6px 12px;"
                    f"background:rgba(48,209,88,0.06);border-radius:8px;"
                    f"border:1px solid rgba(48,209,88,0.15);font-family:DM Mono,monospace;"
                    f"font-size:0.75rem;color:#30d158;'>"
                    f"Total ingreso anual: <b>${_total_annual_income:.2f}</b></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Ninguna posición paga dividendos.")

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 5: ACTIVIDAD RECIENTE (HTML auto-contenido)
    # ═══════════════════════════════════════════════════════════
    txns = transactions_df()
    if not txns.empty:
        recent = txns.sort_values("Date", ascending=False).head(6)
        rows_html = ""
        for _, r in recent.iterrows():
            is_buy  = str(r["Type"]).upper() == "BUY"
            clr     = "#30d158" if is_buy else "#ff453a"
            bg_rgb  = "48,209,88" if is_buy else "255,69,58"
            icon    = "↑" if is_buy else "↓"
            monto   = float(r["Shares"]) * float(r["Price"])
            tipo    = "Compra" if is_buy else "Venta"
            notes_s = f" · {r['Notes']}" if str(r.get("Notes","")).strip() else ""
            sign    = "+" if is_buy else "-"
            rows_html += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:30px;height:30px;border-radius:50%;flex-shrink:0;
                background:rgba({bg_rgb},0.12);display:flex;align-items:center;
                justify-content:center;font-size:1rem;color:{clr};font-weight:700;">{icon}</div>
    <div>
      <span style="font-family:DM Mono,monospace;font-weight:700;color:#ffffff;
                   font-size:0.88rem;">{r["Ticker"]}</span>
      <span style="font-size:0.76rem;color:#8e8e93;margin-left:8px;">
        {tipo} · {float(r["Shares"]):.4f} @ ${float(r["Price"]):.2f}{notes_s}
      </span>
    </div>
  </div>
  <div style="text-align:right;flex-shrink:0;">
    <div style="font-family:DM Mono,monospace;font-weight:700;color:{clr};
                font-size:0.88rem;">{sign}${monto:,.2f}</div>
    <div style="font-size:0.7rem;color:#48484a;">{r["Date"]}</div>
  </div>
</div>"""

        st.markdown(f"""
<div style="margin-top:16px;">
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:1.2px;color:#0a84ff;
              text-transform:uppercase;font-family:DM Mono,monospace;margin-bottom:12px;
              position:relative;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07);">
    ACTIVIDAD RECIENTE
    <div style="position:absolute;bottom:-1px;left:0;width:36px;height:2px;
                background:linear-gradient(90deg,#0a84ff,rgba(10,132,255,0));border-radius:2px;"></div>
  </div>
  <div style="background:rgba(10,14,26,0.6);border:1px solid rgba(255,255,255,0.07);
              border-radius:18px;padding:4px 20px 4px;
              box-shadow:0 4px 24px rgba(0,0,0,0.2);">
    {rows_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CSV BACKUP / RESTORE
# ─────────────────────────────────────────────────────────────

CSV_COLUMNS = ["Ticker", "Type", "Shares", "Price", "Date", "Notes"]

def transactions_to_csv(txns: list[dict]) -> str:
    if not txns:
        return ",".join(CSV_COLUMNS) + "\n"
    df = pd.DataFrame(txns)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CSV_COLUMNS].to_csv(index=False)


def csv_rebalance_to_transactions(content: bytes, txn_date: str | None = None) -> tuple[list[dict], list[str]]:
    """
    Convierte el CSV del plan de rebalanceo (generado por _render_rebalance_result)
    en transacciones para el log.

    Columnas esperadas del CSV de rebalanceo:
      Ticker, Acción (BUY/SELL/HOLD), Precio, Acciones, Delta ($), ...

    Las filas con Acción=HOLD se ignoran automáticamente.
    """
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        return [], [f"No se pudo leer el CSV: {e}"]

    # Normalizar nombres de columna
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    # Detectar columnas con varios nombres posibles
    def _find_col(candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    ticker_col  = _find_col(["ticker", "emisora", "symbol", "activo"])
    action_col  = _find_col(["acción", "accion", "action", "tipo", "type", "operación", "operacion"])
    price_col   = _find_col(["precio", "price", "precio merc.", "precio merc"])
    shares_col  = _find_col(["acciones", "shares", "títulos", "titulos", "qty"])

    missing = []
    if not ticker_col:  missing.append("Ticker")
    if not action_col:  missing.append("Acción/Action")
    if not price_col:   missing.append("Precio/Price")
    if not shares_col:  missing.append("Acciones/Shares")

    if missing:
        return [], [f"Columnas no encontradas: {', '.join(missing)}. "
                    f"Columnas disponibles: {', '.join(df.columns.tolist())}"]

    use_date = txn_date or str(date.today())
    errors, rows = [], []

    for i, row in df.iterrows():
        line = i + 2

        ticker = str(row[ticker_col]).strip().upper()
        if not ticker or ticker == "NAN":
            continue

        action_raw = str(row[action_col]).strip().upper()
        if action_raw in ("HOLD", "—", "-", ""):
            continue  # ignorar silenciosamente

        if action_raw in ("BUY", "COMPRA", "COMPRAR", "B", "C", "1"):
            txn_type = "BUY"
        elif action_raw in ("SELL", "VENTA", "VENDER", "S", "V", "-1"):
            txn_type = "SELL"
        else:
            errors.append(f"Fila {line}: Acción '{action_raw}' no reconocida, omitida.")
            continue

        try:
            shares = abs(float(str(row[shares_col]).replace(",", "").replace("$", "")))
            if shares < 1e-6:
                errors.append(f"Fila {line}: {ticker} — acciones = 0, omitida.")
                continue
        except Exception:
            errors.append(f"Fila {line}: {ticker} — valor de acciones inválido, omitida.")
            continue

        try:
            price = abs(float(str(row[price_col]).replace(",", "").replace("$", "")))
        except Exception:
            errors.append(f"Fila {line}: {ticker} — precio inválido, se usará 0.")
            price = 0.0

        rows.append({
            "Ticker": ticker,
            "Type":   txn_type,
            "Shares": round(shares, 8),
            "Price":  round(price, 4),
            "Date":   use_date,
            "Notes":  "Importado desde plan de rebalanceo",
        })

    return rows, errors


def csv_to_transactions(content: bytes) -> tuple[list[dict], list[str]]:
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        return [], [f"No se pudo leer el CSV: {e}"]

    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "ticker":"Ticker","symbol":"Ticker","emisora":"Ticker",
        "type":"Type","tipo":"Type","accion":"Type","acción":"Type",
        "shares":"Shares","titulos":"Shares","títulos":"Shares","cantidad":"Shares","qty":"Shares",
        "price":"Price","precio":"Price","costo":"Price","avgcost":"Price","avg_cost":"Price",
        "date":"Date","fecha":"Date",
        "notes":"Notes","notas":"Notes","nota":"Notes",
    }
    df.rename(columns={c: rename_map.get(c.lower(), c) for c in df.columns}, inplace=True)

    errors, rows = [], []
    for i, row in df.iterrows():
        line = i + 2
        ticker = str(row.get("Ticker","")).strip().upper()
        if not ticker or ticker == "NAN":
            errors.append(f"Fila {line}: Ticker vacío, omitida.")
            continue
        raw_type = str(row.get("Type","BUY")).strip().upper()
        txn_type = "BUY" if raw_type in ("BUY","COMPRA","C","B","1") else                    "SELL" if raw_type in ("SELL","VENTA","V","S","-1") else "BUY"
        try:
            shares = float(row.get("Shares",0))
            if shares <= 0:
                errors.append(f"Fila {line}: Shares ≤ 0, omitida."); continue
        except Exception:
            errors.append(f"Fila {line}: Shares inválido, omitida."); continue
        try:
            price = float(str(row.get("Price",0)).replace("$","").replace(",",""))
        except Exception:
            price = 0.0
        try:
            parsed = pd.to_datetime(str(row.get("Date","")), errors="coerce")
            txn_date = parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else str(date.today())
        except Exception:
            txn_date = str(date.today())
        notes = str(row.get("Notes","")).strip()
        if notes.lower() == "nan": notes = ""
        rows.append({"Ticker":ticker,"Type":txn_type,"Shares":shares,
                     "Price":price,"Date":txn_date,"Notes":notes})
    return rows, errors


def _logo_url(ticker: str) -> str:
    """URL del logo para usar en st.column_config.ImageColumn."""
    return f"https://assets.parqet.com/logos/symbol/{ticker.upper()}"


def tab_editor() -> None:
    # ── Layout: panel izquierdo (archivos + nueva txn) | derecho (log + pesos) ──
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        # ── Gestión de archivos ───────────────────────────────
        section("GESTIÓN DE ARCHIVOS")
        saved = list_portfolios()
        sel   = st.selectbox("Portafolio:", ["── Nuevo ──"] + saved)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📂 Cargar", use_container_width=True):
                if sel != "── Nuevo ──":
                    data = load_portfolio(sel)
                    if data:
                        st.session_state["portfolio_name"] = data.get("name", sel)
                        st.session_state["benchmark"]      = data.get("benchmark", "SPY")
                        st.session_state["transactions"]    = data.get("transactions", [])
                        st.session_state["target_weights"]  = data.get("target_weights", {})
                        st.session_state["thesis"]           = data.get("thesis", {})
                        st.session_state["alerts"]           = data.get("alerts", [])
                        st.session_state["extra_benchmarks"] = data.get("extra_benchmarks", [])
                        st.session_state["perf_custom_start"]= data.get("perf_custom_start")
                        st.session_state["perf_use_custom"]  = data.get("perf_use_custom", False)
                        st.session_state["watchlist"]        = data.get("watchlist", [])
                        st.success(f"Cargado: {sel}")
                        st.rerun()
        with c2:
            if st.button("🗑️ Eliminar", use_container_width=True, type="secondary"):
                if sel != "── Nuevo ──":
                    delete_portfolio(sel)
                    st.toast(f"Eliminado: {sel}")
                    st.rerun()

        st.divider()
        pname       = st.text_input("Nombre:", value=st.session_state.get("portfolio_name",""), key="pname_in")
        bench_input = st.text_input("Benchmark:", value=st.session_state.get("benchmark","SPY"))
        if st.button("💾 Guardar", use_container_width=True, type="primary"):
            if pname.strip():
                st.session_state["portfolio_name"] = pname.strip()
                st.session_state["benchmark"]      = bench_input.strip().upper()
                save_portfolio(
                    name=pname.strip(),
                    transactions=st.session_state.get("transactions", []),
                    target_weights=st.session_state.get("target_weights", {}),
                    benchmark=bench_input.strip().upper(),
                )
                st.success("Guardado ✓")
            else:
                st.warning("Ingresa un nombre primero.")

        st.divider()

        # ── Backup / Restore CSV ──────────────────────────────
        section("BACKUP / RESTAURAR")

        # Export
        txns_now = st.session_state.get("transactions", [])
        if txns_now:
            csv_bytes = transactions_to_csv(txns_now).encode("utf-8")
            pname_dl  = st.session_state.get("portfolio_name", "portafolio") or "portafolio"
            st.download_button(
                label="⬇️  Exportar transacciones (.csv)",
                data=csv_bytes,
                file_name=f"{pname_dl}_txns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Guarda una copia de seguridad de todas tus transacciones.",
            )
        else:
            st.caption("Sin transacciones que exportar.")

        # Import
        uploaded = st.file_uploader(
            "📤 Importar CSV de transacciones:",
            type=["csv"],
            key="csv_uploader",
            help=(
                "Columnas esperadas: Ticker, Type (BUY/SELL), Shares, Price, Date, Notes.\n"
                "También acepta variantes en español: Emisora, Tipo, Títulos, Precio, Fecha."
            ),
        )
        if uploaded is not None:
            import_mode = st.radio(
                "Al importar:",
                ["Reemplazar todo el log", "Añadir al log existente"],
                horizontal=True,
                key="csv_import_mode",
            )
            if st.button("📥 Confirmar importación", use_container_width=True, type="primary",
                         key="csv_confirm"):
                new_txns, errs = csv_to_transactions(uploaded.read())

                if errs:
                    for e in errs[:5]:
                        st.warning(e)
                    if len(errs) > 5:
                        st.warning(f"… y {len(errs)-5} advertencias más.")

                if new_txns:
                    if import_mode == "Reemplazar todo el log":
                        st.session_state["transactions"] = new_txns
                        st.success(f"✅ {len(new_txns)} transacciones importadas (log reemplazado).")
                    else:
                        existing = st.session_state.get("transactions", [])
                        st.session_state["transactions"] = existing + new_txns
                        st.success(
                            f"✅ {len(new_txns)} transacciones añadidas "
                            f"(total: {len(existing)+len(new_txns)})."
                        )
                    st.rerun()
                else:
                    st.error("No se importó ninguna fila válida. Revisa el formato del CSV.")

        # ── Importar desde plan de rebalanceo ────────────────
        st.divider()
        section("IMPORTAR PLAN DE REBALANCEO")
        st.caption(
            "Sube el CSV que descargaste desde la pestaña **Rebalanceo** "
            "para registrar todas las operaciones en el log automáticamente."
        )

        reb_file = st.file_uploader(
            "📋 CSV del plan de rebalanceo:",
            type=["csv"],
            key="reb_csv_uploader",
            help="El archivo que descargaste con el botón '⬇️ Descargar plan (.csv)'",
        )

        if reb_file is not None:
            # Preview rápida del archivo
            preview_bytes = reb_file.read()
            reb_file.seek(0) if hasattr(reb_file, "seek") else None

            try:
                preview_df = pd.read_csv(pd.io.common.BytesIO(preview_bytes))
                st.caption(f"Archivo leído: {len(preview_df)} filas, columnas: {', '.join(preview_df.columns.tolist())}")
            except Exception:
                st.warning("No se pudo previsualizar el archivo.")

            reb_date = st.date_input(
                "Fecha de ejecución de las operaciones:",
                value=date.today(),
                key="reb_import_date",
                help="Fecha en que ejecutaste las órdenes en tu broker.",
            )

            if st.button("📥 Importar al log de transacciones", type="primary",
                         use_container_width=True, key="reb_import_btn"):
                new_txns, errs = csv_rebalance_to_transactions(
                    preview_bytes, txn_date=str(reb_date)
                )

                if errs:
                    for e_msg in errs[:5]:
                        st.warning(e_msg)
                    if len(errs) > 5:
                        st.warning(f"… y {len(errs)-5} advertencias más.")

                if new_txns:
                    existing = st.session_state.get("transactions", [])
                    st.session_state["transactions"] = existing + new_txns
                    st.success(
                        f"✅ {len(new_txns)} operación(es) del rebalanceo añadidas al log. "
                        f"Total de transacciones: {len(existing) + len(new_txns)}."
                    )
                    # Mostrar preview de lo que se importó
                    preview_imported = pd.DataFrame(new_txns)
                    st.dataframe(
                        preview_imported,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Type":   st.column_config.TextColumn("Tipo",   width="small"),
                            "Shares": st.column_config.NumberColumn("Acciones", format="%.5f"),
                            "Price":  st.column_config.NumberColumn("Precio",   format="$%.2f"),
                            "Date":   st.column_config.TextColumn("Fecha"),
                            "Notes":  st.column_config.TextColumn("Notas"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.rerun()
                else:
                    st.error(
                        "No se pudo importar ninguna operación. "
                        "Verifica que el archivo sea el CSV del plan de rebalanceo."
                    )

        st.divider()

        # ── Formulario nueva transacción ──────────────────────
        section("REGISTRAR TRANSACCIÓN")
        txn_type   = st.radio("Tipo:", ["BUY 🟢", "SELL 🔴"], horizontal=True)
        txn_ticker = st.selectbox("Ticker:", UNIVERSE, key="txn_ticker")

        # Detectar posición actual para botones rápidos de venta
        is_sell    = txn_type.startswith("SELL")
        pos_now    = holdings_df()
        pos_row    = pos_now[pos_now["Ticker"] == txn_ticker] if not pos_now.empty else pd.DataFrame()
        owned_now  = float(pos_row["Shares"].iloc[0]) if not pos_row.empty else 0.0
        avg_cost_now = float(pos_row["AvgCost"].iloc[0]) if not pos_row.empty else 0.0

        # Botones de porcentaje rápido solo al vender y si hay posición
        if is_sell and owned_now > 0:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#8e8e93;font-family:DM Mono,monospace;"
                f"margin-bottom:4px;'>Tienes <b style='color:#ffffff'>{owned_now:.6f}</b> títulos — venta rápida:</div>",
                unsafe_allow_html=True)
            qb1, qb2, qb3, qb4 = st.columns(4)
            pcts = [(qb1,"25%",0.25),(qb2,"50%",0.50),(qb3,"75%",0.75),(qb4,"100% 🔴",1.0)]
            for col, label, pct in pcts:
                with col:
                    if st.button(label, key=f"qsell_{pct}", use_container_width=True):
                        st.session_state["_txn_sh_val"] = round(owned_now * pct, 8)
                        st.rerun()

        # Inicializar valor del input desde botón rápido o default
        _default_shares = st.session_state.pop("_txn_sh_val", 0.0)

        tc1, tc2 = st.columns(2)
        with tc1:
            txn_shares = st.number_input(
                "Títulos:", min_value=0.0, step=0.001, format="%.6f",
                value=float(_default_shares), key="txn_sh",
            )
        with tc2:
            txn_price = st.number_input(
                "Precio ($):", min_value=0.0, step=0.01, format="%.4f", key="txn_px",
            )
        txn_date  = st.date_input("Fecha:", value=date.today(), key="txn_dt")
        txn_notes = st.text_input("Notas (opcional):", key="txn_notes")

        # Advertencia si el precio está vacío y es venta (sugerir precio de mercado)
        if is_sell and owned_now > 0 and txn_shares > 0 and txn_price == 0:
            live_px = fetch_live_prices([txn_ticker]).get(txn_ticker, {}).get("price", 0)
            if live_px > 0:
                st.info(f"💡 Precio de mercado actual de {txn_ticker}: **${live_px:,.2f}**")

        # Preview del impacto ANTES de confirmar
        if txn_shares > 0 and txn_price > 0:
            monto = txn_shares * txn_price
            color_prev = "#30d158" if not is_sell else "#ff453a"
            tipo_label = "Compra" if not is_sell else "Venta"
            pct_pos = f" ({txn_shares/owned_now:.0%} de tu posición)" if is_sell and owned_now > 0 else ""
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border-radius:10px;
            padding:12px 14px;margin-top:8px;font-family:DM Mono,monospace;
            font-size:0.85rem;border:1px solid rgba(255,255,255,0.07);">
  <span style="color:{color_prev};font-weight:700;">{tipo_label}</span>
  &nbsp;·&nbsp; {txn_shares:.6f} títulos de
  <b style="color:#ffffff">{txn_ticker}</b>{pct_pos}
  &nbsp;·&nbsp; @ ${txn_price:.4f}
  <br><span style="color:#aeaeb2;">Monto total: </span>
  <b style="color:{color_prev};">${monto:,.2f}</b>
</div>""", unsafe_allow_html=True)

        if st.button("✅ Confirmar Transacción", use_container_width=True, type="primary"):
            if txn_ticker and txn_shares > 0 and txn_price > 0:
                tipo = "BUY" if not is_sell else "SELL"
                if tipo == "SELL":
                    # FIX: usar tolerancia para permitir vender exactamente lo que se tiene
                    if txn_shares > owned_now + 1e-4:
                        st.error(
                            f"No puedes vender {txn_shares:.6f} de {txn_ticker}. "
                            f"Solo tienes {owned_now:.6f} títulos. "
                            f"Usa el botón **100%** para liquidar la posición completa."
                        )
                        st.stop()
                    # Si se vende "todo" (dentro de tolerancia), usar exactamente owned_now
                    if abs(txn_shares - owned_now) < 1e-4:
                        txn_shares = owned_now

                txns = st.session_state.get("transactions", [])
                txns.append({
                    "Ticker": txn_ticker.upper(),
                    "Type":   tipo,
                    "Shares": txn_shares,
                    "Price":  txn_price,
                    "Date":   str(txn_date),
                    "Notes":  txn_notes.strip(),
                })
                st.session_state["transactions"] = txns
                verb = "Comprada" if tipo == "BUY" else "Vendida"
                st.success(f"{verb}: {txn_shares:.6f} × {txn_ticker} @ ${txn_price:.4f}")
                st.rerun()
            else:
                st.warning("Completa todos los campos (títulos y precio > 0).")

    # ── Panel derecho ─────────────────────────────────────────
    with col_right:
        sub_tab_txn, sub_tab_pos, sub_tab_tw = st.tabs(
            ["📋 Log de Transacciones", "📊 Posiciones Actuales", "🎯 Pesos Objetivo"]
        )

        # ── Sub-tab: Log de transacciones ─────────────────────
        with sub_tab_txn:
            txns_df = transactions_df()
            if txns_df.empty:
                st.info("Sin transacciones. Usa el formulario izquierdo para agregar.")
            else:
                st.caption(f"{len(txns_df)} transacciones registradas")

                # Tabla editable del log
                txns_edit = txns_df.copy()
                txns_edit["Date"] = pd.to_datetime(txns_edit["Date"], errors="coerce")

                edited_txns = st.data_editor(
                    txns_edit,
                    column_config={
                        "Ticker": st.column_config.SelectboxColumn("Ticker",  options=UNIVERSE, width="small"),
                        "Type":   st.column_config.SelectboxColumn("Tipo",    options=["BUY","SELL"], width="small"),
                        "Shares": st.column_config.NumberColumn("Títulos",   format="%.6f", min_value=0),
                        "Price":  st.column_config.NumberColumn("Precio ($)", format="%.4f", min_value=0),
                        "Date":   st.column_config.DateColumn("Fecha"),
                        "Notes":  st.column_config.TextColumn("Notas"),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="txns_editor",
                )

                col_apply, col_clear = st.columns(2)
                with col_apply:
                    if st.button("✅ Aplicar cambios al log", use_container_width=True):
                        rows = edited_txns.dropna(subset=["Ticker"]).copy()
                        rows = rows[rows["Ticker"].astype(str).str.strip() != ""]
                        rows["Date"] = rows["Date"].apply(
                            lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else str(date.today())
                        )
                        st.session_state["transactions"] = rows.to_dict("records")
                        st.success("Log actualizado.")
                        st.rerun()
                with col_clear:
                    if st.button("🗑️ Borrar todo el log", use_container_width=True, type="secondary"):
                        st.session_state["transactions"] = []
                        st.rerun()

        # ── Sub-tab: Posiciones derivadas ─────────────────────
        with sub_tab_pos:
            hdf = holdings_df()
            if hdf.empty:
                st.info("Sin posiciones. Agrega transacciones primero.")
            else:
                st.caption("Calculado automáticamente desde el log (costo promedio ponderado)")
                prices_pos   = fetch_live_prices(hdf["Ticker"].tolist())
                _pnl_pos     = compute_realized_pnl(transactions_df())
                _real_map_pos = _pnl_pos["by_ticker"]
                nav_pos = 0.0
                pos_rows = []
                for _, r in hdf.iterrows():
                    t   = r["Ticker"]
                    sh  = float(r["Shares"])
                    ac  = float(r["AvgCost"])
                    px  = prices_pos.get(t, {}).get("price", 0.0)
                    vm  = sh * px
                    nav_pos += vm
                    real_pnl = _real_map_pos.get(t, {}).get("realized_pnl", 0.0)
                    unreal   = vm - sh * ac
                    pos_rows.append({
                        "Ticker":          t,
                        "Títulos":         sh,
                        "Cto. Prom. $":    ac,
                        "Invertido $":     sh * ac,
                        "Precio Actual":   px,
                        "Valor Merc. $":   vm,
                        "P&L No Real. $":  unreal,
                        "P&L Realizado $": real_pnl,
                        "P&L Total $":     unreal + real_pnl,
                        "P&L %":           (px - ac) / ac if ac > 0 else 0.0,
                    })

                df_pos = pd.DataFrame(pos_rows)
                df_pos["% Cartera"] = df_pos["Valor Merc. $"] / nav_pos if nav_pos > 0 else 0.0

                st.dataframe(
                    df_pos,
                    column_config={
                        "Ticker":          st.column_config.TextColumn("Ticker", width="small"),
                        "Títulos":         st.column_config.NumberColumn(format="%.5f"),
                        "Cto. Prom. $":    st.column_config.NumberColumn(format="$%.4f"),
                        "Invertido $":     st.column_config.NumberColumn(format="$%.2f"),
                        "Precio Actual":   st.column_config.NumberColumn(format="$%.2f"),
                        "Valor Merc. $":   st.column_config.NumberColumn(format="$%.2f"),
                        "P&L No Real. $":  st.column_config.NumberColumn(
                            "P&L Abierto",  format="$%+.2f",
                            help="Ganancia/pérdida latente de la posición actual"),
                        "P&L Realizado $": st.column_config.NumberColumn(
                            "P&L Realizado", format="$%+.2f",
                            help="Ganancia/pérdida ya cobrada por ventas parciales pasadas"),
                        "P&L Total $":     st.column_config.NumberColumn(
                            "P&L Total",    format="$%+.2f",
                            help="Abierto + Realizado"),
                        "P&L %":           st.column_config.NumberColumn(format="%+.2f%%"),
                        "% Cartera":       st.column_config.ProgressColumn(
                            format="%.1f%%", min_value=0, max_value=1),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                # ── Posiciones cerradas ────────────────────────────────
                closed_rows = []
                open_tickers = set(hdf["Ticker"].tolist())
                for t, info in _real_map_pos.items():
                    if t not in open_tickers and info["is_closed"] and info["shares_sold"] > 0:
                        closed_rows.append({
                            "Ticker":       t,
                            "Acciones vend.": info["shares_sold"],
                            "Ingresos $":   info["proceeds"],
                            "Costo base $": info["cost_sold"],
                            "P&L Real. $":  info["realized_pnl"],
                            "P&L Real. %":  info["realized_pnl"] / info["cost_sold"]
                                            if info["cost_sold"] > 0 else 0.0,
                        })

                if closed_rows:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
                        "color:#636366;text-transform:uppercase;font-family:DM Mono,monospace;"
                        "margin-bottom:6px;'>POSICIONES CERRADAS (P&L REALIZADO)</div>",
                        unsafe_allow_html=True)
                    df_closed = pd.DataFrame(closed_rows).sort_values(
                        "P&L Real. $", ascending=False)
                    st.dataframe(
                        df_closed,
                        column_config={
                            "Ticker":         st.column_config.TextColumn("Ticker", width="small"),
                            "Acciones vend.": st.column_config.NumberColumn(format="%.5f"),
                            "Ingresos $":     st.column_config.NumberColumn(format="$%.2f"),
                            "Costo base $":   st.column_config.NumberColumn(format="$%.2f"),
                            "P&L Real. $":    st.column_config.NumberColumn(
                                "P&L Realizado", format="$%+.2f"),
                            "P&L Real. %":    st.column_config.NumberColumn(
                                "Retorno", format="%+.2f%%"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                    total_closed_pnl = sum(r["P&L Real. $"] for r in closed_rows)
                    clr = "#30d158" if total_closed_pnl >= 0 else "#ff453a"
                    st.markdown(
                        f"<div style='font-family:DM Mono,monospace;font-size:0.82rem;"
                        f"color:#aeaeb2;margin-top:4px;'>Total P&L realizado posiciones cerradas: "
                        f"<b style='color:{clr};'>${total_closed_pnl:+,.2f}</b></div>",
                        unsafe_allow_html=True)

                total_real_pos = _pnl_pos["total_realized"]
                total_unreal_pos = sum(r["P&L No Real. $"] for r in pos_rows)
                st.caption(
                    f"**NAV total:** ${nav_pos:,.2f}  ·  "
                    f"P&L abierto: **${total_unreal_pos:+,.2f}**  ·  "
                    f"P&L realizado (total): **${total_real_pos:+,.2f}**  ·  "
                    f"P&L combinado: **${total_unreal_pos + total_real_pos:+,.2f}**"
                )

        # ── Sub-tab: Pesos objetivo ────────────────────────────
        with sub_tab_tw:
            section("TARGET ALLOCATION")
            hdf2 = holdings_df()
            tickers_held  = hdf2["Ticker"].unique().tolist() if not hdf2.empty else []
            extra_raw = st.text_input("Agregar tickers adicionales (coma):", placeholder="NVDA, GLD")
            extra_tickers = [t.strip().upper() for t in extra_raw.split(",") if t.strip()]
            all_tw_tickers = list(dict.fromkeys(tickers_held + extra_tickers))

            tw_current = st.session_state.get("target_weights", {})
            tw_updated: dict[str, float] = {}

            if all_tw_tickers:
                st.caption("Define el % objetivo de cada activo. Deben sumar exactamente 100%.")
                cols = st.columns(min(4, len(all_tw_tickers)))
                for i, ticker in enumerate(all_tw_tickers):
                    with cols[i % len(cols)]:
                        default_v = tw_current.get(ticker, 0.0) * 100
                        val = st.number_input(
                            f"{ticker}", min_value=0.0, max_value=100.0,
                            value=float(f"{default_v:.2f}"),
                            step=0.5, key=f"tw_{ticker}",
                        )
                        tw_updated[ticker] = val / 100

                total_w = sum(tw_updated.values())
                ok = abs(total_w - 1.0) < 0.005
                color = "#30d158" if ok else "#ff453a"
                st.markdown(
                    f"<p style='font-family:var(--mono); font-size:0.9rem; margin-top:10px;'>"
                    f"Total: <strong style='color:{color};'>{total_w:.1%}</strong> "
                    f"{'✓' if ok else '← debe ser 100%'}</p>",
                    unsafe_allow_html=True,
                )
                if st.button("💾 Guardar pesos objetivo", use_container_width=True):
                    st.session_state["target_weights"] = {k: v for k, v in tw_updated.items() if v > 0}
                    st.success("Pesos objetivo guardados.")
            else:
                st.info("Agrega transacciones primero para configurar los pesos objetivo.")


# ─────────────────────────────────────────────────────────────
# MOTOR DE OPTIMIZACIÓN (MPT) — ESTRATEGIAS
# ─────────────────────────────────────────────────────────────

from scipy.optimize import minimize as _sp_minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STRATEGIES = ["SmartCluster", "MinVar", "MaxReturn", "ERC"]

def _portfolio_stats(w, mu, cov, rf):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sharpe

def _get_cluster_constraints(tickers: list[str], mu: np.ndarray, vol: np.ndarray,
                              returns_df: pd.DataFrame,
                              n_clusters: int = 4, max_cluster_w: float = 0.40) -> tuple:
    """
    Agrupa activos usando clustering jerárquico (complete linkage) sobre la
    MATRIZ DE CORRELACIONES, con rebalanceo forzado de clusters para evitar
    que un solo cluster absorba la mayoría de activos.

    Garantías:
      · Ningún cluster tiene más de ceil(n / n_clusters * 1.5) activos
      · Si un cluster queda vacío, se redistribuyen los excedentes
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance  import squareform

    n = len(tickers)
    # Ajustar n_clusters: máximo n//2, mínimo 2
    n_clusters = min(n_clusters, max(2, n // 2))

    # ── 1. Distancia de correlación ────────────────────────────
    corr = returns_df[tickers].corr().fillna(0).values
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist_matrix = np.sqrt(np.clip((1 - corr) / 2, 0.0, 1.0))
    np.fill_diagonal(dist_matrix, 0.0)

    condensed = squareform(dist_matrix, checks=False)

    # ── 2. Complete linkage: más balanceado que Ward para stocks ─
    Z         = linkage(condensed, method="complete")
    labels_raw = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed
    labels    = labels_raw.astype(int)

    # ── 3. Rebalancear: evitar clusters con > max_per_cluster ──
    # Máximo permitido por cluster = ceil(n / n_clusters) + 1
    import math
    max_per = math.ceil(n / n_clusters) + 1
    assignment = list(labels)

    for k in range(n_clusters):
        members = [i for i, lbl in enumerate(assignment) if lbl == k]
        while len(members) > max_per:
            # Mover el miembro más alejado del centroide del cluster
            # (el que tiene mayor distancia promedio con los otros del cluster)
            avg_dists = []
            for i in members:
                others = [j for j in members if j != i]
                avg_d = np.mean([dist_matrix[i, j] for j in others]) if others else 0
                avg_dists.append((avg_d, i))
            avg_dists.sort(reverse=True)  # más alejado primero
            orphan = avg_dists[0][1]

            # Asignar al cluster más cercano que tenga espacio
            best_k, best_d = None, np.inf
            for kk in range(n_clusters):
                if kk == k: continue
                kk_members = [i for i, lbl in enumerate(assignment) if lbl == kk]
                if len(kk_members) >= max_per: continue
                # Distancia promedio al cluster candidato
                if kk_members:
                    d = np.mean([dist_matrix[orphan, j] for j in kk_members])
                else:
                    d = 0.0
                if d < best_d:
                    best_d, best_k = d, kk

            if best_k is not None:
                assignment[orphan] = best_k
                members = [i for i, lbl in enumerate(assignment) if lbl == k]
            else:
                break  # no hay espacio en ningún otro cluster

    labels = np.array(assignment, dtype=int)

    # ── 4. Construir restricciones ─────────────────────────────
    constraints = []
    for k in range(n_clusters):
        idx_k = [i for i, lbl in enumerate(labels) if lbl == k]
        if not idx_k:
            continue
        def cluster_ineq(w, idxs=idx_k, limit=max_cluster_w):
            return limit - np.sum(w[idxs])
        constraints.append({"type": "ineq", "fun": cluster_ineq})

    return constraints, labels


def _run_optimization(mu: np.ndarray, cov: np.ndarray, returns_df: pd.DataFrame,
                      rf: float, strategy: str, max_w: float,
                      extra_constraints: list[dict] | None = None,
                      bounds: list[tuple] | None = None) -> np.ndarray:
    n   = len(mu)
    w0  = np.repeat(1/n, n)
    if bounds is None:
        bounds = [(0.0, max_w)] * n
    eq_sum  = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    constrs = [eq_sum] + (extra_constraints or [])

    if strategy in ("MaxSharpe", "SmartCluster"):
        # SmartCluster es MaxSharpe con cluster constraints (añadidos externamente)
        def obj(w):
            r, v, _ = _portfolio_stats(w, mu, cov, rf)
            return -(r - rf) / v if v > 1e-9 else 1e9
    elif strategy == "MinVar":
        def obj(w): return float(w @ cov @ w)
    elif strategy == "MaxReturn":
        def obj(w): return -float(np.dot(w, mu))
    elif strategy == "ERC":
        def obj(w):
            pv = float(w @ cov @ w)
            rc = w * (cov @ w)
            return np.sum((rc - pv / n) ** 2)
    elif strategy == "MinCVaR":
        def obj(w):
            pr = returns_df.values @ w
            var = np.percentile(pr, 5)
            tail = pr[pr <= var]
            return -tail.mean() if len(tail) > 0 else 1e9
    else:
        return w0

    # Warm-start: shift w0 so every asset respects its lower bound
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    # Ensure w0 is feasible: start from equal weight clamped to bounds
    w0 = np.clip(w0, lo_arr, hi_arr)
    w0 /= w0.sum()

    res = _sp_minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constrs,
                       options={"maxiter": 1000, "ftol": 1e-10})
    w = res.x if res.success else w0

    # Clip using PER-ASSET bounds (NOT hardcoded 0!)
    w = np.clip(w, lo_arr, hi_arr)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def optimize_to_target_weights(tickers: list[str], returns_df: pd.DataFrame,
                                rf: float, strategy: str, max_w: float,
                                n_clusters: int = 4,
                                max_cluster_w: float = 0.40,
                                locked_tickers: list[str] | None = None,
                                min_assets: int = 1,
                                max_assets: int | None = None) -> dict[str, float]:
    """
    Corre MPT sobre los tickers dados. Retorna {ticker: weight}.

    locked_tickers : activos que NO se pueden reducir a 0
                     (se les impone peso mínimo = current_weight o 1/2n).
    min_assets     : número mínimo de activos con peso > 0 en el resultado.
    max_assets     : número máximo de activos con peso > 0 (two-pass si se activa).
                     None = sin límite.
    SmartCluster   : MaxSharpe con restricciones de cluster por correlación.
    """
    locked_tickers = [t for t in (locked_tickers or []) if t in tickers]
    rets = returns_df[tickers].dropna(how="all").fillna(0)
    if len(rets) < 30 or len(tickers) < 2:
        eq = 1.0 / len(tickers)
        return {t: eq for t in tickers}

    def _single_pass(tkrs: list[str], rets_df: pd.DataFrame,
                     locked: list[str]) -> tuple[dict, np.ndarray | None]:
        """Ejecuta una pasada de optimización y devuelve (result_dict, cluster_labels)."""
        r   = rets_df[tkrs].dropna(how="all").fillna(0)
        mu_ = _robust_mu(r, tkrs)
        vol_= np.sqrt(np.diag(r.cov().values * TRADING_DAYS))
        cov_= r.cov().values * TRADING_DAYS
        n_  = len(tkrs)

        lf   = max(1 / (2 * max(n_, 1)), 0.01)
        mf   = max(lf, 0.005)
        e_min = min(max(min_assets, 0), n_)

        bds = []
        for i_, t_ in enumerate(tkrs):
            if t_ in locked:
                lo_ = lf
            elif i_ < e_min:
                lo_ = mf
            else:
                lo_ = 0.0
            bds.append((lo_, max_w))

        tlo = sum(b[0] for b in bds)
        if tlo > 0.95:
            sc_ = 0.90 / tlo
            bds = [(lo * sc_, hi) for lo, hi in bds]

        extra_ = []
        clabels_ = None
        if strategy == "SmartCluster":
            cc_, clabels_ = _get_cluster_constraints(
                tkrs, mu_, vol_, r, n_clusters=n_clusters, max_cluster_w=max_cluster_w)
            extra_.extend(cc_)

        w_ = _run_optimization(mu_, cov_, r, rf, strategy, max_w,
                               bounds=bds, extra_constraints=extra_)
        res_ = {t_: float(w_[i_]) for i_, t_ in enumerate(tkrs)}

        # Completar con 0 los tickers del universo completo que no están en esta pasada
        for t_all in tickers:
            if t_all not in res_:
                res_[t_all] = 0.0

        return res_, clabels_, mu_, vol_

    # ── PASADA 1: universo completo ───────────────────────────
    result, cluster_labels, mu, vol = _single_pass(tickers, rets, locked_tickers)

    # ── PASADA 2 (si max_assets está activo) ─────────────────
    if max_assets is not None:
        locked_set  = set(locked_tickers)
        # Contar activos con peso significativo
        non_zero    = [t for t in tickers if result.get(t, 0) > 0.005]

        if len(non_zero) > max_assets:
            # Siempre conservar locked tickers
            kept = [t for t in locked_set if t in tickers]
            remaining_slots = max(0, max_assets - len(kept))

            # Rellenar con los de mayor peso (no locked)
            non_locked_sorted = sorted(
                [(t, result[t]) for t in tickers if t not in locked_set],
                key=lambda x: -x[1],
            )
            for t, _ in non_locked_sorted:
                if remaining_slots <= 0:
                    break
                kept.append(t)
                remaining_slots -= 1

            kept = list(dict.fromkeys(kept))   # dedup preservando orden

            if len(kept) >= 2:
                # Re-optimizar solo con los activos seleccionados
                result, cluster_labels, mu, vol = _single_pass(kept, rets, locked_tickers)

    # ── Guardar info de clusters para visualización ───────────
    if cluster_labels is not None:
        _tkrs_pass = [t for t in tickers if result.get(t, 0) > 0.001]
        mu_full    = _robust_mu(rets[tickers].dropna(how="all").fillna(0), tickers)
        vol_full   = np.sqrt(np.diag(
            rets[tickers].dropna(how="all").fillna(0).cov().values * TRADING_DAYS))
        st.session_state["smart_cluster_labels"] = {
            t: int(cluster_labels[i]) for i, t in enumerate(_tkrs_pass)
            if i < len(cluster_labels)
        }
        st.session_state["smart_cluster_mu"]  = {t: float(mu_full[i])  for i, t in enumerate(tickers)}
        st.session_state["smart_cluster_vol"] = {t: float(vol_full[i]) for i, t in enumerate(tickers)}

    return result

def _robust_mu(returns_df: pd.DataFrame, tickers: list[str],
               shrink: float = 0.3, cap: float = 0.60) -> np.ndarray:
    """
    Estimador de retorno esperado robusto para optimización MPT.

    Problemas del estimador naive (mean * 252):
      - Activos con historial corto (SNDK ~1 año) tienen medias infladas
      - Sin shrinkage los extremos dominan la optimización
      - Acciones con run-ups recientes extrapolados dan >90% "esperado"

    Esta función aplica:
      1. Media aritmética anualizada (base)
      2. Shrinkage de James-Stein hacia la media del grupo
         mu_shrunk = (1-k)*mu_individual + k*mu_mean_group
         k=0.3 por defecto: modera sin destruir la señal
      3. Cap simétrico: mu se limita a [-cap, +cap] (default ±60%)
         Para referencia: S&P 500 ha promediado ~10% anual históricamente.
         60% permite capturar growth stocks sin permitir extrapolaciones absurdas.
    """
    rets  = returns_df[tickers].dropna(how="all").fillna(0)
    mu    = rets.mean().values * TRADING_DAYS           # aritmética base

    # James-Stein shrinkage toward cross-sectional mean
    mu_bar = mu.mean()                                  # media del grupo
    mu_sh  = (1 - shrink) * mu + shrink * mu_bar       # shrinkage

    # Cap simétrico
    mu_capped = np.clip(mu_sh, -cap, cap)

    return mu_capped


def get_frontier_points(tickers: list[str], returns_df: pd.DataFrame,
                        rf: float, max_w: float, n_pts: int = 400) -> pd.DataFrame:
    """Genera puntos de la frontera eficiente para graficar."""
    rets = returns_df[tickers].dropna(how="all").fillna(0)
    if len(rets) < 30: return pd.DataFrame()
    mu  = _robust_mu(rets, list(rets.columns))
    cov = rets.cov().values * TRADING_DAYS
    rows = []
    for _ in range(n_pts):
        w = np.random.dirichlet(np.ones(len(tickers)))
        w = np.clip(w, 0, max_w); w /= w.sum()
        r, v, s = _portfolio_stats(w, mu, cov, rf)
        rows.append({"Retorno": r, "Volatilidad": v, "Sharpe": s})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# ANÁLISIS TÉCNICO + FUNDAMENTAL POR ACTIVO
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def score_asset(ticker: str, price_series: pd.Series) -> dict:
    """
    Calcula score técnico (0-100) y obtiene datos fundamentales básicos.
    """
    result = {
        "Ticker": ticker, "Tech Score": 50, "Señal": "—",
        "RSI": None, "Sobre SMA50": None, "Mom 3M": None,
        "PE": None, "ROE": None, "Margen": None, "Crec. Rev.": None,
        "Fund. Score": 50,
    }

    # ── Técnico ────────────────────────────────────────────
    s = price_series.dropna()
    if len(s) >= 20:
        # RSI-14
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]
        result["RSI"] = round(float(rsi), 1) if not np.isnan(rsi) else None

        # SMA 50
        if len(s) >= 50:
            sma50 = s.rolling(50).mean().iloc[-1]
            result["Sobre SMA50"] = bool(s.iloc[-1] > sma50)

        # Momentum 3M (~63 días)
        if len(s) >= 63:
            mom = (s.iloc[-1] / s.iloc[-63]) - 1
            result["Mom 3M"] = round(float(mom), 4)

        # Score técnico
        score = 50
        if result["RSI"] is not None:
            if   result["RSI"] < 30: score += 20   # sobreventa = oportunidad
            elif result["RSI"] > 70: score -= 15   # sobrecompra
            elif result["RSI"] > 55: score += 8
            else:                    score -= 5
        if result["Sobre SMA50"] is True:  score += 15
        elif result["Sobre SMA50"] is False: score -= 10
        if result["Mom 3M"] is not None:
            if result["Mom 3M"] >  0.10: score += 12
            elif result["Mom 3M"] > 0:   score +=  5
            elif result["Mom 3M"] < -0.10: score -= 12
            else:                          score -=  5

        score = max(0, min(100, score))
        result["Tech Score"] = score
        if   score >= 75: result["Señal"] = "🚀 Fuerte Compra"
        elif score >= 60: result["Señal"] = "📈 Compra"
        elif score <= 25: result["Señal"] = "🔴 Venta Fuerte"
        elif score <= 40: result["Señal"] = "📉 Venta"
        else:             result["Señal"] = "😐 Neutral"

    # ── Fundamental (yfinance info) ────────────────────────
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        roe  = info.get("returnOnEquity")
        mgn  = info.get("profitMargins")
        rev  = info.get("revenueGrowth")

        result["PE"]         = round(float(pe),  2) if pe  else None
        result["ROE"]        = round(float(roe)*100, 1) if roe else None
        result["Margen"]     = round(float(mgn)*100, 1) if mgn else None
        result["Crec. Rev."] = round(float(rev)*100, 1) if rev else None

        # Score fundamental
        fs = 50
        if pe:
            if   0 < pe < 15: fs += 15
            elif 15 <= pe < 25: fs += 5
            elif pe > 40: fs -= 10
        if roe:
            if   roe > 0.20: fs += 15
            elif roe > 0.10: fs += 5
            elif roe < 0:    fs -= 10
        if mgn:
            if   mgn > 0.20: fs += 10
            elif mgn < 0.05: fs -= 5
        if rev:
            if   rev > 0.15: fs += 10
            elif rev < 0:    fs -= 8

        result["Fund. Score"] = max(0, min(100, fs))
    except Exception:
        pass

    result["Score Total"] = round(result["Tech Score"] * 0.55 + result["Fund. Score"] * 0.45)
    return result


@st.cache_data(ttl=600, show_spinner=False)
def analyze_candidates(candidates: list[str], history_start: str, history_end: str) -> pd.DataFrame:
    """Descarga historial y calcula scores para una lista de candidatos."""
    if not candidates:
        return pd.DataFrame()
    prices = fetch_history(candidates, history_start, history_end)
    rows = []
    for t in candidates:
        s = prices[t] if t in prices.columns else pd.Series(dtype=float)
        row = score_asset(t, s)
        # Rendimiento en el período
        if not s.dropna().empty and len(s.dropna()) > 1:
            row["Ret. Período"] = round((s.dropna().iloc[-1] / s.dropna().iloc[0] - 1) * 100, 2)
        else:
            row["Ret. Período"] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


# ─────────────────────────────────────────────────────────────
# TAB 3: REBALANCEO
# ─────────────────────────────────────────────────────────────

def _build_rebalance_explanation(
    valid_tickers: list,
    opt_weights: dict,
    current_weights: dict,
    mu_arr: np.ndarray,
    cov_arr: np.ndarray,
    rf: float,
    strategy: str,
    locked: list,
    min_assets: int,
) -> str:
    """
    Genera una explicación estructurada basada en reglas de por qué
    el optimizador asignó esos pesos. Sin IA, siempre disponible.
    """
    n = len(valid_tickers)
    vol_arr = np.sqrt(np.diag(cov_arr))
    corr    = cov_arr / np.outer(vol_arr, vol_arr)
    corr    = np.clip(corr, -1, 1)

    lines = []
    lines.append(f"**Estrategia: {strategy}**\n")

    # Portafolio resultante
    w_arr   = np.array([opt_weights.get(t, 0) for t in valid_tickers])
    port_r  = float(np.dot(w_arr, mu_arr))
    port_v  = float(np.sqrt(w_arr @ cov_arr @ w_arr))
    port_sh = (port_r - rf) / port_v if port_v > 0 else 0

    lines.append(
        f"El portafolio resultante tiene un **retorno esperado de {port_r:.1%}** "
        f"con una **volatilidad de {port_v:.1%}** y un **Sharpe de {port_sh:.2f}**.\n"
    )

    # Explicación por activo
    lines.append("### Por qué cada activo recibió ese peso:\n")

    for i, t in enumerate(valid_tickers):
        new_w = opt_weights.get(t, 0)
        old_w = current_weights.get(t, 0)
        delta = new_w - old_w
        mu_t  = mu_arr[i]
        vol_t = vol_arr[i]
        sh_t  = (mu_t - rf) / vol_t if vol_t > 0 else 0

        # Correlación promedio con los demás
        others = [j for j in range(n) if j != i and w_arr[j] > 0.001]
        avg_corr = float(np.mean([corr[i,j] for j in others])) if others else 0.0

        reasons = []

        if t in locked:
            reasons.append(f"protegido de venta (locked) con peso mínimo garantizado")

        if strategy == "MinVar":
            if vol_t < float(vol_arr.mean()):
                reasons.append(f"baja volatilidad propia ({vol_t:.1%} vs media {vol_arr.mean():.1%})")
            if avg_corr < 0.3:
                reasons.append(f"baja correlación con el resto ({avg_corr:.2f}) — mejora la diversificación")
            elif avg_corr > 0.6:
                reasons.append(f"alta correlación con el portafolio ({avg_corr:.2f}) — reduce su peso")

        elif strategy in ("SmartCluster", "MaxSharpe"):
            if sh_t > port_sh:
                reasons.append(f"Sharpe individual alto ({sh_t:.2f}) — supera el portafolio")
            if avg_corr < 0.35:
                reasons.append(f"baja correlación con el resto ({avg_corr:.2f}) — diversificador eficiente")
            if mu_t > float(mu_arr.mean()):
                reasons.append(f"retorno esperado por encima del promedio ({mu_t:.1%} vs {mu_arr.mean():.1%})")

        elif strategy == "ERC":
            target_risk = 1.0 / n
            if vol_t > float(vol_arr.mean()):
                reasons.append(f"volatilidad alta ({vol_t:.1%}) — ERC reduce su peso para igualar aportes al riesgo")
            else:
                reasons.append(f"volatilidad baja ({vol_t:.1%}) — ERC le da más peso para equilibrar")

        elif strategy == "MaxReturn":
            if mu_t > float(mu_arr.mean()):
                reasons.append(f"mayor retorno esperado del grupo ({mu_t:.1%})")
            else:
                reasons.append(f"retorno por debajo del promedio ({mu_t:.1%})")

        if new_w < 0.005 and t not in locked:
            reasons.append("peso residual — el optimizador no encontró beneficio marginal en esta posición")

        delta_str = f"{delta:+.1%}" if abs(delta) > 0.005 else "sin cambio"
        reason_str = "; ".join(reasons) if reasons else "distribución residual del optimizador"

        sign = "🟢" if delta > 0.01 else ("🔴" if delta < -0.01 else "⚪")
        lines.append(
            f"{sign} **{t}** — {new_w:.1%} (antes {old_w:.1%}, {delta_str})\n"
            f"  → {reason_str}\n"
        )

    # Advertencias
    lines.append("### Consideraciones importantes:\n")
    if min_assets > 1:
        lines.append(f"- Se garantizó un mínimo de **{min_assets} posiciones** con peso > 0.")
    if locked:
        lines.append(f"- Los activos **{', '.join(locked)}** están protegidos de venta.")
    high_vol = [valid_tickers[i] for i in range(n) if vol_arr[i] > 0.5]
    if high_vol:
        lines.append(f"- Activos con alta volatilidad (>50% anual): **{', '.join(high_vol)}**. "
                     f"Considera si el riesgo es apropiado para ti.")

    return "\n".join(lines)


def _build_rebalance_ai_prompt(
    strategy: str, opt_weights: dict, current_weights: dict,
    mu_dict: dict, vol_dict: dict, opt_ret: float, opt_vol: float,
    opt_sharpe: float, locked: list, min_assets: int,
) -> str:
    """Prompt para explicación narrativa del rebalanceo con IA."""
    cambios = []
    for t in opt_weights:
        nw = opt_weights.get(t, 0)
        ow = current_weights.get(t, 0)
        cambios.append(
            f"{t}: {ow:.1%} → {nw:.1%} | "
            f"ret={mu_dict.get(t,0):.1%} vol={vol_dict.get(t,0):.1%}"
        )

    return f"""Eres un asesor financiero experto. Explica en español, de forma clara y útil para un inversor individual, POR QUÉ el algoritmo de optimización tomó las siguientes decisiones de rebalanceo. Usa lenguaje accesible, evita jerga técnica excesiva.

Responde en máximo 400 palabras con 3 secciones:
## ¿Qué hizo el optimizador?
## ¿Por qué estos cambios tienen sentido?
## ¿Qué deberías vigilar?

=== CONTEXTO ===
Estrategia: {strategy}
Activos protegidos (no vender): {locked if locked else 'ninguno'}
Mínimo de posiciones: {min_assets}

=== RESULTADO ===
Retorno esperado: {opt_ret:.1%} | Volatilidad: {opt_vol:.1%} | Sharpe: {opt_sharpe:.2f}

=== CAMBIOS DE PESO (actual → nuevo) ===
{chr(10).join(cambios)}
"""


def tab_rebalance() -> None:
    hdf = holdings_df()
    if hdf.empty:
        st.info("📂  Agrega posiciones en **Portfolio Editor** primero.")
        return

    bench   = st.session_state.get("benchmark", "SPY")
    rf_rate = st.session_state.get("rf_rate", 0.045)

    sub1, sub2, sub3, sub4 = st.tabs([
        "⚖️  Rebalanceo a Objetivo",
        "🧠  Optimización de Estrategia",
        "💡  Sugerencias Inteligentes",
        "🔍  Descubrimiento de Activos",
    ])

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 1: Rebalanceo clásico a pesos objetivo manuales
    # ══════════════════════════════════════════════════════════
    with sub1:
        tw = st.session_state.get("target_weights", {})
        if not tw:
            st.info("⚖️  Define pesos objetivo en **Portfolio Editor → Pesos Objetivo** primero.")
        else:
            col_cfg, col_res = st.columns([1, 2], gap="large")
            with col_cfg:
                section("PARÁMETROS")
                mode = st.radio("Modo:", ["Activo (compra + venta)", "Pasivo (solo comprar)"],
                                key="reb1_mode")
                is_active   = mode.startswith("Activo")
                new_capital = st.number_input("Capital adicional ($USD):", min_value=0.0,
                                              value=0.0, step=100.0, key="reb1_cap")
                commission  = st.number_input("Comisión por trade ($):", min_value=0.0,
                                              value=0.0, step=0.5, key="reb1_comm")
                min_trade   = st.number_input("Monto mínimo op. ($):", min_value=1.0,
                                              value=10.0, key="reb1_min")
                st.divider()
                st.markdown("**Target weights:**")
                for t, w in sorted(tw.items(), key=lambda x: -x[1]):
                    st.markdown(
                        f"<span style='font-family:var(--mono);font-size:0.83rem;'>"
                        f"<b style='color:#ffffff'>{t}</b> &nbsp; {w:.1%}</span>",
                        unsafe_allow_html=True)
                st.divider()
                run1 = st.button("⚖️ Calcular Rebalanceo", type="primary",
                                 use_container_width=True, key="run_reb1")

            with col_res:
                if run1 or "reb1_result" in st.session_state:
                    if run1:
                        tickers_needed = list(set(hdf["Ticker"].tolist() + list(tw.keys())))
                        with st.spinner("Obteniendo precios..."):
                            prices = fetch_live_prices(tickers_needed)
                        result = calculate_rebalance(
                            holdings=hdf, target_weights=tw, prices=prices,
                            new_capital=new_capital,
                            mode="active" if is_active else "passive",
                            min_trade_usd=min_trade, commission_per_trade=commission,
                        )
                        st.session_state["reb1_result"] = result

                    _render_rebalance_result(
                        st.session_state["reb1_result"],
                        new_capital=new_capital,
                    )
                else:
                    st.info("Configura parámetros y presiona **Calcular Rebalanceo**.")

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 2: Optimización MPT → genera pesos → rebalancea
    # ══════════════════════════════════════════════════════════
    with sub2:
        col_cfg2, col_res2 = st.columns([1, 2], gap="large")

        with col_cfg2:
            section("ESTRATEGIA DE OPTIMIZACIÓN")

            strategy = st.selectbox(
                "Estrategia:",
                STRATEGIES,
                help="MaxSharpe: max Sharpe | MinVar: min volatilidad | MaxReturn: max retorno | ERC: riesgo igualado",
                key="opt_strategy",
            )
            max_w    = st.slider("Peso máx. por activo (%):", 5, 50, 25, key="opt_maxw") / 100
            mode2    = st.radio("Modo ejecución:", ["Activo (compra + venta)", "Pasivo (solo comprar)"],
                                key="reb2_mode")
            is_active2   = mode2.startswith("Activo")
            new_capital2 = st.number_input("Capital adicional ($USD):", min_value=0.0,
                                           value=0.0, step=100.0, key="reb2_cap")
            commission2  = st.number_input("Comisión por trade ($):", min_value=0.0,
                                           value=0.0, step=0.5, key="reb2_comm")
            min_trade2   = st.number_input("Monto mínimo op. ($):", min_value=1.0,
                                           value=10.0, key="reb2_min")

            st.divider()
            st.markdown("**Activos a optimizar:**")
            tickers_held  = hdf["Ticker"].unique().tolist()
            extra_raw2 = st.text_input("Agregar activos al universo (coma):",
                                       placeholder="NVDA, META, GLD", key="opt_extra")
            extra2 = [t.strip().upper() for t in extra_raw2.split(",") if t.strip()]
            # IMPORTANT: held tickers always first
            opt_universe = list(dict.fromkeys(tickers_held + extra2))
            for t in opt_universe:
                st.markdown(
                    f"<span style='font-family:var(--mono);font-size:0.82rem;color:#aeaeb2;'>• {t}</span>",
                    unsafe_allow_html=True)

            st.divider()

            # ── FIX: Locked tickers — keep current weight, no sell ─
            locked2 = st.multiselect(
                "🔒 Proteger de ventas:",
                options=opt_universe,
                default=[],
                key="opt_locked",
                help="El optimizador no venderá NADA de estos activos. "
                     "Se mantiene exactamente el peso actual.",
            )

            # ── Mínimo / Máximo de activos ────────────────────────
            col_mn, col_mx = st.columns(2)
            with col_mn:
                min_assets2 = st.number_input(
                    "Mínimo de activos:",
                    min_value=1,
                    max_value=len(opt_universe),
                    value=max(1, len(tickers_held)),
                    step=1,
                    key="opt_min_assets",
                    help="El optimizador mantendrá al menos este número de posiciones.",
                )
            with col_mx:
                max_assets2 = st.number_input(
                    "Máximo de activos:",
                    min_value=min_assets2,
                    max_value=len(opt_universe),
                    value=len(opt_universe),
                    step=1,
                    key="opt_max_assets",
                    help=(
                        "Limita cuántos activos pueden tener peso > 0 en el resultado. "
                        "El optimizador hará dos pasadas: primero elige los mejores N, "
                        "luego re-optimiza solo con ellos."
                    ),
                )
            # None = sin límite (cuando el usuario deja el valor máximo)
            max_assets2_val = None if max_assets2 >= len(opt_universe) else int(max_assets2)

            # ── Máx. activos NUEVOS (no están en portafolio hoy) ──
            n_extra = len(extra2)
            if n_extra > 0:
                max_new_pos2 = st.number_input(
                    "Máx. activos nuevos a incorporar:",
                    min_value=0,
                    max_value=n_extra,
                    value=n_extra,
                    step=1,
                    key="opt_max_new",
                    help=(
                        "De los activos que agregaste al universo, ¿cuántos como máximo "
                        "pueden entrar al portafolio optimizado? "
                        "Ej: si añadiste 3 candidatos pero solo quieres 1 swap nuevo, pon 1."
                    ),
                )
                max_new_pos2_val = int(max_new_pos2)
            else:
                max_new_pos2_val = 0   # sin candidatos externos, no aplica

            if strategy == "SmartCluster":
                st.divider()
                st.markdown("**⚙️ SmartCluster:**")
                _nc_max = max(3, min(6, len(opt_universe)//2 + 1))
                _nc_def = min(4, max(2, len(opt_universe)//2))
                _nc_def = min(_nc_def, _nc_max)
                n_clusters2 = st.slider("Clusters:", 2, _nc_max, _nc_def,
                                        key="opt_nclusters") if _nc_max > 2 else 2
                max_cluster_w2 = st.slider("Peso máx. por cluster (%):", 20, 80, 40,
                                           key="opt_maxcluster") / 100
            else:
                n_clusters2    = 4
                max_cluster_w2 = 0.40

            # ── Filtro técnico (consistencia con Sugerencias) ──
            st.divider()
            st.markdown("**🧠 Consistencia técnica:**")
            use_tech_filter = st.toggle(
                "Aplicar filtro de señales técnicas",
                value=False,
                key="opt_use_tech",
                help=(
                    "Antes de optimizar, calcula RSI, momentum y fundamentales "
                    "de cada activo. Los que tengan señal de venta fuerte se "
                    "excluyen del universo — así el optimizador no puede "
                    "sobre-ponderar activos que el análisis técnico rechaza.\n\n"
                    "Los activos 🔒 protegidos nunca se excluyen."
                ),
            )
            tech_min_score2 = 35
            if use_tech_filter:
                tech_min_score2 = st.slider(
                    "Score mínimo para incluir en optimización:",
                    min_value=10, max_value=70, value=35, step=5,
                    key="opt_tech_score",
                    help="Activos con score técnico+fundamental por debajo de este valor se excluyen. "
                         "Score 35 = excluye solo señales de Venta / Venta Fuerte.",
                )

            st.divider()
            run2 = st.button("🧠 Optimizar & Rebalancear", type="primary",
                             use_container_width=True, key="run_opt")

        with col_res2:
            if run2 or "reb2_result" in st.session_state:
                if run2:
                    # Clear stale optimization results
                    for _k in ["opt_stats","opt_frontier","opt_weights_generated",
                               "opt_valid_t","opt_mu","opt_vol_assets","reb2_result",
                               "ai_rebalance_explanation","opt_tech_report"]:
                        st.session_state.pop(_k, None)
                    tech_filter_report = []   # se sobreescribe si el filtro está activo
                    end_h   = datetime.today()
                    start_h = end_h - timedelta(days=365 * 3)
                    with st.spinner("Descargando historial (3 años)..."):
                        hist = fetch_history(
                            opt_universe + [bench],
                            start_h.strftime("%Y-%m-%d"),
                            end_h.strftime("%Y-%m-%d"),
                        )

                    if hist.empty or len([t for t in opt_universe if t in hist.columns]) < 2:
                        st.error("No hay suficientes datos históricos para optimizar.")
                    else:
                        valid_tickers = [t for t in opt_universe if t in hist.columns]
                        ret_df = hist[valid_tickers].pct_change().dropna(how="all")

                        # ── Filtro técnico ───────────────────────────────
                        tech_filter_report = []   # para mostrar qué se excluyó
                        if use_tech_filter and valid_tickers:
                            with st.spinner("Calculando señales técnicas del universo…"):
                                end_tech   = datetime.today()
                                start_tech = end_tech - timedelta(days=180)
                                tech_scores_df = analyze_candidates(
                                    valid_tickers,
                                    start_tech.strftime("%Y-%m-%d"),
                                    end_tech.strftime("%Y-%m-%d"),
                                )
                            locked_set2 = set(locked2)
                            filtered_out = []
                            kept_in      = []
                            for t in valid_tickers:
                                # Los activos protegidos nunca se excluyen
                                if t in locked_set2:
                                    kept_in.append(t)
                                    continue
                                score = 50   # default si no hay datos
                                if not tech_scores_df.empty and t in tech_scores_df.index:
                                    row = tech_scores_df.loc[t]
                                    score = int(
                                        row.get("Tech Score", 50) * 0.55 +
                                        row.get("Fund. Score", 50) * 0.45
                                    )
                                if score >= tech_min_score2:
                                    kept_in.append(t)
                                else:
                                    filtered_out.append((t, score,
                                        tech_scores_df.loc[t, "Señal"]
                                        if not tech_scores_df.empty and t in tech_scores_df.index
                                        else "—"))

                            if len(kept_in) < 2:
                                st.warning(
                                    "El filtro técnico eliminó demasiados activos. "
                                    "Bajando el umbral a score=0 para esta corrida."
                                )
                                kept_in = valid_tickers
                                filtered_out = []

                            valid_tickers  = kept_in
                            ret_df         = hist[valid_tickers].pct_change().dropna(how="all")
                            tech_filter_report = filtered_out

                        # ── Pre-filtro: límite de activos nuevos ─────────────
                        # Se hace ANTES de optimizar para que min/max_assets
                        # sea coherente con el universo final.
                        dropped_new_tickers = []
                        extra_in_universe   = [t for t in extra2 if t in valid_tickers]
                        if n_extra > 0 and max_new_pos2_val < len(extra_in_universe):
                            # Pasada rápida para rankear candidatos por peso esperado
                            with st.spinner("Seleccionando mejores candidatos nuevos…"):
                                _pre_w = optimize_to_target_weights(
                                    valid_tickers, ret_df, rf_rate, strategy, max_w,
                                    n_clusters=n_clusters2,
                                    max_cluster_w=max_cluster_w2,
                                    locked_tickers=locked2,
                                    min_assets=2,
                                    max_assets=None,
                                )
                            extra_ranked = sorted(
                                [(t, _pre_w.get(t, 0)) for t in extra_in_universe],
                                key=lambda x: -x[1],
                            )
                            keep_new  = {t for t, _ in extra_ranked[:max_new_pos2_val]}
                            drop_new  = [t for t, _ in extra_ranked[max_new_pos2_val:]]
                            if drop_new:
                                valid_tickers       = [t for t in valid_tickers
                                                        if t not in set(drop_new)]
                                ret_df              = hist[valid_tickers].pct_change().dropna(how="all")
                                dropped_new_tickers = drop_new

                        _spin_msg = (
                            f"Calculando {strategy}"
                            + (f" (máx {max_assets2_val} activos, 2 pasadas)…"
                               if max_assets2_val else "…")
                        )
                        with st.spinner(_spin_msg):
                            opt_weights = optimize_to_target_weights(
                                valid_tickers, ret_df, rf_rate, strategy, max_w,
                                n_clusters=n_clusters2,
                                max_cluster_w=max_cluster_w2,
                                locked_tickers=locked2,
                                min_assets=min_assets2,
                                max_assets=max_assets2_val,
                            )

                        if dropped_new_tickers:
                            st.info(
                                f"🔢 Límite de activos nuevos: se excluyeron "
                                f"**{', '.join(dropped_new_tickers)}** del universo antes de optimizar. "
                                f"Sube 'Máx. activos nuevos' si quieres incluirlos."
                            )

                        # ── FIX CRÍTICO: aplicar locks al target antes del rebalanceo ─
                        # Para activos locked: target = max(opt_weight, current_weight)
                        # Esto garantiza que calculate_rebalance NUNCA sugiera venderlos
                        if locked2:
                            prices_now = fetch_live_prices(valid_tickers)
                            nav_now    = sum(
                                float(r["Shares"]) * prices_now.get(r["Ticker"],{}).get("price",0)
                                for _, r in hdf.iterrows()
                                if r["Ticker"] in prices_now
                            )
                            curr_w_now = {}
                            for _, r in hdf.iterrows():
                                t_  = r["Ticker"]
                                val = float(r["Shares"]) * prices_now.get(t_,{}).get("price",0)
                                curr_w_now[t_] = val / nav_now if nav_now > 0 else 0

                            adjusted = dict(opt_weights)
                            for t_ in locked2:
                                if t_ in adjusted and t_ in curr_w_now:
                                    # Never go below current weight for locked tickers
                                    adjusted[t_] = max(adjusted[t_], curr_w_now.get(t_, 0))

                            # Re-normalise so weights still sum to 1
                            total_adj = sum(adjusted.values())
                            if total_adj > 0:
                                opt_weights = {k: v/total_adj for k, v in adjusted.items()}

                        st.session_state["opt_weights_generated"] = opt_weights
                        st.session_state["opt_strategy_used"]     = strategy
                        st.session_state["opt_locked2"]           = locked2
                        st.session_state["opt_tech_report"]       = tech_filter_report

                        mu_arr  = _robust_mu(ret_df, valid_tickers)
                        cov_arr = ret_df[valid_tickers].cov().values * TRADING_DAYS
                        w_arr   = np.array([opt_weights.get(t, 0) for t in valid_tickers])
                        opt_ret, opt_vol, opt_sharpe = _portfolio_stats(w_arr, mu_arr, cov_arr, rf_rate)

                        frontier = get_frontier_points(valid_tickers, ret_df, rf_rate, max_w)

                        st.session_state["opt_frontier"]   = frontier
                        st.session_state["opt_stats"]      = (opt_ret, opt_vol, opt_sharpe)
                        st.session_state["opt_valid_t"]    = valid_tickers
                        st.session_state["opt_mu"]         = dict(zip(valid_tickers, mu_arr))
                        st.session_state["opt_vol_assets"] = dict(zip(valid_tickers,
                            np.sqrt(np.diag(cov_arr))))

                        with st.spinner("Calculando transacciones..."):
                            prices = fetch_live_prices(valid_tickers)
                        result2 = calculate_rebalance(
                            holdings=hdf, target_weights=opt_weights, prices=prices,
                            new_capital=new_capital2,
                            mode="active" if is_active2 else "passive",
                            min_trade_usd=min_trade2,
                            commission_per_trade=commission2,
                        )
                        st.session_state["reb2_result"] = result2

                # ── Mostrar resultados ──────────────────────────
                if "opt_stats" in st.session_state:
                    opt_ret, opt_vol, opt_sharpe = st.session_state["opt_stats"]
                    strat_used = st.session_state.get("opt_strategy_used", "")

                    section(f"PORTAFOLIO ÓPTIMO — {strat_used}")
                    sk1, sk2, sk3 = st.columns(3)
                    with sk1:
                        kpi_card("Retorno Esperado", f"{opt_ret:.1%}", "anualizado",
                                 accent="#30d158" if opt_ret > 0 else "#ff453a")
                    with sk2:
                        kpi_card("Volatilidad", f"{opt_vol:.1%}", "anualizada", accent="#ffd60a")
                    with sk3:
                        kpi_card("Sharpe Ratio", f"{opt_sharpe:.2f}", "", accent="#0a84ff")

                    # ── Reporte del filtro técnico ────────────────
                    _tech_rep = st.session_state.get("opt_tech_report", [])
                    if _tech_rep:
                        _excl_lines = "".join(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>"
                            f"<span style='font-family:DM Mono,monospace;font-weight:700;"
                            f"color:#ffffff;'>{t}</span>"
                            f"<span style='color:#ff453a;font-size:0.82rem;'>{sig}</span>"
                            f"<span style='color:#8e8e93;font-family:DM Mono,monospace;"
                            f"font-size:0.8rem;'>score {sc}</span></div>"
                            for t, sc, sig in _tech_rep
                        )
                        st.markdown(
                            f"<div style='background:rgba(255,69,58,0.07);"
                            f"border:1px solid rgba(255,69,58,0.25);border-radius:12px;"
                            f"padding:14px 18px;margin-bottom:12px;'>"
                            f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:1.5px;"
                            f"color:#ff453a;text-transform:uppercase;font-family:DM Mono,monospace;"
                            f"margin-bottom:10px;'>🚫 EXCLUIDOS POR FILTRO TÉCNICO</div>"
                            f"{_excl_lines}"
                            f"<div style='font-size:0.72rem;color:#8e8e93;margin-top:8px;'>"
                            f"Estos activos tenían señal de venta y se excluyeron del universo "
                            f"de optimización. Los 🔒 protegidos nunca se excluyen.</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    frontier = st.session_state.get("opt_frontier", pd.DataFrame())
                    if not frontier.empty:
                        valid_t   = st.session_state.get("opt_valid_t", [])
                        mu_d      = st.session_state.get("opt_mu", {})
                        vol_d     = st.session_state.get("opt_vol_assets", {})
                        opt_w_now = st.session_state.get("opt_weights_generated", {})

                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(
                            x=frontier["Volatilidad"] * 100,
                            y=frontier["Retorno"] * 100,
                            mode="markers",
                            marker=dict(color=frontier["Sharpe"],
                                        colorscale="Viridis", size=4, opacity=0.5,
                                        colorbar=dict(title="Sharpe")),
                            name="Frontera",
                            hovertemplate="Vol: %{x:.1f}%  Ret: %{y:.1f}%<extra></extra>",
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[vol_d.get(t,0)*100 for t in valid_t],
                            y=[mu_d.get(t,0)*100  for t in valid_t],
                            mode="markers+text",
                            text=valid_t, textposition="top center",
                            marker=dict(color="#8e8e93", size=9, symbol="circle-open",
                                        line=dict(width=1.5)),
                            name="Activos",
                            hovertemplate="%{text}<br>Vol:%{x:.1f}% Ret:%{y:.1f}%<extra></extra>",
                        ))
                        fig_ef.add_trace(go.Scatter(
                            x=[opt_vol*100], y=[opt_ret*100],
                            mode="markers+text",
                            text=[strat_used], textposition="top right",
                            marker=dict(color="#30d158", size=18, symbol="star",
                                        line=dict(color="white", width=1.5)),
                            name=strat_used,
                        ))
                        fig_ef.update_layout(
                            **_pl(),
                            title="Frontera Eficiente",
                            xaxis_title="Volatilidad (%)",
                            yaxis_title="Retorno Esperado (%)",
                            height=400,
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                            hovermode="closest",
                        )
                        st.plotly_chart(fig_ef, use_container_width=True, config=PLOTLY_CONFIG)

                    # SmartCluster visualization
                    if strat_used == "SmartCluster" and "smart_cluster_labels" in st.session_state:
                        sc_labels = st.session_state["smart_cluster_labels"]
                        sc_mu     = st.session_state["smart_cluster_mu"]
                        sc_vol    = st.session_state["smart_cluster_vol"]
                        sc_w      = st.session_state.get("opt_weights_generated", {})
                        CCOLS = ["#0a84ff","#30d158","#ffd60a","#ff453a","#a78bfa","#fb923c"]
                        fig_sc = go.Figure()
                        for k in sorted(set(sc_labels.values())):
                            t_in_k = [t for t, lbl in sc_labels.items() if lbl == k]
                            color  = CCOLS[k % len(CCOLS)]
                            fig_sc.add_trace(go.Scatter(
                                x=[sc_vol[t]*100 for t in t_in_k],
                                y=[sc_mu[t]*100  for t in t_in_k],
                                mode="markers+text", name=f"Cluster {k+1}",
                                text=t_in_k, textposition="top center",
                                marker=dict(
                                    size=[max(12, sc_w.get(t,0)*200) for t in t_in_k],
                                    color=color, line=dict(color="white", width=1),
                                ),
                            ))
                        fig_sc.update_layout(
                            **_pl(),
                            title="Clusters (tamaño = peso)",
                            xaxis_title="Volatilidad (%)", yaxis_title="Retorno (%)",
                            height=380,
                            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                            hovermode="closest",
                        )
                        st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)

                    # Pesos generados
                    opt_w = st.session_state.get("opt_weights_generated", {})
                    if opt_w:
                        st.markdown("<br>", unsafe_allow_html=True)
                        section("PESOS ÓPTIMOS CALCULADOS")
                        df_ow = pd.DataFrame([
                            {"Ticker": t, "Peso Óptimo": w,
                             "Estado": "🔒 Protegido" if t in st.session_state.get("opt_locked2",[])
                                       else ("✅ Mantener" if w > 0.005 else "⬇️ Liquidar")}
                            for t, w in sorted(opt_w.items(), key=lambda x: -x[1])
                            if w > 0.001
                        ])
                        st.dataframe(
                            df_ow,
                            column_config={
                                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                                "Peso Óptimo": st.column_config.ProgressColumn(
                                    "Peso", format="%.1f%%", min_value=0, max_value=1),
                                "Estado": st.column_config.TextColumn("Estado"),
                            },
                            hide_index=True, use_container_width=True,
                        )
                        if st.button("💾 Usar como Target Weights", key="adopt_opt_weights"):
                            st.session_state["target_weights"] = {
                                k: v for k, v in opt_w.items() if v > 0.001}
                            st.success("Pesos guardados como Target Weights.")

                    if "reb2_result" in st.session_state:
                        st.markdown("<br>", unsafe_allow_html=True)
                        section("PLAN DE TRANSICIÓN")
                        _render_rebalance_result(
                            st.session_state["reb2_result"],
                            new_capital=new_capital2,
                        )

                    # ── Explicación ────────────────────────────
                    if "opt_weights_generated" in st.session_state:
                        st.markdown("<br>", unsafe_allow_html=True)
                        section("🧠 ¿POR QUÉ ESTAS DECISIONES?")
                        opt_w2  = st.session_state["opt_weights_generated"]
                        strat_u = st.session_state.get("opt_strategy_used","")
                        valid_t = st.session_state.get("opt_valid_t", [])
                        mu_d    = st.session_state.get("opt_mu", {})
                        vol_d   = st.session_state.get("opt_vol_assets", {})
                        o_ret, o_vol, o_sh = st.session_state.get("opt_stats",(0,0,0))

                        hdf_now   = holdings_df()
                        live_p2   = fetch_live_prices(hdf_now["Ticker"].tolist())
                        nav2      = sum(
                            float(r["Shares"]) * live_p2.get(r["Ticker"],{}).get("price",0)
                            for _, r in hdf_now.iterrows()
                        )
                        curr_w2 = {
                            r["Ticker"]: (float(r["Shares"]) * live_p2.get(r["Ticker"],{}).get("price",0)) / nav2
                            for _, r in hdf_now.iterrows()
                            if nav2 > 0
                        }

                        exp_tab1, exp_tab2 = st.tabs(
                            ["📋 Explicación por reglas", "🤖 Narrativa con IA"])

                        with exp_tab1:
                            if valid_t:
                                mu_arr2  = np.array([mu_d.get(t,0) for t in valid_t])
                                vol_arr2 = np.array([vol_d.get(t,0) for t in valid_t])
                                cov_approx = np.diag(vol_arr2**2)
                                explanation = _build_rebalance_explanation(
                                    valid_tickers  = valid_t,
                                    opt_weights    = opt_w2,
                                    current_weights= curr_w2,
                                    mu_arr         = mu_arr2,
                                    cov_arr        = cov_approx,
                                    rf             = rf_rate,
                                    strategy       = strat_u,
                                    locked         = locked2,
                                    min_assets     = min_assets2,
                                )
                                st.markdown(explanation)

                        with exp_tab2:
                            ai_reb_key = "ai_rebalance_explanation"
                            gen_reb = st.button("✨ Generar con IA", type="primary",
                                                key="gen_reb_ai")
                            if gen_reb or ai_reb_key in st.session_state:
                                if gen_reb or ai_reb_key not in st.session_state:
                                    with st.spinner("Claude analizando…"):
                                        p = _build_rebalance_ai_prompt(
                                            strategy=strat_u, opt_weights=opt_w2,
                                            current_weights=curr_w2,
                                            mu_dict=mu_d, vol_dict=vol_d,
                                            opt_ret=o_ret, opt_vol=o_vol, opt_sharpe=o_sh,
                                            locked=locked2, min_assets=min_assets2,
                                        )
                                        st.session_state[ai_reb_key] = _call_claude(p, max_tokens=800)
                                st.markdown(st.session_state[ai_reb_key])
                                if st.button("🔄 Regenerar", key="regen_reb", type="secondary"):
                                    del st.session_state[ai_reb_key]; st.rerun()
                            else:
                                if st.session_state.get("anthropic_api_key",""):
                                    st.info("Presiona **Generar con IA** para narrativa.")
                                else:
                                    st.warning("Configura tu API Key en el sidebar.")
            else:
                st.info("Selecciona estrategia y presiona **Optimizar & Rebalancear**.")

    # ══════════════════════════════════════════════════════════
    # SUB-TAB 3: SMART SWAP ADVISOR (integrado con análisis)
    # ══════════════════════════════════════════════════════════
    with sub3:
        section("SMART SWAP ADVISOR")
        st.caption(
            "Analiza todas tus posiciones con técnico + fundamental, "
            "busca candidatos automáticamente y propone swaps específicos "
            "que puedes aprobar o rechazar uno a uno."
        )

        col_s1, col_s2 = st.columns([1, 2], gap="large")

        with col_s1:
            section("CONFIGURACIÓN")

            n_candidates = st.slider("Candidatos a escanear:", 5, 30, 12, key="sw_ncand")
            candidate_pool_raw = st.text_input(
                "Pool manual (opcional, coma):",
                placeholder="NVDA, META, AAPL…",
                key="sw_pool",
                help="Vacío = escanea automáticamente el S&P 500.",
            )
            tech_w2    = st.slider("Peso análisis técnico (%):", 20, 80, 55, key="sw_tw") / 100
            fund_w2    = 1 - tech_w2
            sell_thr   = st.slider("Score mín. para mantener:", 20, 60, 35, key="sw_sell",
                                   help="Posiciones bajo este score se marcan como candidatas a venta.")
            buy_thr    = st.slider("Score mín. para comprar:", 55, 85, 62, key="sw_buy")
            max_swaps  = st.slider("Máx. swaps a proponer:", 1, 5, 1, key="sw_maxswaps")

            # Locked tickers (no vender aunque el score sea bajo)
            tickers_now = hdf["Ticker"].unique().tolist() if not hdf.empty else []
            sw_locked   = st.multiselect(
                "🔒 No vender nunca:",
                options=tickers_now,
                default=[],
                key="sw_locked",
                help="Estas posiciones NO aparecerán como candidatas a venta aunque tengan score bajo.",
            )

            period_days2 = st.selectbox("Período análisis:", [90, 180, 365], index=1,
                                        key="sw_days", format_func=lambda d: f"{d} días")

            run_sw = st.button("🔍 Analizar & Generar Swaps", type="primary",
                               use_container_width=True, key="run_sw")

        with col_s2:
            if run_sw or "sw_result" in st.session_state:
                if run_sw:
                    end_sw   = datetime.today()
                    start_sw = end_sw - timedelta(days=period_days2)

                    # Analizar posiciones actuales
                    with st.spinner("Analizando posiciones actuales…"):
                        curr_scores = analyze_candidates(
                            tickers_now,
                            start_sw.strftime("%Y-%m-%d"),
                            end_sw.strftime("%Y-%m-%d"),
                        )

                    # Seleccionar candidatos a venta (bajo score, no locked)
                    sells_pool = []
                    if not curr_scores.empty:
                        for t in tickers_now:
                            if t in sw_locked:
                                continue
                            if t in curr_scores.index:
                                sc = curr_scores.loc[t]
                                score = (sc.get("Tech Score", 50) * tech_w2 +
                                         sc.get("Fund. Score", 50) * fund_w2)
                                if score < sell_thr:
                                    sells_pool.append((t, round(score), sc))

                    sells_pool.sort(key=lambda x: x[1])  # peor primero

                    # Buscar candidatos de compra
                    pool_raw2 = [t.strip().upper()
                                 for t in candidate_pool_raw.replace("\n"," ").split(",")
                                 if t.strip()]
                    if not pool_raw2:
                        import random as _rnd2
                        _avail2 = [t for t in UNIVERSE if t not in tickers_now]
                        _seed2  = int(datetime.today().strftime("%Y%m%d"))
                        _rng2   = _rnd2.Random(_seed2)
                        _rng2.shuffle(_avail2)
                        pool_raw2 = _avail2[:n_candidates * 6]   # pool amplio
                    candidates2 = [t for t in pool_raw2 if t not in tickers_now][:n_candidates * 3]

                    with st.spinner(f"Escaneando {len(candidates2)} candidatos…"):
                        cand_scores = analyze_candidates(
                            candidates2,
                            start_sw.strftime("%Y-%m-%d"),
                            end_sw.strftime("%Y-%m-%d"),
                        )

                    # Recalcular score con pesos del usuario
                    if not cand_scores.empty:
                        cand_scores["_score"] = (
                            cand_scores["Tech Score"] * tech_w2 +
                            cand_scores["Fund. Score"] * fund_w2
                        )
                        top_buys = (cand_scores[cand_scores["_score"] >= buy_thr]
                                    .sort_values("_score", ascending=False)
                                    .head(max_swaps * 2))
                    else:
                        top_buys = pd.DataFrame()

                    # Recalcular score para posiciones actuales también
                    if not curr_scores.empty:
                        curr_scores["_score"] = (
                            curr_scores["Tech Score"] * tech_w2 +
                            curr_scores["Fund. Score"] * fund_w2
                        )

                    st.session_state["sw_result"] = {
                        "sells_pool": sells_pool,
                        "top_buys":   top_buys,
                        "curr_scores": curr_scores,
                        "cand_scores": cand_scores,
                        "max_swaps":  max_swaps,
                        "sw_locked":  sw_locked,
                        "tech_w": tech_w2,
                        "fund_w": fund_w2,
                    }
                    # Reset per-swap decisions whenever a fresh analysis runs
                    st.session_state["sw_decisions"] = {}

                res = st.session_state.get("sw_result", {})
                sells_pool  = res.get("sells_pool", [])
                top_buys    = res.get("top_buys", pd.DataFrame())
                curr_scores = res.get("curr_scores", pd.DataFrame())

                # ── Estado de posiciones actuales ─────────────
                section("ESTADO DE TUS POSICIONES")
                if not curr_scores.empty:
                    df_curr = curr_scores.copy()
                    df_curr["Score"] = df_curr["_score"].round(0).astype(int) if "_score" in df_curr.columns else 50
                    df_curr["Estado"] = df_curr.apply(lambda r: (
                        "🔒 Protegida" if r.name in res.get("sw_locked",[])
                        else ("🔴 Candidata a venta" if r["Score"] < sell_thr
                              else ("🟡 Vigilar" if r["Score"] < (sell_thr + 10)
                                    else "🟢 OK"))),
                        axis=1
                    )
                    st.dataframe(
                        df_curr[["Score","Estado","Señal","Tech Score",
                                 "Fund. Score","RSI","Mom 3M","Ret. Período"]].reset_index(),
                        column_config={
                            "Ticker":      st.column_config.TextColumn("Ticker", width="small"),
                            "Score":       st.column_config.ProgressColumn("Score", format="%d",
                                                                            min_value=0, max_value=100),
                            "Estado":      st.column_config.TextColumn("Estado"),
                            "Señal":       st.column_config.TextColumn("Señal"),
                            "Tech Score":  st.column_config.NumberColumn("Técnico", format="%d"),
                            "Fund. Score": st.column_config.NumberColumn("Fundamental", format="%d"),
                            "RSI":         st.column_config.NumberColumn("RSI", format="%.1f"),
                            "Mom 3M":      st.column_config.NumberColumn("Mom 3M", format="%.1f%%"),
                            "Ret. Período":st.column_config.NumberColumn("Ret. %", format="%.1f%%"),
                        },
                        hide_index=True, use_container_width=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Swaps propuestos ──────────────────────────
                section("SWAPS PROPUESTOS")

                # Per-swap decisions: {swap_index: True=accepted / False=rejected}
                sw_decisions = st.session_state.get("sw_decisions", {})

                if not sells_pool and top_buys.empty:
                    st.success("✅ Todas tus posiciones tienen score aceptable y no hay "
                               "candidatos superiores. No se proponen swaps.")
                else:
                    # Construir pares SELL → BUY
                    n_pairs   = min(res.get("max_swaps", 3), max(len(sells_pool), 1))
                    sell_list = [s[0] for s in sells_pool[:n_pairs]]
                    buy_list  = (top_buys.index.tolist()[:n_pairs]
                                 if not top_buys.empty else [])

                    if not sell_list:
                        st.info("✅ Ninguna posición está por debajo del umbral de venta — "
                                "tu portafolio tiene señales técnicas/fundamentales aceptables. "
                                "Si quieres ver más candidatas a venta, **sube** el slider "
                                "'Score mín. para mantener' (ej. a 45–55).")
                    if not buy_list:
                        st.info("Sin candidatos con score alto suficiente. "
                                "Baja el umbral 'Score mín. para comprar' o amplía el pool.")

                    # ── Renderizar cada par con botones de decisión ──
                    for i, (sell_t, sell_score, sell_row) in enumerate(sells_pool[:n_pairs]):
                        buy_t     = buy_list[i] if i < len(buy_list) else None
                        buy_row   = (top_buys.loc[buy_t]
                                     if buy_t and buy_t in top_buys.index else None)
                        buy_score = int(buy_row["_score"]) if buy_row is not None else None

                        sell_sig   = sell_row.get("Señal","—") if hasattr(sell_row,"get") else "—"
                        sell_ret   = sell_row.get("Ret. Período") if hasattr(sell_row,"get") else None
                        sell_ret_s = f"{sell_ret:+.1f}%" if sell_ret is not None else "n/d"

                        buy_sig   = (buy_row.get("Señal","—")
                                     if buy_row is not None and hasattr(buy_row,"get") else "—")
                        buy_ret   = (buy_row.get("Ret. Período")
                                     if buy_row is not None and hasattr(buy_row,"get") else None)
                        buy_ret_s = f"{buy_ret:+.1f}%" if buy_ret is not None else "n/d"

                        decision = sw_decisions.get(i)  # None=pending, True=accepted, False=rejected

                        # Card visual state based on decision
                        if decision is True:
                            card_border  = "rgba(48,209,88,0.45)"
                            card_bg      = "rgba(48,209,88,0.04)"
                            status_badge = ("<span style='background:rgba(48,209,88,0.15);"
                                            "color:#30d158;padding:2px 10px;border-radius:20px;"
                                            "font-size:0.7rem;font-weight:700;'>✅ ACEPTADO</span>")
                        elif decision is False:
                            card_border  = "rgba(142,142,147,0.25)"
                            card_bg      = "rgba(17,17,24,0.6)"
                            status_badge = ("<span style='background:rgba(142,142,147,0.15);"
                                            "color:#8e8e93;padding:2px 10px;border-radius:20px;"
                                            "font-size:0.7rem;font-weight:700;'>❌ RECHAZADO</span>")
                        else:
                            card_border  = "rgba(255,255,255,0.08)"
                            card_bg      = "rgba(22,22,31,0.85)"
                            status_badge = ("<span style='background:rgba(255,214,10,0.15);"
                                            "color:#ffd60a;padding:2px 10px;border-radius:20px;"
                                            "font-size:0.7rem;font-weight:700;'>⏳ PENDIENTE</span>")

                        # Pre-build buy panel HTML to avoid backslash in f-string
                        if buy_t:
                            buy_name_html  = (f"<div style='font-size:1.3rem;font-weight:800;"
                                              f"color:#fff;font-family:DM Mono,monospace;'>"
                                              f"{buy_t}</div>")
                            buy_sig_html   = (f"<div style='font-size:0.78rem;color:#aeaeb2;"
                                              f"margin-top:4px;'>{buy_sig}</div>")
                            buy_score_clr  = "color:#30d158;"
                            buy_score_html = (f"<div style='font-size:0.72rem;color:#8e8e93;"
                                              f"margin-top:2px;'>Ret: {buy_ret_s} &nbsp;·&nbsp; "
                                              f"Score: <span style='{buy_score_clr}'>"
                                              f"{buy_score}</span></div>"
                                              if buy_score is not None else "")
                        else:
                            buy_name_html  = ("<div style='font-size:0.85rem;color:#8e8e93;'>"
                                              "Sin candidato disponible</div>")
                            buy_sig_html   = ""
                            buy_score_html = ""

                        st.markdown(f"""
<div style="background:{card_bg};border:1px solid {card_border};
            border-radius:16px;padding:18px 20px;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div style="font-size:0.62rem;font-weight:700;letter-spacing:2px;color:#636366;
                text-transform:uppercase;font-family:DM Mono,monospace;">SWAP #{i+1}</div>
    {status_badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 40px 1fr;gap:12px;align-items:center;">
    <div style="background:rgba(255,69,58,0.07);border:1px solid rgba(255,69,58,0.2);
                border-radius:12px;padding:14px 16px;">
      <div style="font-size:0.65rem;color:#ff453a;text-transform:uppercase;
                  letter-spacing:1px;font-family:DM Mono,monospace;margin-bottom:6px;">VENDER</div>
      <div style="font-size:1.3rem;font-weight:800;color:#fff;
                  font-family:DM Mono,monospace;">{sell_t}</div>
      <div style="font-size:0.78rem;color:#aeaeb2;margin-top:4px;">{sell_sig}</div>
      <div style="font-size:0.72rem;color:#8e8e93;margin-top:2px;">
        Ret: {sell_ret_s} &nbsp;·&nbsp; Score: <span style="color:#ff453a;">{sell_score}</span>
      </div>
    </div>
    <div style="text-align:center;font-size:1.4rem;color:#636366;">→</div>
    <div style="background:rgba(48,209,88,0.07);border:1px solid rgba(48,209,88,0.2);
                border-radius:12px;padding:14px 16px;">
      <div style="font-size:0.65rem;color:#30d158;text-transform:uppercase;
                  letter-spacing:1px;font-family:DM Mono,monospace;margin-bottom:6px;">COMPRAR</div>
      {buy_name_html}
      {buy_sig_html}
      {buy_score_html}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                        # Accept / Reject buttons (or Undo if already decided)
                        if decision is None:
                            _bc1, _bc2, _bc3 = st.columns([2, 1, 1])
                            with _bc2:
                                if st.button("✅ Aceptar", key=f"sw_accept_{i}",
                                             type="primary", use_container_width=True):
                                    st.session_state.setdefault("sw_decisions", {})[i] = True
                                    st.rerun()
                            with _bc3:
                                if st.button("❌ Rechazar", key=f"sw_reject_{i}",
                                             use_container_width=True):
                                    st.session_state.setdefault("sw_decisions", {})[i] = False
                                    st.rerun()
                        else:
                            _, _bundo = st.columns([5, 1])
                            with _bundo:
                                if st.button("↩️ Deshacer", key=f"sw_undo_{i}",
                                             use_container_width=True):
                                    st.session_state["sw_decisions"].pop(i, None)
                                    st.rerun()
                        st.markdown("<div style='margin-bottom:4px;'></div>",
                                    unsafe_allow_html=True)

                    # Tabla completa de candidatos
                    if not top_buys.empty:
                        with st.expander(f"📋 Todos los candidatos analizados ({len(top_buys)})"):
                            disp_cand = top_buys[["_score","Señal","Tech Score",
                                                   "Fund. Score","RSI","Mom 3M","Ret. Período"]].copy()
                            disp_cand = disp_cand.rename(columns={"_score":"Score"})
                            st.dataframe(
                                disp_cand.reset_index(),
                                column_config={
                                    "Ticker":      st.column_config.TextColumn("Ticker", width="small"),
                                    "Score":       st.column_config.ProgressColumn("Score", format="%.0f",
                                                                                    min_value=0, max_value=100),
                                    "Señal":       st.column_config.TextColumn("Señal"),
                                    "Tech Score":  st.column_config.NumberColumn("Técnico", format="%d"),
                                    "Fund. Score": st.column_config.NumberColumn("Fundamental", format="%d"),
                                    "RSI":         st.column_config.NumberColumn("RSI", format="%.1f"),
                                    "Mom 3M":      st.column_config.NumberColumn("Mom 3M", format="%.1f%%"),
                                    "Ret. Período":st.column_config.NumberColumn("Ret. %", format="%.1f%%"),
                                },
                                hide_index=True, use_container_width=True,
                            )

                    # ── Resumen y acción sobre swaps aceptados ─
                    accepted_idx = [idx for idx, v in sw_decisions.items() if v is True]
                    if accepted_idx:
                        st.markdown("<br>", unsafe_allow_html=True)
                        section("RESUMEN — SWAPS ACEPTADOS")

                        for idx in sorted(accepted_idx):
                            if idx >= len(sells_pool[:n_pairs]):
                                continue
                            _st, _ss, _ = sells_pool[idx]
                            _bt = buy_list[idx] if idx < len(buy_list) else None
                            if _bt:
                                st.markdown(
                                    f"**Swap #{idx+1}:** Vender `{_st}` (score {_ss})"
                                    f" → Comprar `{_bt}`"
                                )
                            else:
                                st.markdown(
                                    f"**Swap #{idx+1}:** Vender `{_st}` (score {_ss})"
                                    f" — sin candidato de compra disponible"
                                )

                        accepted_buys = [buy_list[idx] for idx in sorted(accepted_idx)
                                         if idx < len(buy_list) and buy_list[idx]]
                        if accepted_buys:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                "<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
                                "color:#636366;text-transform:uppercase;font-family:DM Mono,monospace;"
                                "margin-bottom:8px;'>LLEVAR AL OPTIMIZADOR</div>",
                                unsafe_allow_html=True)
                            st.caption(
                                "Solo los swaps **aceptados** se añaden al universo del "
                                "sub-tab **🧠 Optimización** para que el algoritmo decida "
                                "el peso óptimo de cada uno.")
                            st.code(", ".join(accepted_buys), language=None)
                            if st.button("➕ Agregar aceptados al universo de optimización",
                                         type="primary", use_container_width=True,
                                         key="sw_to_opt"):
                                existing = st.session_state.get("_sw_extras", "")
                                combined = list(dict.fromkeys(
                                    [t.strip() for t in existing.split(",") if t.strip()]
                                    + accepted_buys
                                ))
                                st.session_state["_sw_extras"] = ", ".join(combined)
                                st.success(
                                    f"✅ {', '.join(accepted_buys)} añadidos. "
                                    "Ve al sub-tab **🧠 Optimización** y verás estos tickers "
                                    "en el campo 'Agregar activos al universo'.")
            else:
                st.info("Configura los parámetros y presiona **Analizar & Generar Swaps**.")

    with sub4:
        _render_discovery_tab()


# ─────────────────────────────────────────────────────────────
# HELPER: Render rebalance result (shared by sub-tabs 1 y 2)
# ─────────────────────────────────────────────────────────────

def _render_rebalance_result(result: dict, new_capital: float = 0.0) -> None:
    if not result.get("ok"):
        st.error(result.get("error", "Error desconocido."))
        return

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("NAV Actual",      f"${result['nav']:,.0f}")
    with k2: kpi_card("Capital Total",   f"${result['total_capital']:,.0f}",
                      f"+ ${new_capital:,.0f} nuevo" if new_capital > 0 else "")
    with k3: kpi_card("Compras",  f"${result['buys_total']:,.0f}",  accent="#30d158")
    with k4: kpi_card("Ventas",   f"${result['sells_total']:,.0f}", accent="#ff453a" if result["sells_total"] > 0 else "")

    k5, k6, k7, _ = st.columns(4)
    with k5: kpi_card("Turnover",    f"{result['turnover']:.1%}", accent="#ffd60a")
    with k6: kpi_card("N° Trades",   str(result.get("n_trades", 0)))
    with k7:
        comm = result.get("estimated_commissions", 0)
        kpi_card("Comisiones", f"${comm:.2f}" if comm > 0 else "—")

    trades = result.get("trades", [])
    if not trades:
        st.success("Portafolio ya balanceado."); return

    df_t = pd.DataFrame(trades)
    df_t["_o"] = df_t["Acción"].map({"SELL":0,"BUY":1,"HOLD":2})
    df_t = df_t.sort_values(["_o","Delta USD"], ascending=[True,True]).drop("_o", axis=1)

    display = df_t[df_t["Acción"] != "HOLD"].copy()
    if display.empty:
        st.success("✅ Sin transacciones necesarias (dentro del umbral mínimo).")
    else:
        # Columna "Monto USD" = |Delta USD| — lo que debes ingresar en GBM
        display["Monto USD"] = display["Delta USD"].abs()

        # ── Vista para EJECUTAR en broker (Ticker, Acción, Monto, Títulos) ──
        st.markdown(
            "<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
            "color:#636366;text-transform:uppercase;font-family:DM Mono,monospace;"
            "margin-bottom:6px;'>ÓRDENES A EJECUTAR</div>",
            unsafe_allow_html=True)

        exec_view = display[["Ticker","Acción","Precio","Monto USD","Acciones"]].copy()
        st.dataframe(
            exec_view,
            column_config={
                "Ticker":    st.column_config.TextColumn("Ticker",   width="small"),
                "Acción":    st.column_config.TextColumn("Acción",   width="small"),
                "Precio":    st.column_config.NumberColumn("Precio", format="$%.2f"),
                "Monto USD": st.column_config.NumberColumn(
                    "💵 Monto a invertir",
                    format="$%.2f",
                    help="Este es el monto en USD que debes ingresar en GBM u otro broker",
                ),
                "Acciones":  st.column_config.NumberColumn("Títulos",format="%+.6f"),
            },
            hide_index=True, use_container_width=True,
        )

        st.caption(
            "💡 En **GBM** y la mayoría de brokers usa la columna **Monto a invertir** — "
            "ingresa ese valor en dólares en el campo de monto. "
            "Los títulos son de referencia para brokers que operan por cantidad."
        )

        # ── Vista completa con todos los datos ──────────────────
        with st.expander("📋 Ver tabla completa con pesos y drift"):
            st.dataframe(
                display.drop("Monto USD", axis=1),
                column_config={
                    "Ticker":        st.column_config.TextColumn("Ticker",    width="small"),
                    "Acción":        st.column_config.TextColumn("Acción",    width="small"),
                    "Precio":        st.column_config.NumberColumn("Precio",  format="$%.2f"),
                    "Peso Actual":   st.column_config.NumberColumn("Actual",  format="%.1f%%"),
                    "Peso Objetivo": st.column_config.NumberColumn("Objetivo",format="%.1f%%"),
                    "Drift":         st.column_config.NumberColumn("Drift",   format="%+.1f%%"),
                    "Val. Actual":   st.column_config.NumberColumn("Actual $",format="$%.0f"),
                    "Val. Objetivo": st.column_config.NumberColumn("Obj. $",  format="$%.0f"),
                    "Delta USD":     st.column_config.NumberColumn("Delta",   format="$%+,.2f"),
                    "Acciones":      st.column_config.NumberColumn("Títulos", format="%+.6f"),
                },
                hide_index=True, use_container_width=True,
            )

        csv = display.drop("Monto USD", axis=1).to_csv(index=False)
        st.download_button("⬇️ Descargar plan (.csv)", data=csv,
                           file_name=f"rebalanceo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

    holds = df_t[df_t["Acción"] == "HOLD"].copy()
    if not holds.empty:
        with st.expander(f"📌 Sin cambio ({len(holds)})"):
            st.dataframe(holds[["Ticker","Precio","Peso Actual","Peso Objetivo","Val. Actual"]],
                         column_config={
                             "Precio":        st.column_config.NumberColumn(format="$%.2f"),
                             "Peso Actual":   st.column_config.NumberColumn(format="%.1f%%"),
                             "Peso Objetivo": st.column_config.NumberColumn(format="%.1f%%"),
                             "Val. Actual":   st.column_config.NumberColumn(format="$%.0f"),
                         }, hide_index=True, use_container_width=True)

    # Gráfico Before/After
    cw = result["current_weights"]
    tw = result["target_weights"]
    all_t = sorted(set(list(cw.keys()) + list(tw.keys())))
    fig_ba = go.Figure()
    fig_ba.add_trace(go.Bar(name="Actual",   x=all_t,
                            y=[cw.get(t,0)*100 for t in all_t],
                            marker_color="#0a84ff",
                            hovertemplate="%{x}: %{y:.1f}%<extra>Actual</extra>"))
    fig_ba.add_trace(go.Bar(name="Objetivo", x=all_t,
                            y=[tw.get(t,0)*100 for t in all_t],
                            marker_color="#30d158", opacity=0.75,
                            hovertemplate="%{x}: %{y:.1f}%<extra>Objetivo</extra>"))
    fig_ba.update_layout(**PLOTLY_LAYOUT, barmode="group",
                         yaxis_title="Peso (%)", height=320)
    st.plotly_chart(fig_ba, use_container_width=True, config=PLOTLY_CONFIG)


# ─────────────────────────────────────────────────────────────
# TAB 4: PERFORMANCE VS BENCHMARK
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# DEEP ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_deep_analysis(ticker: str) -> dict:
    """
    Análisis completo de un activo:
      · Fundamentales: EBITDA, EV/EBITDA, ratios clave
      · Consenso de analistas: buy/hold/sell, precio objetivo
      · Noticias recientes (últimas 5)
      · Histórico: retornos 1M/3M/6M/1Y, volatilidad, Sharpe, max DD
    """
    out = {
        "ticker": ticker, "name": ticker, "sector": "—", "industry": "—",
        "currency": "USD", "price": 0.0, "change_1d": 0.0,
        # Valuación
        "pe":  None, "forward_pe": None, "peg": None,
        "ev_ebitda": None, "pb": None, "ps": None,
        # Calidad
        "roe": None, "roa": None, "profit_margin": None,
        "op_margin": None, "rev_growth": None, "earn_growth": None,
        "ebitda": None, "fcf": None,
        # Salud financiera
        "debt_equity": None, "current_ratio": None,
        # Analistas
        "analyst_key": None, "n_analysts": 0,
        "target_mean": None, "target_high": None, "target_low": None,
        "upside": None,
        "buy": 0, "hold": 0, "sell": 0,
        # Noticias
        "news": [],
        # Histórico
        "ret_1m": None, "ret_3m": None, "ret_6m": None, "ret_1y": None,
        "vol_1y": None, "sharpe_1y": None, "max_dd_1y": None,
        # Scores
        "score_fund": 50, "score_analyst": 50, "score_hist": 50, "score_total": 50,
        "ok": False, "error": None,
    }

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        # ── Identidad y precio ───────────────────────────────
        out["name"]      = info.get("longName") or info.get("shortName") or ticker
        out["sector"]    = info.get("sector",   "—")
        out["industry"]  = info.get("industry", "—")
        out["currency"]  = info.get("currency", "USD")
        out["price"]     = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        prev = float(info.get("regularMarketPreviousClose") or info.get("previousClose") or 0)
        out["change_1d"] = (out["price"] - prev) / prev if prev > 0 else 0.0

        # ── Valuación ────────────────────────────────────────
        def _f(key): return round(float(v), 2) if (v := info.get(key)) is not None else None

        out["pe"]         = _f("trailingPE")
        out["forward_pe"] = _f("forwardPE")
        out["peg"]        = _f("pegRatio")
        out["ev_ebitda"]  = _f("enterpriseToEbitda")
        out["pb"]         = _f("priceToBook")
        out["ps"]         = _f("priceToSalesTrailing12Months")

        # ── Calidad / Rentabilidad ───────────────────────────
        out["roe"]          = _f("returnOnEquity")
        out["roa"]          = _f("returnOnAssets")
        out["profit_margin"]= _f("profitMargins")
        out["op_margin"]    = _f("operatingMargins")
        out["rev_growth"]   = _f("revenueGrowth")
        out["earn_growth"]  = _f("earningsGrowth")
        ebitda = info.get("ebitda")
        fcf    = info.get("freeCashflow")
        out["ebitda"] = int(ebitda) if ebitda else None
        out["fcf"]    = int(fcf)    if fcf    else None

        # ── Salud financiera ─────────────────────────────────
        out["debt_equity"]  = _f("debtToEquity")
        out["current_ratio"]= _f("currentRatio")

        # ── Consenso analistas ───────────────────────────────
        out["analyst_key"] = info.get("recommendationKey", "—")
        out["n_analysts"]  = int(info.get("numberOfAnalystOpinions") or 0)
        out["target_mean"] = _f("targetMeanPrice")
        out["target_high"] = _f("targetHighPrice")
        out["target_low"]  = _f("targetLowPrice")
        if out["target_mean"] and out["price"] > 0:
            out["upside"] = (out["target_mean"] - out["price"]) / out["price"]

        # Buy / Hold / Sell counts desde recommendations_summary
        try:
            rec = tk.recommendations
            if rec is not None and not rec.empty:
                # Tomar últimos 3 meses
                last = rec.tail(3)
                for col in last.columns:
                    cl = col.lower()
                    if "strong buy" in cl or "strongbuy" in cl:
                        out["buy"]  += int(last[col].sum())
                    elif "buy" in cl:
                        out["buy"]  += int(last[col].sum())
                    elif "hold" in cl or "neutral" in cl:
                        out["hold"] += int(last[col].sum())
                    elif "sell" in cl or "underperform" in cl:
                        out["sell"] += int(last[col].sum())
        except Exception:
            pass

        # ── Noticias ─────────────────────────────────────────
        try:
            news_raw = tk.news or []
            out["news"] = [
                {
                    "title":     n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link":      n.get("link", ""),
                    "age":       _news_age(n.get("providerPublishTime", 0)),
                }
                for n in news_raw[:5]
            ]
        except Exception:
            pass

        # ── Histórico 1 año ───────────────────────────────────
        try:
            hist = tk.history(period="1y", auto_adjust=True)
            if not hist.empty and len(hist) > 20:
                prices = hist["Close"]
                rets   = prices.pct_change().dropna()

                def _ret(days):
                    if len(prices) >= days:
                        return round((prices.iloc[-1] / prices.iloc[-days] - 1) * 100, 2)
                    return None

                out["ret_1m"]  = _ret(21)
                out["ret_3m"]  = _ret(63)
                out["ret_6m"]  = _ret(126)
                out["ret_1y"]  = _ret(252)
                vol = rets.std() * np.sqrt(252)
                out["vol_1y"]  = round(float(vol) * 100, 2)
                ann_ret = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
                out["sharpe_1y"] = round((ann_ret - 0.045) / float(vol), 2) if vol > 0 else None
                peak   = prices.cummax()
                dd     = (prices - peak) / peak
                out["max_dd_1y"] = round(float(dd.min()) * 100, 2)
        except Exception:
            pass

        # ── Scores ────────────────────────────────────────────
        out["score_fund"]     = _score_fundamental(out)
        out["score_analyst"]  = _score_analyst(out)
        out["score_hist"]     = _score_historical(out)
        out["score_total"]    = round(
            out["score_fund"]    * 0.40 +
            out["score_analyst"] * 0.35 +
            out["score_hist"]    * 0.25
        )
        out["ok"] = True

    except Exception as e:
        out["error"] = str(e)

    return out


def _news_age(ts: int) -> str:
    if not ts:
        return ""
    try:
        delta = datetime.now() - datetime.fromtimestamp(ts)
        if delta.days == 0:
            h = delta.seconds // 3600
            return f"hace {h}h" if h > 0 else "hace menos de 1h"
        return f"hace {delta.days}d"
    except Exception:
        return ""


def _score_fundamental(d: dict) -> int:
    s = 50
    # P/E
    pe = d.get("pe")
    if pe:
        if   0 < pe < 12:  s += 18
        elif 12 <= pe < 20: s += 10
        elif 20 <= pe < 30: s += 3
        elif pe > 50:       s -= 12
    # Forward P/E mejor que trailing = expectativa positiva
    fpe = d.get("forward_pe")
    if pe and fpe and fpe < pe:
        s += 5
    # PEG
    peg = d.get("peg")
    if peg:
        if   0 < peg < 1:   s += 12
        elif 1 <= peg < 2:  s += 4
        elif peg > 3:        s -= 8
    # EV/EBITDA
    ev = d.get("ev_ebitda")
    if ev:
        if   0 < ev < 10:  s += 10
        elif 10 <= ev < 20: s += 3
        elif ev > 30:        s -= 8
    # ROE
    roe = d.get("roe")
    if roe:
        if   roe > 0.30:  s += 14
        elif roe > 0.15:  s += 7
        elif roe < 0:     s -= 12
    # Margen
    mg = d.get("profit_margin")
    if mg:
        if   mg > 0.25:  s += 10
        elif mg > 0.10:  s += 4
        elif mg < 0.02:  s -= 6
    # Crecimiento ingresos
    rg = d.get("rev_growth")
    if rg:
        if   rg > 0.20:  s += 12
        elif rg > 0.08:  s += 5
        elif rg < 0:     s -= 10
    # Deuda
    de = d.get("debt_equity")
    if de:
        if   de < 30:   s += 6
        elif de < 80:   s += 2
        elif de > 200:  s -= 10
    return max(0, min(100, s))


def _score_analyst(d: dict) -> int:
    s = 50
    # Recomendación clave
    key = (d.get("analyst_key") or "").lower()
    if   "strong_buy" in key or "strongbuy" in key: s += 30
    elif "buy"        in key:                        s += 18
    elif "hold"       in key or "neutral" in key:    s +=  0
    elif "underperform" in key or "sell" in key:     s -= 20
    # Upside vs precio objetivo
    up = d.get("upside")
    if up is not None:
        if   up > 0.30:  s += 20
        elif up > 0.15:  s += 12
        elif up > 0.05:  s +=  5
        elif up < -0.05: s -= 10
        elif up < -0.15: s -= 18
    # Consenso buy/hold/sell
    total = d["buy"] + d["hold"] + d["sell"]
    if total > 0:
        buy_pct = d["buy"] / total
        if   buy_pct > 0.75: s += 10
        elif buy_pct > 0.50: s +=  4
        elif buy_pct < 0.25: s -= 10
    return max(0, min(100, s))


def _score_historical(d: dict) -> int:
    s = 50
    r1y = d.get("ret_1y")
    if r1y is not None:
        if   r1y > 40:  s += 18
        elif r1y > 15:  s += 10
        elif r1y > 0:   s +=  3
        elif r1y < -20: s -= 15
        elif r1y < -5:  s -=  7
    sh = d.get("sharpe_1y")
    if sh is not None:
        if   sh > 2.0:  s += 15
        elif sh > 1.0:  s +=  8
        elif sh > 0.5:  s +=  2
        elif sh < 0:    s -= 12
    vol = d.get("vol_1y")
    if vol is not None:
        if   vol < 20:  s += 8
        elif vol < 35:  s += 3
        elif vol > 60:  s -= 8
    r3m = d.get("ret_3m")
    if r3m is not None:
        if   r3m > 15:  s += 8
        elif r3m > 5:   s += 3
        elif r3m < -10: s -= 8
    return max(0, min(100, s))


def _fmt_big(n) -> str:
    """Formatea números grandes: 45_000_000_000 → '$45.0B'"""
    if n is None: return "—"
    if abs(n) >= 1e12: return f"${n/1e12:.1f}T"
    if abs(n) >= 1e9:  return f"${n/1e9:.1f}B"
    if abs(n) >= 1e6:  return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"


def _analyst_badge(key: str) -> str:
    k = (key or "").lower()
    if "strong_buy" in k or "strongbuy" in k:
        return ("🟢", "STRONG BUY",  "#30d158")
    elif "buy" in k:
        return ("🟢", "BUY",         "#86efac")
    elif "hold" in k or "neutral" in k:
        return ("🟡", "HOLD",        "#ffd60a")
    elif "underperform" in k:
        return ("🔴", "UNDERPERFORM","#fb923c")
    elif "sell" in k:
        return ("🔴", "SELL",        "#ff453a")
    return ("⚪", "—", "#8e8e93")


def _score_color(s: int) -> str:
    if s >= 70: return "#30d158"
    if s >= 50: return "#ffd60a"
    return "#ff453a"


def _render_candidate_card(d: dict, idx: int) -> None:
    """Renderiza una tarjeta de candidato con análisis completo."""
    is_selected = st.session_state.get("discovery_selected", set())
    selected    = d["ticker"] in is_selected
    border_c    = "#30d158" if selected else "rgba(255,255,255,0.07)"
    score_c     = _score_color(d["score_total"])
    emoji_k, label_k, color_k = _analyst_badge(d.get("analyst_key",""))
    chg_color   = "#30d158" if d["change_1d"] >= 0 else "#ff453a"
    chg_sign    = "▲" if d["change_1d"] >= 0 else "▼"

    st.markdown(f"""
    <div style="background:rgba(22,22,31,0.9); border:1.5px solid {border_c};
                border-radius:18px; padding:20px 22px; margin-bottom:18px;
                transition: border-color 0.2s;">

      <!-- HEADER -->
      <div style="display:flex; align-items:flex-start; justify-content:space-between;
                  margin-bottom:14px;">
        <div>
          <span style="font-size:1.4rem; font-weight:800; color:#fff;
                       font-family:'DM Mono',monospace; letter-spacing:-0.5px;">
            {d['ticker']}
          </span>
          <span style="font-size:0.8rem; color:#aeaeb2; margin-left:10px;">
            {d['name'][:35]}{"…" if len(d['name'])>35 else ""}
          </span><br>
          <span style="font-size:0.75rem; color:#8e8e93;">{d['sector']} · {d['industry'][:30]}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.3rem; font-weight:700; color:#fff;
                      font-family:'DM Mono',monospace;">${d['price']:,.2f}</div>
          <div style="font-size:0.85rem; color:{chg_color}; font-family:'DM Mono',monospace;">
            {chg_sign} {abs(d['change_1d']):.2%} hoy
          </div>
        </div>
      </div>

      <!-- SCORES -->
      <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
                  margin-bottom:16px;">
        <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                    padding:8px 10px; text-align:center;">
          <div style="font-size:0.65rem; color:#8e8e93; text-transform:uppercase;
                      font-family:'DM Mono',monospace; margin-bottom:3px;">Score Total</div>
          <div style="font-size:1.4rem; font-weight:800; color:{score_c};
                      font-family:'DM Mono',monospace;">{d['score_total']}</div>
        </div>
        <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                    padding:8px 10px; text-align:center;">
          <div style="font-size:0.65rem; color:#8e8e93; text-transform:uppercase;
                      font-family:'DM Mono',monospace; margin-bottom:3px;">Fundamental</div>
          <div style="font-size:1.2rem; font-weight:700; color:{_score_color(d['score_fund'])};
                      font-family:'DM Mono',monospace;">{d['score_fund']}</div>
        </div>
        <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                    padding:8px 10px; text-align:center;">
          <div style="font-size:0.65rem; color:#8e8e93; text-transform:uppercase;
                      font-family:'DM Mono',monospace; margin-bottom:3px;">Analistas</div>
          <div style="font-size:1.2rem; font-weight:700; color:{_score_color(d['score_analyst'])};
                      font-family:'DM Mono',monospace;">{d['score_analyst']}</div>
        </div>
        <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                    padding:8px 10px; text-align:center;">
          <div style="font-size:0.65rem; color:#8e8e93; text-transform:uppercase;
                      font-family:'DM Mono',monospace; margin-bottom:3px;">Histórico</div>
          <div style="font-size:1.2rem; font-weight:700; color:{_score_color(d['score_hist'])};
                      font-family:'DM Mono',monospace;">{d['score_hist']}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detalles en expander ──────────────────────────────────
    with st.expander(f"Ver análisis completo de {d['ticker']}"):
        col_f, col_a, col_h = st.columns(3)

        # Fundamentales
        with col_f:
            st.markdown("**📊 Fundamentales**")
            metrics = [
                ("P/E",           d["pe"],           "{:.1f}"),
                ("P/E Forward",   d["forward_pe"],   "{:.1f}"),
                ("PEG",           d["peg"],           "{:.2f}"),
                ("EV/EBITDA",     d["ev_ebitda"],    "{:.1f}"),
                ("P/Book",        d["pb"],            "{:.2f}"),
                ("ROE",           d["roe"],           "{:.1%}" if d["roe"] else "{:.2f}"),
                ("Margen Neto",   d["profit_margin"],"{ :.1%}" if d["profit_margin"] else "{:.2f}"),
                ("Crec. Ingresos",d["rev_growth"],   "{:.1%}" if d["rev_growth"] else "{:.2f}"),
                ("Deuda/Capital", d["debt_equity"],  "{:.1f}"),
                ("EBITDA",        None, ""),
                ("Free Cash Flow",None, ""),
            ]
            # Reemplazar EBITDA y FCF con formato big
            rows_f = []
            for label, val, fmt in metrics[:-2]:
                if val is not None:
                    try:
                        if "%" in fmt:
                            disp = fmt.format(val)
                        else:
                            disp = fmt.format(val)
                    except Exception:
                        disp = str(val)
                    rows_f.append(f"- **{label}**: `{disp}`")
            rows_f.append(f"- **EBITDA**: `{_fmt_big(d['ebitda'])}`")
            rows_f.append(f"- **Free Cash Flow**: `{_fmt_big(d['fcf'])}`")
            st.markdown("\n".join(rows_f))

        # Analistas
        with col_a:
            st.markdown("**🎯 Consenso Analistas**")
            total_an = d["buy"] + d["hold"] + d["sell"]
            st.markdown(f"""
- **Recomendación**: <span style="color:{color_k}; font-weight:700;">{emoji_k} {label_k}</span>
- **N° analistas**: `{d['n_analysts']}`
- **Precio objetivo**: `${d['target_mean'] or '—'}`
- **Rango**: `${d['target_low'] or '—'}` – `${d['target_high'] or '—'}`
- **Upside**: `{f"{d['upside']:+.1%}" if d['upside'] is not None else '—'}`
""", unsafe_allow_html=True)
            if total_an > 0:
                buy_w  = int(d["buy"]  / total_an * 100)
                hold_w = int(d["hold"] / total_an * 100)
                sell_w = 100 - buy_w - hold_w
                st.markdown(
                    f'<div style="display:flex; height:10px; border-radius:5px; overflow:hidden; margin-top:8px;">'
                    f'<div style="width:{buy_w}%;  background:#30d158;"></div>'
                    f'<div style="width:{hold_w}%; background:#ffd60a;"></div>'
                    f'<div style="width:{sell_w}%; background:#ff453a;"></div>'
                    f'</div>'
                    f'<div style="font-size:0.72rem; color:#8e8e93; margin-top:4px; font-family:var(--mono);">'
                    f'🟢 {d["buy"]} compra &nbsp; 🟡 {d["hold"]} mantener &nbsp; 🔴 {d["sell"]} vender'
                    f'</div>', unsafe_allow_html=True
                )
            # Noticias
            if d["news"]:
                st.markdown("**📰 Noticias recientes**")
                for n in d["news"]:
                    if n["title"]:
                        age_str = f" · *{n['age']}*" if n["age"] else ""
                        pub_str = f" — {n['publisher']}" if n["publisher"] else ""
                        link    = n.get("link","")
                        if link:
                            st.markdown(f"- [{n['title'][:70]}]({link}){pub_str}{age_str}")
                        else:
                            st.markdown(f"- {n['title'][:70]}{pub_str}{age_str}")

        # Histórico
        with col_h:
            st.markdown("**📈 Rendimiento Histórico**")
            def _ret_badge(v):
                if v is None: return "—"
                c = "#30d158" if v > 0 else "#ff453a"
                return f'<span style="color:{c}; font-family:var(--mono);">{v:+.1f}%</span>'

            st.markdown(
                f"- **Retorno 1M**: {_ret_badge(d['ret_1m'])}<br>"
                f"- **Retorno 3M**: {_ret_badge(d['ret_3m'])}<br>"
                f"- **Retorno 6M**: {_ret_badge(d['ret_6m'])}<br>"
                f"- **Retorno 1A**: {_ret_badge(d['ret_1y'])}<br>"
                f"- **Volatilidad 1A**: `{d['vol_1y'] or '—'}%`<br>"
                f"- **Sharpe 1A**: `{d['sharpe_1y'] or '—'}`<br>"
                f"- **Max Drawdown**: "
                + (f'<span style="color:#ff453a;font-family:var(--mono);">'
                   f'{d["max_dd_1y"]}%</span>' if d["max_dd_1y"] is not None else "—"),
                unsafe_allow_html=True,
            )


def _render_discovery_tab() -> None:
    section("DESCUBRIMIENTO DE ACTIVOS")
    st.caption(
        "Analiza candidatos con fundamentales (EBITDA, ratios, márgenes), "
        "consenso de analistas, noticias recientes e historial de rendimiento. "
        "Elige los que más te convenzan y llévalos al optimizador."
    )

    col_cfg, col_res = st.columns([1, 3], gap="large")

    with col_cfg:
        section("CONFIGURACIÓN")
        n_show = st.slider("Mejores candidatos a mostrar:", 3, 10, 5, key="disc_n")
        pool_raw = st.text_area(
            "Tickers a analizar (uno por línea o coma):",
            placeholder="NVDA\nMETA\nMSFT\nAMZN\nGOOGL",
            height=160,
            key="disc_pool",
            help="Deja vacío para usar los primeros activos del universo S&P 500.",
        )
        # Excluir los que ya tengo
        hdf_now     = holdings_df()
        already_own = set(hdf_now["Ticker"].tolist() if not hdf_now.empty else [])
        excl = st.multiselect(
            "Excluir de candidatos:",
            sorted(already_own),
            default=sorted(already_own),
            key="disc_excl",
            help="Por defecto se excluyen activos que ya tienes.",
        )

        # Pesos del score
        st.markdown("**Ponderación del score total:**")
        w_fund = st.slider("Fundamentales %:", 10, 70, 40, 5, key="disc_wf")
        w_an   = st.slider("Analistas %:",     10, 70, 35, 5, key="disc_wa")
        w_hist = max(0, 100 - w_fund - w_an)
        st.caption(f"Histórico: {w_hist}% (calculado automáticamente)")

        run_disc = st.button("🔍 Analizar candidatos", type="primary",
                             use_container_width=True, key="run_disc")

    with col_res:
        if run_disc or "disc_results" in st.session_state:
            if run_disc:
                # Construir pool
                raw = [t.strip().upper().replace(",","")
                       for t in pool_raw.replace("\n"," ").split()
                       if t.strip()]
                if not raw:
                    import random as _rnd
                    _available = [t for t in UNIVERSE if t not in excl]
                    # Semilla diaria: mismo resultado durante el día, varía cada día
                    _seed = int(datetime.today().strftime("%Y%m%d"))
                    _rng  = _rnd.Random(_seed)
                    _rng.shuffle(_available)
                    raw = _available[:n_show * 8]   # pool amplio para que haya variedad

                candidates = [t for t in raw if t not in excl][:n_show * 8]

                if not candidates:
                    st.warning("Sin candidatos válidos. Revisa los tickers o quita exclusiones.")
                    st.stop()

                progress = st.progress(0, text="Iniciando análisis...")
                results  = []
                for i, ticker in enumerate(candidates):
                    progress.progress(
                        (i + 1) / len(candidates),
                        text=f"Analizando {ticker} ({i+1}/{len(candidates)})…"
                    )
                    d = fetch_deep_analysis(ticker)
                    if d["ok"]:
                        results.append(d)
                progress.empty()

                if not results:
                    st.error("No se obtuvieron datos. Verifica los tickers.")
                    st.stop()

                # Recalcular score con pesos del usuario
                for d in results:
                    d["score_total"] = round(
                        d["score_fund"]    * (w_fund / 100) +
                        d["score_analyst"] * (w_an   / 100) +
                        d["score_hist"]    * (w_hist  / 100)
                    )

                # Ordenar y tomar top N
                results.sort(key=lambda x: x["score_total"], reverse=True)
                st.session_state["disc_results"]    = results[:n_show]
                st.session_state["disc_w_fund"]     = w_fund
                st.session_state["disc_w_an"]       = w_an
                st.session_state["disc_w_hist"]     = w_hist
                if "discovery_selected" not in st.session_state:
                    st.session_state["discovery_selected"] = set()

            results = st.session_state.get("disc_results", [])
            if not results:
                st.info("Sin resultados.")
                return

            # ── Resumen comparativo ───────────────────────────
            section(f"TOP {len(results)} CANDIDATOS (ordenados por score)")

            # Tabla rápida de comparación
            df_sum = pd.DataFrame([{
                "Ticker":    d["ticker"],
                "Nombre":    d["name"][:25],
                "Score":     d["score_total"],
                "Fund.":     d["score_fund"],
                "Analistas": d["score_analyst"],
                "Histór.":   d["score_hist"],
                "Recom.":    (d.get("analyst_key") or "—").replace("_"," ").upper(),
                "Upside":    d["upside"],
                "Ret. 1A":   d["ret_1y"],
                "Sharpe 1A": d["sharpe_1y"],
                "P/E":       d["pe"],
                "EV/EBITDA": d["ev_ebitda"],
                "ROE":       d["roe"],
            } for d in results])

            st.dataframe(
                df_sum,
                column_config={
                    "Ticker":    st.column_config.TextColumn("Ticker",   width="small"),
                    "Nombre":    st.column_config.TextColumn("Empresa",  width="medium"),
                    "Score":     st.column_config.ProgressColumn("Score Total", format="%d",
                                                                  min_value=0, max_value=100),
                    "Fund.":     st.column_config.NumberColumn("Fund.",  format="%d"),
                    "Analistas": st.column_config.NumberColumn("Anal.",  format="%d"),
                    "Histór.":   st.column_config.NumberColumn("Hist.",  format="%d"),
                    "Recom.":    st.column_config.TextColumn("Recom.",   width="medium"),
                    "Upside":    st.column_config.NumberColumn("Upside", format="%+.1f%%"),
                    "Ret. 1A":   st.column_config.NumberColumn("Ret. 1A",format="%+.1f%%"),
                    "Sharpe 1A": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "P/E":       st.column_config.NumberColumn("P/E",    format="%.1f"),
                    "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA", format="%.1f"),
                    "ROE":       st.column_config.NumberColumn("ROE",    format="%.1%"),
                },
                hide_index=True, use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Instrucción de selección ──────────────────────
            st.info(
                "👇 Revisa cada tarjeta y **selecciona** los activos que quieres "
                "añadir a tu universo de optimización. Luego usa el botón al final."
            )

            # ── Tarjetas detalladas ───────────────────────────
            for i, d in enumerate(results):
                _render_candidate_card(d, i)

                # Botón de selección debajo de cada tarjeta
                sel_set = st.session_state.get("discovery_selected", set())
                is_sel  = d["ticker"] in sel_set
                btn_label = f"✅ Seleccionado — {d['ticker']}" if is_sel else f"➕ Seleccionar {d['ticker']}"
                btn_type  = "primary" if is_sel else "secondary"
                if st.button(btn_label, key=f"sel_{d['ticker']}_{i}",
                             use_container_width=True, type=btn_type):
                    if is_sel:
                        sel_set.discard(d["ticker"])
                    else:
                        sel_set.add(d["ticker"])
                    st.session_state["discovery_selected"] = sel_set
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Panel de acción: llevar a optimizador ─────────
            sel_set = st.session_state.get("discovery_selected", set())
            if sel_set:
                st.markdown("---")
                section("ACTIVOS SELECCIONADOS")
                st.markdown(
                    " &nbsp; ".join(
                        f'<span style="background:rgba(10,132,255,0.15); border:1px solid '
                        f'rgba(10,132,255,0.4); border-radius:20px; padding:4px 14px; '
                        f'font-family:var(--mono); font-weight:700; color:#0a84ff;">{t}</span>'
                        for t in sorted(sel_set)
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                hdf_now   = holdings_df()
                held_now  = hdf_now["Ticker"].tolist() if not hdf_now.empty else []
                merged    = list(dict.fromkeys(held_now + list(sel_set)))

                st.markdown(
                    f"Al llevarlos al optimizador, el universo quedará: "
                    f"**{len(held_now)} actuales + {len(sel_set)} nuevos = {len(merged)} activos**"
                )

                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("🧠 Llevar a Optimización (Sub-tab 2)",
                                 type="primary", use_container_width=True,
                                 key="disc_to_opt"):
                        # Pre-cargar el campo de texto del sub-tab 2
                        st.session_state["disc_opt_extra"] = ", ".join(sorted(sel_set))
                        st.success(
                            f"Activos copiados. Ve al sub-tab **🧠 Optimización de Estrategia** "
                            f"y pégalos en el campo 'Agregar activos al universo':\n\n"
                            f"`{', '.join(sorted(sel_set))}`"
                        )
                        # Mostrar el texto a copiar con st.code para facilidad
                        st.code(", ".join(sorted(sel_set)), language=None)
                with cc2:
                    if st.button("🗑️ Limpiar selección", type="secondary",
                                 use_container_width=True, key="disc_clear"):
                        st.session_state["discovery_selected"] = set()
                        st.rerun()
        else:
            st.info(
                "Configura el análisis en el panel izquierdo y presiona "
                "**Analizar candidatos** para ver los resultados."
            )


def tab_performance() -> None:
    hdf = holdings_df()
    if hdf.empty:
        st.info("📂  Agrega posiciones en **Portfolio Editor** primero.")
        return

    bench   = st.session_state.get("benchmark", "SPY")
    rf_rate = st.session_state.get("rf_rate", 0.045)

    # Incluir TODOS los tickers que hayan aparecido en transacciones,
    # no solo los actuales. Si una acción fue vendida completamente,
    # sus precios históricos siguen siendo necesarios para calcular
    # el NAV de los períodos en que estuvo en el portafolio.
    _txns_for_tickers = transactions_df()
    _all_txn_tickers = (
        _txns_for_tickers["Ticker"].unique().tolist()
        if not _txns_for_tickers.empty else []
    )
    tickers = list(dict.fromkeys(_all_txn_tickers + hdf["Ticker"].unique().tolist()))

    # ── Fecha de inicio: primera transacción o fallback a 1 año ──
    txns = transactions_df()
    first_txn_date: date | None = None
    if not txns.empty:
        try:
            first_txn_date = pd.to_datetime(txns["Date"]).min().date()
        except Exception:
            first_txn_date = None

    end_dt   = datetime.today()
    default_start = first_txn_date if first_txn_date else (end_dt - timedelta(days=365)).date()

    # ── Restore saved custom date if available ───────────────
    _saved_custom = st.session_state.get("perf_custom_start")
    _use_custom   = st.session_state.get("perf_use_custom", False)
    if _saved_custom:
        try:
            _saved_custom = pd.to_datetime(_saved_custom).date()
        except Exception:
            _saved_custom = default_start

    col_ctrl, col_bench_extra, col_info = st.columns([1, 1, 2])
    with col_ctrl:
        period_mode = st.radio(
            "Período:",
            ["Desde primera compra", "Personalizado"],
            index=1 if _use_custom else 0,
            horizontal=True, key="perf_period_mode",
        )
    with col_bench_extra:
        extra_bench_raw = st.text_input(
            "Benchmarks extra (coma):", placeholder="QQQ, IWM, SOXX",
            key="extra_bench_input",
            help="Compara vs múltiples índices simultáneamente",
        )
        extra_benches = [t.strip().upper() for t in extra_bench_raw.split(",") if t.strip()]
    with col_info:
        if first_txn_date:
            days_since = (date.today() - first_txn_date).days
            st.markdown(
                f"<div style='font-family:var(--mono);font-size:0.82rem;color:#aeaeb2;padding-top:28px;'>"
                f"Primera compra: <b style='color:#ffffff'>{first_txn_date}</b>"
                f"&nbsp;·&nbsp;{days_since} días</div>",
                unsafe_allow_html=True,
            )

    if period_mode == "Personalizado":
        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        with col_d1:
            custom_start = st.date_input(
                "Desde:", value=_saved_custom or default_start, key="perf_start")
        with col_d2:
            custom_end = st.date_input("Hasta:", value=date.today(), key="perf_end")
        with col_d3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Guardar fecha como predeterminada",
                         use_container_width=True, key="save_perf_date",
                         help="Guarda esta fecha de inicio para este portafolio"):
                st.session_state["perf_custom_start"] = str(custom_start)
                st.session_state["perf_use_custom"]   = True
                st.success(f"✅ Fecha guardada: {custom_start}")
        start_dt = datetime.combine(custom_start, datetime.min.time())
        end_dt   = datetime.combine(custom_end,   datetime.min.time())
    else:
        start_dt = datetime.combine(default_start, datetime.min.time())
        # Clear saved custom if user switches to "desde primera compra"
        if _use_custom:
            st.session_state["perf_use_custom"] = False

    extra_benches = [t.strip().upper() for t in
                     st.session_state.get("extra_bench_input", "").split(",") if t.strip()]
    with st.spinner("Descargando histórico..."):
        all_bench_t = list(dict.fromkeys([bench] + extra_benches))
        prices_df = fetch_history(
            tickers + all_bench_t,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

    if prices_df.empty:
        st.error("No se pudieron obtener datos históricos.")
        return

    # ── Curva de equity del portafolio (basada en transacciones reales) ──
    # Usa el historial de compras/ventas para saber exactamente qué se
    # tenía en cada fecha — no los holdings actuales para todo el período.
    equity = build_portfolio_equity_from_transactions(txns, prices_df)

    if equity.empty:
        # Fallback: si no hay transacciones con fecha, usar holdings actuales
        equity = build_portfolio_equity(hdf, prices_df)

    if equity.empty:
        st.error("No hay datos suficientes para construir la curva de portafolio.")
        return

    # Normalizar a base 100 desde el primer día con datos
    base = equity.iloc[0]
    eq_norm = equity / base * 100

    bench_series = prices_df[bench].dropna() if bench in prices_df.columns else pd.Series(dtype=float)
    # Alinear benchmark al mismo punto de inicio que el portafolio
    if not bench_series.empty:
        bench_aligned = bench_series.reindex(equity.index).ffill().bfill()
        bench_norm = bench_aligned / bench_aligned.iloc[0] * 100
    else:
        bench_norm = pd.Series(dtype=float)

    # ── TWR: retorno limpio sin efecto de aportaciones ───────────
    # Usa Modified Dietz diario para eliminar los "saltos" por depósitos.
    # Todas las métricas (Sharpe, Max DD, retorno anual) usan esta serie.
    twr_series = build_twr_series(txns, prices_df)
    if twr_series.empty:
        twr_series = eq_norm   # fallback: NAV normalizado

    # Alinear TWR al índice del equity para que tengan la misma base temporal
    twr_aligned = twr_series.reindex(equity.index).ffill().bfill()

    # Retornos diarios limpios (sin efecto de depósitos)
    port_returns  = twr_aligned.pct_change().dropna()
    bench_returns = bench_series.pct_change().dropna() if not bench_series.empty else pd.Series(dtype=float)

    port_metrics  = compute_performance_metrics(port_returns, rf_rate)
    bench_metrics = compute_performance_metrics(bench_returns, rf_rate) if not bench_returns.empty else {}

    # ── KPIs comparativos ─────────────────────────────────────
    section("MÉTRICAS COMPARATIVAS")

    # Aviso si el período es corto (anualización poco confiable)
    n_days_port  = port_metrics.get("N Días", 0)
    ann_est_port = port_metrics.get("Retorno Anual Est", False)
    if ann_est_port:
        st.warning(
            f"⚠️ **Período corto ({n_days_port} días de datos — menos de 1 año).** "
            f"El Retorno Anualizado es una **proyección matemática**, no un histórico real. "
            f"Usa el **Retorno del Período** como referencia principal."
        )

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"**Tu Portafolio** *(TWR — sin efecto de aportaciones)*")
        k_cols = st.columns(3)
        # Retorno del período como métrica principal (sin anualizar, siempre real)
        port_period = port_metrics.get("Retorno Período", 0)
        port_ann    = port_metrics.get("Retorno Anualizado", 0)
        ann_label   = "Retorno Anual*" if ann_est_port else "Retorno Anual"
        with k_cols[0]:
            color = "#30d158" if port_period >= 0 else "#ff453a"
            kpi_card("Retorno Período", f"{port_period:.1%}", accent=color)
        with k_cols[1]:
            color = "#30d158" if port_metrics.get("Sharpe Ratio", 0) > 0 else "#ff453a"
            kpi_card("Sharpe", f"{port_metrics.get('Sharpe Ratio', 0):.2f}", accent=color)
        with k_cols[2]:
            color = "#ff453a" if port_metrics.get("Max Drawdown", 0) < 0 else "#30d158"
            kpi_card("Max DD", f"{port_metrics.get('Max Drawdown', 0):.1%}", accent=color)
        if ann_est_port:
            st.caption(f"\\* Retorno anualizado proyectado: **{port_ann:.1%}** "
                       f"(basado en {n_days_port} días — dato indicativo)")
        else:
            st.caption(f"Retorno anualizado: **{port_ann:.1%}**")

    with mc2:
        st.markdown(f"**{bench} (Benchmark)**")
        k_cols2 = st.columns(3)
        bench_period = bench_metrics.get("Retorno Período", 0)
        bench_ann    = bench_metrics.get("Retorno Anualizado", 0)
        bench_est    = bench_metrics.get("Retorno Anual Est", False)
        with k_cols2[0]:
            color = "#30d158" if bench_period >= 0 else "#ff453a"
            kpi_card("Retorno Período", f"{bench_period:.1%}", accent=color)
        with k_cols2[1]:
            color = "#30d158" if bench_metrics.get("Sharpe Ratio", 0) > 0 else "#ff453a"
            kpi_card("Sharpe", f"{bench_metrics.get('Sharpe Ratio', 0):.2f}", accent=color)
        with k_cols2[2]:
            color = "#ff453a" if bench_metrics.get("Max Drawdown", 0) < 0 else "#30d158"
            kpi_card("Max DD", f"{bench_metrics.get('Max Drawdown', 0):.1%}", accent=color)
        st.caption(f"Retorno anualizado: **{bench_ann:.1%}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico de performance ────────────────────────────────
    section("RENDIMIENTO ACUMULADO (BASE 100)")
    st.caption(
        "📐 **TWR (Time-Weighted Return):** la curva azul muestra rendimiento "
        "puro — las aportaciones mensuales de capital **no** inflan la línea. "
        "Activa 'NAV Real' en la leyenda para ver el valor bruto del portafolio."
    )
    fig_perf = go.Figure()
    # Curva principal: TWR (suave, sin saltos por aportaciones)
    fig_perf.add_trace(go.Scatter(
        x=twr_aligned.index, y=twr_aligned.values,
        name="Tu Portafolio (TWR)",
        line=dict(color="#0a84ff", width=2.5),
        hovertemplate="%{y:.1f}<extra>Portafolio TWR</extra>",
    ))
    # Curva secundaria: NAV real (visible solo si el usuario la activa)
    fig_perf.add_trace(go.Scatter(
        x=eq_norm.index, y=eq_norm.values,
        name="NAV Real (con aportaciones)",
        line=dict(color="#0a84ff", width=1.2, dash="dot"),
        opacity=0.45,
        visible="legendonly",   # oculta por defecto, activar desde la leyenda
        hovertemplate="%{y:.1f}<extra>NAV Real</extra>",
    ))
    if not bench_norm.empty:
        fig_perf.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            name=bench,
            line=dict(color="#8e8e93", width=1.5, dash="dash"),
            hovertemplate="%{y:.1f}<extra>" + bench + "</extra>",
        ))
    # Extra benchmarks
    _extra_colors = ["#ffd60a","#a78bfa","#fb923c","#38bdf8","#f472b6"]
    for _ei, _eb in enumerate(extra_benches):
        if _eb in prices_df.columns:
            _eb_s = prices_df[_eb].dropna()
            if not _eb_s.empty:
                _eb_aligned = _eb_s.reindex(equity.index).ffill().bfill()
                _eb_norm    = _eb_aligned / _eb_aligned.iloc[0] * 100
                fig_perf.add_trace(go.Scatter(
                    x=_eb_norm.index, y=_eb_norm.values, name=_eb,
                    line=dict(color=_extra_colors[_ei % len(_extra_colors)],
                              width=1.5, dash="dashdot"),
                    hovertemplate="%{y:.1f}<extra>" + _eb + "</extra>",
                ))
    perf_layout = {**PLOTLY_LAYOUT}
    perf_layout["xaxis"] = {
        **PLOTLY_LAYOUT["xaxis"],
        "rangeselector": dict(
            bgcolor="#16161f",
            font_color="#aeaeb2",
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(step="all", label="MAX"),
            ],
        ),
    }
    fig_perf.update_layout(**perf_layout, yaxis_title="Valor (Base 100)", height=420)
    st.plotly_chart(fig_perf, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Drawdown ──────────────────────────────────────────────
    section("UNDERWATER — CAÍDAS DESDE MÁXIMOS")
    cum  = (1 + port_returns).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak * 100

    bench_dd_series = pd.Series(dtype=float)
    if not bench_returns.empty:
        cum_b  = (1 + bench_returns).cumprod()
        peak_b = cum_b.cummax()
        bench_dd_series = (cum_b - peak_b) / peak_b * 100

    max_dd_val  = float(dd.min())
    max_dd_date = dd.idxmin()
    curr_dd_val = float(dd.iloc[-1]) if len(dd) else 0.0

    fig_dd = go.Figure()
    # Severity zones
    fig_dd.add_hrect(y0=-10, y1=0,   fillcolor="rgba(255,69,58,0.04)", line_width=0)
    fig_dd.add_hrect(y0=-20, y1=-10, fillcolor="rgba(255,69,58,0.08)", line_width=0)
    fig_dd.add_hrect(y0=-100,y1=-20, fillcolor="rgba(255,69,58,0.13)", line_width=0)
    # Severity labels
    for lvl, lbl in [(-5,"Leve"),(-15,"Moderado"),(-30,"Severo")]:
        fig_dd.add_annotation(x=dd.index[min(5,len(dd)-1)], y=lvl,
            text=lbl, showarrow=False,
            font=dict(size=9, color="rgba(255,69,58,0.4)", family="DM Mono"),
            xanchor="left")
    # Benchmark line
    if not bench_dd_series.empty:
        fig_dd.add_trace(go.Scatter(
            x=bench_dd_series.index, y=bench_dd_series.values,
            name=bench, line=dict(color="#8e8e93", width=1.2, dash="dot"),
            hovertemplate="%{y:.1f}%<extra>" + bench + "</extra>",
        ))
    # Portfolio fill
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", name="Portafolio",
        line=dict(color="#ff453a", width=2),
        fillcolor="rgba(255,69,58,0.18)",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))
    # Max DD marker
    if max_dd_date is not None:
        fig_dd.add_annotation(
            x=max_dd_date, y=max_dd_val,
            text=f"Máx DD<br>{max_dd_val:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor="#ff453a", arrowsize=0.8,
            font=dict(size=10, color="#ff453a", family="DM Mono"),
            bgcolor="rgba(10,10,15,0.85)", bordercolor="#ff453a",
            borderwidth=1, borderpad=4, ax=30, ay=-30,
        )
    # Rango del eje Y: 50% de margen bajo el max DD real (mínimo -15% para que no se vea flat)
    _dd_floor  = min(max_dd_val * 1.5, -15.0)
    # Si también el benchmark tiene drawdowns más profundos, incluirlos
    if not bench_dd_series.empty:
        _dd_floor = min(_dd_floor, float(bench_dd_series.min()) * 1.3)
    _dd_floor  = min(_dd_floor, -5.0)   # nunca menos de -5% de margen visual

    fig_dd.update_layout(**PLOTLY_LAYOUT, height=320)
    fig_dd.update_yaxes(title_text="Caída desde máximo (%)",
                        range=[_dd_floor, 0.5],
                        ticksuffix="%",
                        showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig_dd.update_xaxes(showgrid=False)
    st.plotly_chart(fig_dd, use_container_width=True, config=PLOTLY_CONFIG)

    # Stats row
    _in_dd = curr_dd_val < -0.5
    _dd_clr = "#ff453a" if _in_dd else "#30d158"
    st.markdown(
        f"<div style='display:flex;gap:32px;font-family:DM Mono,monospace;"
        f"font-size:0.82rem;color:#8e8e93;margin-top:-8px;margin-bottom:16px;'>"
        f"<span>Máx Drawdown: <b style='color:#ff453a;'>{max_dd_val:.2f}%</b></span>"
        f"<span>DD actual: <b style='color:{_dd_clr};'>{curr_dd_val:.2f}%</b></span>"
        f"<span>{'⚠️ En drawdown' if _in_dd else '✓ En máximos'}</span>"
        f"</div>", unsafe_allow_html=True)

    # ── Tabla completa de métricas ────────────────────────────
    section("MÉTRICAS COMPLETAS")
    _ann_note = " *" if ann_est_port else ""
    all_metric_labels = [
        ("Retorno Período",              ".1%"),   # siempre real, sin anualizar
        (f"Retorno Anualizado{_ann_note}",".1%"),  # puede ser proyección
        ("Volatilidad Anual",            ".1%"),
        ("Sharpe Ratio",                 ".2f"),
        ("Sortino Ratio",                ".2f"),
        ("Calmar Ratio",                 ".2f"),
        ("Max Drawdown",                 ".1%"),
        ("Win Rate",                     ".1%"),
        ("VaR 95%",                      ".2%"),
        ("CVaR 95%",                     ".2%"),
    ]
    # Mapear nombre de display → clave real en el dict
    _label_key = {
        f"Retorno Anualizado{_ann_note}": "Retorno Anualizado",
    }
    rows = []
    for label, fmt in all_metric_labels:
        key = _label_key.get(label, label)
        pv   = port_metrics.get(key, 0)
        bv   = bench_metrics.get(key, 0)
        diff = pv - bv
        rows.append({
            "Métrica":    label,
            "Portafolio": f"{pv:{fmt}}",
            bench:        f"{bv:{fmt}}",
            "Diferencia": diff,
            "_pv":        pv,
            "_bv":        bv,
        })
    if ann_est_port:
        rows_note = [{"Métrica": f"* Proyección basada en {n_days_port} días (<1 año)",
                      "Portafolio": "", bench: "", "Diferencia": float("nan"),
                      "_pv": 0, "_bv": 0}]
        rows += rows_note

    df_metrics = pd.DataFrame(rows)
    st.dataframe(
        df_metrics.drop(columns=["_pv", "_bv"]),
        column_config={
            "Diferencia": st.column_config.NumberColumn("Diferencia", format="%+.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # ── Retornos mensuales (heatmap) ──────────────────────────
    section("HEATMAP DE RETORNOS MENSUALES")
    monthly = (port_returns + 1).resample("ME").prod() - 1
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Ret":   monthly.values,
    })
    if not monthly_df.empty:
        pivot = monthly_df.pivot(index="Year", columns="Month", values="Ret")
        month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        fig_heat = px.imshow(
            pivot,
            text_auto=".1%",
            color_continuous_scale="RdYlGn",
            zmin=-0.1, zmax=0.1,
            aspect="auto",
        )
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            height=max(200, len(pivot) * 45 + 80),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)


# ─────────────────────────────────────────────────────────────
# TAB 5: ANÁLISIS POR ACTIVO
# ─────────────────────────────────────────────────────────────

def tab_analytics() -> None:
    hdf = holdings_df()
    if hdf.empty:
        st.info("📂  Agrega posiciones en **Portfolio Editor** primero.")
        return

    bench   = st.session_state.get("benchmark", "SPY")
    tickers = hdf["Ticker"].unique().tolist()

    col_c, _ = st.columns([1, 3])
    with col_c:
        n_years = st.selectbox("Período:", [1, 2, 3, 5], index=1,
                               format_func=lambda y: f"{y} año{'s' if y > 1 else ''}",
                               key="analytics_years")

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=365 * n_years)

    with st.spinner("Descargando histórico..."):
        prices_df = fetch_history(
            tickers + [bench],
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
        )

    if prices_df.empty:
        st.error("Sin datos.")
        return

    returns_df = prices_df.pct_change().dropna(how="all")

    # ── Métricas por activo ───────────────────────────────────
    section("MÉTRICAS POR ACTIVO")
    rows = []
    for t in tickers:
        if t not in returns_df.columns:
            continue
        m = compute_performance_metrics(returns_df[t].dropna(), st.session_state.get("rf_rate", 0.045))
        if not m:
            continue

        # Beta vs benchmark
        beta = np.nan
        if bench in returns_df.columns:
            aligned = returns_df[[t, bench]].dropna()
            if len(aligned) > 30:
                cov_mat = aligned.cov()
                var_b   = aligned[bench].var()
                beta    = cov_mat.loc[t, bench] / var_b if var_b > 0 else np.nan

        rows.append({
            "Ticker":        t,
            "Ret. Anual":    m.get("Retorno Anualizado", 0),
            "Volatilidad":   m.get("Volatilidad Anual", 0),
            "Sharpe":        m.get("Sharpe Ratio", 0),
            "Max DD":        m.get("Max Drawdown", 0),
            "Beta":          beta,
            "Win Rate":      m.get("Win Rate", 0),
        })

    if rows:
        df_a = pd.DataFrame(rows)
        st.dataframe(
            df_a,
            column_config={
                "Ticker":      st.column_config.TextColumn("Ticker", width="small"),
                "Ret. Anual":  st.column_config.NumberColumn("Ret. Anual", format="%.1f%%"),
                "Volatilidad": st.column_config.NumberColumn("Volatilidad", format="%.1f%%"),
                "Sharpe":      st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "Max DD":      st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                "Beta":        st.column_config.NumberColumn("Beta", format="%.2f"),
                "Win Rate":    st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ── Rolling Sharpe del portafolio ────────────────────────
    section("ROLLING SHARPE — PORTAFOLIO")
    _rf_daily = st.session_state.get("rf_rate", 0.09) / 252
    _prices_live_an = fetch_live_prices(tickers)
    _nav_an = sum(
        hdf.loc[hdf["Ticker"] == t, "Shares"].sum()
        * _prices_live_an.get(t, {}).get("price", 0)
        for t in tickers if t in returns_df.columns
    )
    _port_r = pd.Series(dtype=float)  # default vacío
    if _nav_an > 0:
        _w_an = {
            t: hdf.loc[hdf["Ticker"] == t, "Shares"].sum()
               * _prices_live_an.get(t, {}).get("price", 0) / _nav_an
            for t in tickers if t in returns_df.columns
        }
        _port_r = sum(returns_df[t].fillna(0) * w for t, w in _w_an.items())
        _sharpe_fig = go.Figure()
        _sharpe_colors  = {"30D": "#0a84ff", "60D": "#30d158", "90D": "#ffd60a"}
        _sharpe_windows = {"30D": 30, "60D": 60, "90D": 90}
        _all_sharpe_vals = []
        for _lbl, _win in _sharpe_windows.items():
            if len(_port_r) < _win + 5:
                continue
            _rs = _port_r.rolling(_win).apply(
                lambda x: (x.mean() - _rf_daily) / (x.std() + 1e-9) * np.sqrt(252),
                raw=True,
            ).dropna()
            _all_sharpe_vals.extend(_rs.values.tolist())
            _sharpe_fig.add_trace(go.Scatter(
                x=_rs.index, y=_rs.values, name=_lbl,
                line=dict(color=_sharpe_colors[_lbl], width=1.8),
                hovertemplate=f"<b>{_lbl}</b>: %{{y:.2f}}<extra></extra>",
            ))

        # Rango Y basado en datos reales + padding generoso
        if _all_sharpe_vals:
            _sh_min = min(_all_sharpe_vals)
            _sh_max = max(_all_sharpe_vals)
            _sh_pad = max((_sh_max - _sh_min) * 0.20, 0.5)
            _sh_lo  = _sh_min - _sh_pad
            _sh_hi  = _sh_max + _sh_pad
            # Asegurar que 0 y 1 siempre queden visibles
            _sh_lo  = min(_sh_lo, -0.5)
            _sh_hi  = max(_sh_hi,  1.5)
        else:
            _sh_lo, _sh_hi = -1, 2

        # Zonas con add_shape (respeta el rango del eje, no lo fuerza)
        _sharpe_fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=_sh_hi,
            xref="paper", yref="y",
            fillcolor="rgba(48,209,88,0.04)", line_width=0)
        _sharpe_fig.add_shape(type="rect", x0=0, x1=1, y0=_sh_lo, y1=0,
            xref="paper", yref="y",
            fillcolor="rgba(255,69,58,0.04)", line_width=0)

        _sharpe_fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dash")
        _sharpe_fig.add_hline(y=1, line_color="rgba(48,209,88,0.35)", line_dash="dot",
                              annotation_text="Sharpe = 1",
                              annotation_font=dict(size=9, color="#30d158", family="DM Mono"),
                              annotation_position="bottom right")

        _sharpe_fig.update_layout(**_pl(), height=300,
                                  paper_bgcolor="#000000", plot_bgcolor="#000000")
        _sharpe_fig.update_yaxes(title_text="Sharpe (rolling)", range=[_sh_lo, _sh_hi],
                                 showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                 tickfont=dict(size=9, color="#636366", family="DM Mono"))
        _sharpe_fig.update_xaxes(showgrid=False,
                                 tickfont=dict(size=9, color="#636366", family="DM Mono"))
        st.plotly_chart(_sharpe_fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.info("Sin precios live para calcular rolling Sharpe.")

    # ── RSI + Bandas de Bollinger por acción ──────────────────
    section("RSI + BANDAS DE BOLLINGER")
    _bb_tickers = [t for t in tickers if t in prices_df.columns]
    if _bb_tickers:
        from plotly.subplots import make_subplots as _make_subplots
        _col_bb, _ = st.columns([1, 3])
        with _col_bb:
            _sel = st.selectbox("Acción:", _bb_tickers, key="bb_ticker_sel")
        _ps = prices_df[_sel].dropna()
        if len(_ps) >= 20:
            _sma20 = _ps.rolling(20).mean()
            _std20 = _ps.rolling(20).std()
            _bb_up = _sma20 + 2 * _std20
            _bb_lo = _sma20 - 2 * _std20
            _delta = _ps.diff()
            _gain  = _delta.clip(lower=0).rolling(14).mean()
            _loss  = (-_delta.clip(upper=0)).rolling(14).mean()
            _rsi14 = 100 - 100 / (1 + _gain / (_loss + 1e-9))

            _fig_bb = _make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                                     shared_xaxes=True, vertical_spacing=0.04)
            # Bollinger fill
            _fig_bb.add_trace(go.Scatter(
                x=_bb_up.index, y=_bb_up.values, name="Banda Sup.",
                line=dict(color="rgba(10,132,255,0.35)", width=1, dash="dot"),
                showlegend=False), row=1, col=1)
            _fig_bb.add_trace(go.Scatter(
                x=_bb_lo.index, y=_bb_lo.values, name="Banda Inf.",
                fill="tonexty", fillcolor="rgba(10,132,255,0.07)",
                line=dict(color="rgba(10,132,255,0.35)", width=1, dash="dot"),
                showlegend=False), row=1, col=1)
            _fig_bb.add_trace(go.Scatter(
                x=_sma20.index, y=_sma20.values, name="SMA 20",
                line=dict(color="rgba(255,214,10,0.7)", width=1.5, dash="dash")),
                row=1, col=1)
            _fig_bb.add_trace(go.Scatter(
                x=_ps.index, y=_ps.values, name=_sel,
                line=dict(color="#0a84ff", width=2),
                hovertemplate=f"<b>{_sel}</b> $%{{y:,.2f}}<extra></extra>"),
                row=1, col=1)
            # RSI — usar add_shape con yref="y2" (más fiable en subplots que add_hline)
            for _rlvl, _rclr, _rtxt, _ranch in [
                (70, "#ff453a", "Sobrecompra", "bottom"),
                (30, "#30d158", "Sobreventa",  "top"),
            ]:
                _fig_bb.add_shape(
                    type="line", x0=0, x1=1, y0=_rlvl, y1=_rlvl,
                    xref="paper", yref="y2",
                    line=dict(color=_rclr, width=1.2, dash="dot"),
                )
                _fig_bb.add_annotation(
                    x=0.01, y=_rlvl, xref="paper", yref="y2",
                    text=_rtxt, showarrow=False,
                    font=dict(size=9, color=_rclr, family="DM Mono"),
                    xanchor="left", yanchor=_ranch,
                )
            # Zonas de fondo para RSI
            _fig_bb.add_shape(type="rect", x0=0, x1=1, y0=70, y1=100,
                xref="paper", yref="y2",
                fillcolor="rgba(255,69,58,0.07)", line_width=0)
            _fig_bb.add_shape(type="rect", x0=0, x1=1, y0=0, y1=30,
                xref="paper", yref="y2",
                fillcolor="rgba(48,209,88,0.07)", line_width=0)
            _fig_bb.add_trace(go.Scatter(
                x=_rsi14.index, y=_rsi14.values, name="RSI 14",
                line=dict(color="#a78bfa", width=1.8),
                hovertemplate="RSI: %{y:.1f}<extra></extra>"),
                row=2, col=1)
            _fig_bb.update_layout(**PLOTLY_LAYOUT, height=520)
            _fig_bb.update_layout(legend=dict(orientation="h", y=1.04, x=0,
                                              font=dict(size=10, color="#8e8e93"),
                                              bgcolor="rgba(0,0,0,0)"))
            _fig_bb.update_yaxes(title_text=f"Precio {_sel}", row=1, col=1,
                                  showgrid=True, gridcolor="rgba(255,255,255,0.04)")
            _fig_bb.update_yaxes(title_text="RSI", row=2, col=1,
                                  range=[0,100], showgrid=False)
            _fig_bb.update_xaxes(showgrid=False)
            st.plotly_chart(_fig_bb, use_container_width=True, config=PLOTLY_CONFIG)

            # RSI actual
            _rsi_now = float(_rsi14.dropna().iloc[-1]) if not _rsi14.dropna().empty else None
            if _rsi_now is not None:
                _rc = "#ff453a" if _rsi_now>70 else ("#30d158" if _rsi_now<30 else "#aeaeb2")
                _rl = "Sobrecompra ↑" if _rsi_now>70 else ("Sobreventa ↓" if _rsi_now<30 else "Neutral")
                st.markdown(
                    f"<div style='font-family:DM Mono,monospace;font-size:0.82rem;"
                    f"color:#8e8e93;margin-top:-8px;margin-bottom:12px;'>"
                    f"RSI actual <b style='color:{_rc};'>{_rsi_now:.1f}</b>"
                    f" — <span style='color:{_rc};'>{_rl}</span></div>",
                    unsafe_allow_html=True)

    # ── Correlaciones ─────────────────────────────────────────
    section("MATRIZ DE CORRELACIÓN")
    corr_cols = [t for t in tickers if t in returns_df.columns]
    if len(corr_cols) >= 2:
        corr = returns_df[corr_cols].corr()
        _n = len(corr_cols)
        _z = corr.values.tolist()
        _txt = [[f"{corr.iloc[i,j]:.2f}" for j in range(_n)] for i in range(_n)]
        _tclr = [["#ffffff" if abs(corr.iloc[i,j]) > 0.3 else "#aeaeb2"
                  for j in range(_n)] for i in range(_n)]
        fig_corr = go.Figure(go.Heatmap(
            z=_z,
            x=corr_cols,
            y=corr_cols,
            text=_txt,
            texttemplate="%{text}",
            textfont=dict(size=11, family="DM Mono"),
            colorscale=[
                [0.00, "#dc2626"],
                [0.20, "#b91c1c"],
                [0.40, "#7f1d1d"],
                [0.50, "#27272a"],
                [0.60, "#14532d"],
                [0.80, "#15803d"],
                [1.00, "#4ade80"],
            ],
            zmin=-1, zmax=1,
            hovertemplate="<b>%{x} — %{y}</b><br>Correlación: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Correlación", font=dict(size=10, color="#8e8e93"),
                           side="right"),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                tickfont=dict(size=9, color="#8e8e93", family="DM Mono"),
                thickness=12,
                len=0.9,
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        ))
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=max(340, _n * 56 + 100))
        fig_corr.update_xaxes(tickfont=dict(size=11, color="#aeaeb2", family="DM Mono"),
                              side="bottom", showgrid=False)
        fig_corr.update_yaxes(tickfont=dict(size=11, color="#aeaeb2", family="DM Mono"),
                              showgrid=False, autorange="reversed")
        st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

        # Auto-insight: par más/menos correlacionado
        _pairs = [(corr_cols[i], corr_cols[j], float(corr.iloc[i,j]))
                  for i in range(_n) for j in range(i+1, _n)]
        if _pairs:
            _top  = max(_pairs, key=lambda x: abs(x[2]))
            _div  = min(_pairs, key=lambda x: x[2])
            st.markdown(
                f"<div style='font-size:0.76rem;color:#8e8e93;font-family:DM Mono,"
                f"monospace;margin-top:-10px;margin-bottom:16px;line-height:1.7;'>"
                f"Par más correlacionado: "
                f"<span style='color:#0a84ff;font-weight:700;'>{_top[0]} / {_top[1]}</span>"
                f" ({_top[2]:+.2f})&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"Mayor diversificador: "
                f"<span style='color:#30d158;font-weight:700;'>{_div[0]} / {_div[1]}</span>"
                f" ({_div[2]:+.2f})"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── Retornos normalizados ─────────────────────────────────
    section("RENDIMIENTO COMPARADO (BASE 100)")
    fig_norm = go.Figure()
    colors_list = px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    for i, t in enumerate(corr_cols):
        s = prices_df[t].dropna()
        if s.empty:
            continue
        norm = s / s.iloc[0] * 100
        fig_norm.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=t,
            line=dict(color=colors_list[i % len(colors_list)], width=2),
            hovertemplate=f"<b>{t}</b>: %{{y:.1f}}<extra></extra>",
        ))
    fig_norm.update_layout(**PLOTLY_LAYOUT, yaxis_title="Base 100", height=400)
    st.plotly_chart(fig_norm, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Contribución al riesgo ────────────────────────────────
    section("CONTRIBUCIÓN AL RIESGO DEL PORTAFOLIO")
    tickers_with_data = [t for t in tickers if t in returns_df.columns]
    if len(tickers_with_data) >= 2:
        prices_live = fetch_live_prices(tickers_with_data)
        nav = sum(
            hdf.loc[hdf["Ticker"] == t, "Shares"].sum() * prices_live.get(t, {}).get("price", 0)
            for t in tickers_with_data
        )
        weights_arr = np.array([
            hdf.loc[hdf["Ticker"] == t, "Shares"].sum() * prices_live.get(t, {}).get("price", 0) / nav
            if nav > 0 else 1 / len(tickers_with_data)
            for t in tickers_with_data
        ])

        cov = returns_df[tickers_with_data].cov().values * TRADING_DAYS
        port_vol = np.sqrt(weights_arr @ cov @ weights_arr)
        if port_vol > 0:
            marginal_rc = (cov @ weights_arr) / port_vol
            risk_contrib = weights_arr * marginal_rc / port_vol  # % del riesgo total

            df_rc = pd.DataFrame({
                "Ticker":          tickers_with_data,
                "Peso":            weights_arr * 100,
                "Contribución %":  risk_contrib * 100,
            }).sort_values("Contribución %", ascending=False)

            fig_rc = go.Figure()
            fig_rc.add_trace(go.Bar(
                x=df_rc["Ticker"], y=df_rc["Peso"],
                name="Peso Actual", marker_color="#0a84ff",
            ))
            fig_rc.add_trace(go.Bar(
                x=df_rc["Ticker"], y=df_rc["Contribución %"],
                name="Aporte al Riesgo", marker_color="#ff453a",
            ))
            fig_rc.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                yaxis_title="% del Total",
                height=340,
                title=f"Volatilidad del portafolio: {port_vol*100:.1f}% anual",
            )
            st.plotly_chart(fig_rc, use_container_width=True, config=PLOTLY_CONFIG)

    # ── DCA Calculator ────────────────────────────────────────
    section("DCA CALCULATOR — APORTACIÓN PERIÓDICA")
    st.caption("Simula el efecto de agregar capital fijo de forma periódica a tu portafolio actual.")
    _dca_c1, _dca_c2, _dca_c3 = st.columns(3)
    with _dca_c1:
        _dca_amt = st.number_input("Aportación por período ($)", min_value=0.0,
                                   value=500.0, step=100.0, key="dca_amount")
    with _dca_c2:
        _dca_freq = st.selectbox("Frecuencia", ["Mensual","Quincenal","Semanal"],
                                 key="dca_freq")
    with _dca_c3:
        _dca_years = st.slider("Horizonte (años)", 1, 20, 5, key="dca_years")

    if _dca_amt > 0:
        _freq_map = {"Mensual": 12, "Quincenal": 26, "Semanal": 52}
        _periods_per_year = _freq_map[_dca_freq]
        _total_periods = _dca_years * _periods_per_year

        # Usar retorno anualizado histórico del portafolio si está disponible
        if not _port_r.empty and len(_port_r) > 20:
            _port_ann_ret = float((1 + _port_r.mean()) ** 252 - 1)
            _port_vol_ann = float(_port_r.std() * np.sqrt(252))
        else:
            _port_ann_ret = 0.08
            _port_vol_ann = 0.15
        _r_per_period = _port_ann_ret / _periods_per_year
        _vol_per_period = _port_vol_ann / np.sqrt(_periods_per_year)

        # Simulación Monte Carlo (200 paths)
        np.random.seed(42)
        _nav_now = sum(
            hdf.loc[hdf["Ticker"] == t, "Shares"].sum()
            * _prices_live_an.get(t, {}).get("price", 0)
            for t in tickers
        ) if _nav_an > 0 else 0

        _n_paths = 200
        _paths = np.zeros((_n_paths, _total_periods + 1))
        _paths[:, 0] = _nav_now
        for _i in range(1, _total_periods + 1):
            _shocks = np.random.normal(_r_per_period, _vol_per_period, _n_paths)
            _paths[:, _i] = _paths[:, _i-1] * (1 + _shocks) + _dca_amt

        _invested = _nav_now + _dca_amt * np.arange(_total_periods + 1)
        _p10  = np.percentile(_paths, 10, axis=0)
        _p50  = np.percentile(_paths, 50, axis=0)
        _p90  = np.percentile(_paths, 90, axis=0)
        _t_ax = np.arange(_total_periods + 1) / _periods_per_year

        _fig_dca = go.Figure()
        _fig_dca.add_trace(go.Scatter(
            x=_t_ax, y=_p90, name="Optimista (P90)",
            line=dict(color="rgba(48,209,88,0.4)", width=1, dash="dot"),
            showlegend=True))
        _fig_dca.add_trace(go.Scatter(
            x=_t_ax, y=_p10, name="Pesimista (P10)",
            fill="tonexty", fillcolor="rgba(10,132,255,0.07)",
            line=dict(color="rgba(255,69,58,0.4)", width=1, dash="dot"),
            showlegend=True))
        _fig_dca.add_trace(go.Scatter(
            x=_t_ax, y=_p50, name="Mediana (P50)",
            line=dict(color="#0a84ff", width=2.5)))
        _fig_dca.add_trace(go.Scatter(
            x=_t_ax, y=_invested, name="Capital aportado",
            line=dict(color="#8e8e93", width=1.5, dash="dash")))
        _fig_dca.update_layout(
            **PLOTLY_LAYOUT, height=360,
            xaxis_title="Años", yaxis_title="Valor portafolio ($)",
            yaxis_tickprefix="$",
        )
        st.plotly_chart(_fig_dca, use_container_width=True, config=PLOTLY_CONFIG)

        _v50_f = _p50[-1]
        _v10_f = _p10[-1]
        _v90_f = _p90[-1]
        _tot_in = _nav_now + _dca_amt * _total_periods
        st.markdown(
            f"<div style='display:flex;gap:28px;font-family:DM Mono,monospace;"
            f"font-size:0.82rem;color:#8e8e93;margin-top:-8px;'>"
            f"<span>Capital total aportado: <b style='color:#ffffff;'>${_tot_in:,.0f}</b></span>"
            f"<span>Mediana a {_dca_years}A: <b style='color:#0a84ff;'>${_v50_f:,.0f}</b></span>"
            f"<span>Rango P10–P90: <b style='color:#30d158;'>${_v10_f:,.0f} – ${_v90_f:,.0f}</b></span>"
            f"</div>", unsafe_allow_html=True)

    # ── Noticias por ticker ───────────────────────────────────
    section("NOTICIAS RECIENTES")
    st.caption("Headlines de Yahoo Finance para tus posiciones actuales.")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _fetch_news(ticker_list: tuple) -> dict:
        news_map = {}
        for _nt in ticker_list:
            try:
                _tk = yf.Ticker(_nt)
                _nws = _tk.news or []
                news_map[_nt] = _nws[:4]
            except Exception:
                news_map[_nt] = []
        return news_map

    _news_data = _fetch_news(tuple(tickers))
    _news_cols = st.columns(min(3, len(tickers)))
    for _ni, _nt in enumerate(tickers):
        with _news_cols[_ni % len(_news_cols)]:
            st.markdown(
                f"<div style='font-family:DM Mono,monospace;font-weight:800;"
                f"font-size:0.82rem;color:#0a84ff;margin-bottom:8px;"
                f"padding:4px 10px;background:rgba(10,132,255,.08);"
                f"border-radius:6px;display:inline-block;'>{_nt}</div>",
                unsafe_allow_html=True)
            _arts = _news_data.get(_nt, [])
            if _arts:
                for _art in _arts:
                    # yfinance >=0.2.50 anida los datos en "content"
                    _c = _art.get("content", _art)
                    _title = (_c.get("title") or _art.get("title") or "Sin título")[:80]
                    _link  = (_c.get("canonicalUrl", {}).get("url")
                              or _art.get("link") or "#")
                    _pub   = (_c.get("provider", {}).get("displayName")
                              or _art.get("publisher") or "")
                    _ts    = _art.get("providerPublishTime", 0)
                    _pd_raw = _c.get("pubDate", "")
                    try:
                        if _pd_raw:
                            _fecha = _pd_raw[:10]
                        elif _ts:
                            _fecha = datetime.fromtimestamp(_ts).strftime("%d %b")
                        else:
                            _fecha = ""
                    except Exception:
                        _fecha = ""
                    st.markdown(
                        f"<div style='margin-bottom:10px;padding:8px 10px;"
                        f"background:rgba(255,255,255,0.025);border-radius:8px;"
                        f"border:1px solid rgba(255,255,255,0.05);'>"
                        f"<a href='{_link}' target='_blank' style='color:#ffffff;"
                        f"font-size:0.78rem;text-decoration:none;line-height:1.4;"
                        f"display:block;'>{_title}</a>"
                        f"<div style='font-size:0.65rem;color:#636366;"
                        f"font-family:DM Mono,monospace;margin-top:4px;'>"
                        f"{_pub} · {_fecha}</div>"
                        f"</div>",
                        unsafe_allow_html=True)
            else:
                st.caption("Sin noticias disponibles.")

    # ── Stress Test ───────────────────────────────────────────
    section("STRESS TEST — ESCENARIOS DE CAÍDA")
    st.caption("Simula el impacto en tu portafolio ante distintos choques de mercado.")

    _stress_presets = {
        "Corrección leve  –10%": -0.10,
        "Corrección media –20%": -0.20,
        "Bear market      –40%": -0.40,
        "Crash severo     –60%": -0.60,
    }
    _sc1, _sc2, _sc3 = st.columns([3, 1, 1])
    with _sc1:
        _st_preset = st.select_slider(
            "Escenario:", options=list(_stress_presets.keys()),
            value="Corrección media –20%", key="stress_preset",
        )
    with _sc2:
        _st_custom = st.number_input(
            "% personalizado:", min_value=-95, max_value=-1,
            value=-20, step=5, key="stress_custom_pct",
        )
    with _sc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _use_custom_st = st.toggle("Usar %", key="stress_use_custom")

    _shock_pct = (_st_custom / 100) if _use_custom_st else _stress_presets[_st_preset]
    _beta_adj  = st.toggle("Ajustar por Beta (más realista)", key="stress_beta_adj", value=True)

    # Obtener precios actuales y calcular impacto
    _st_prices = fetch_live_prices(tuple(sorted(tickers)))
    _st_rows = []
    _st_curr_total = 0.0
    _st_stress_total = 0.0

    for _, _row in hdf.iterrows():
        _tk = _row["Ticker"]
        _sh = float(_row.get("Shares", 0))
        _px = _st_prices.get(_tk, {}).get("price", 0)
        if _px <= 0 or _sh <= 0:
            continue
        _curr_val = _sh * _px
        # Beta ajustado (usa info de yfinance, fallback a 1.0)
        _beta = 1.0
        if _beta_adj:
            try:
                _beta = float(yf.Ticker(_tk).info.get("beta") or 1.0)
                _beta = max(0.1, min(_beta, 3.0))  # clamp
            except Exception:
                _beta = 1.0
        _adj_shock = _shock_pct * _beta
        _stress_val = _curr_val * (1 + _adj_shock)
        _loss       = _stress_val - _curr_val
        _st_curr_total   += _curr_val
        _st_stress_total += _stress_val
        _st_rows.append({
            "Ticker": _tk,
            "Valor actual": _curr_val,
            "Beta": round(_beta, 2),
            "Choque adj.": f"{_adj_shock:.1%}",
            "Valor estresado": _stress_val,
            "Impacto ($)": _loss,
            "Impacto (%)": _adj_shock,
        })

    if _st_rows:
        _st_total_loss = _st_stress_total - _st_curr_total
        _st_pct_loss   = _st_total_loss / _st_curr_total if _st_curr_total else 0

        # KPIs principales
        _sk1, _sk2, _sk3 = st.columns(3)
        with _sk1:
            kpi_card("Valor actual", f"${_st_curr_total:,.2f}")
        with _sk2:
            kpi_card("Valor estresado", f"${_st_stress_total:,.2f}", accent="#ff453a")
        with _sk3:
            kpi_card("Pérdida estimada", f"${_st_total_loss:,.2f}",
                     f"{_st_pct_loss:.1%}", accent="#ff453a")

        # Tabla por posición
        _st_df = pd.DataFrame(_st_rows).sort_values("Impacto ($)")
        _st_html_rows = ""
        for _, _r in _st_df.iterrows():
            _imp_clr = "#ff453a" if _r["Impacto ($)"] < 0 else "#30d158"
            _st_html_rows += (
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04);'>"
                f"<td style='padding:7px 10px;font-family:DM Mono,monospace;font-size:0.8rem;"
                f"color:#fff;font-weight:600;'>{_r['Ticker']}</td>"
                f"<td style='padding:7px 10px;text-align:right;font-family:DM Mono,monospace;"
                f"font-size:0.78rem;color:#aeaeb2;'>${_r['Valor actual']:,.2f}</td>"
                f"<td style='padding:7px 10px;text-align:center;font-family:DM Mono,monospace;"
                f"font-size:0.78rem;color:#8e8e93;'>{_r['Beta']}</td>"
                f"<td style='padding:7px 10px;text-align:right;font-family:DM Mono,monospace;"
                f"font-size:0.78rem;color:{_imp_clr};font-weight:600;'>{_r['Impacto ($)']:+,.2f}</td>"
                f"<td style='padding:7px 10px;text-align:right;font-family:DM Mono,monospace;"
                f"font-size:0.78rem;color:{_imp_clr};'>{_r['Impacto (%)']:.1%}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>"
            f"<th style='padding:6px 10px;text-align:left;font-size:0.65rem;color:#636366;"
            f"text-transform:uppercase;font-family:DM Mono,monospace;'>Ticker</th>"
            f"<th style='padding:6px 10px;text-align:right;font-size:0.65rem;color:#636366;"
            f"text-transform:uppercase;font-family:DM Mono,monospace;'>Valor</th>"
            f"<th style='padding:6px 10px;text-align:center;font-size:0.65rem;color:#636366;"
            f"text-transform:uppercase;font-family:DM Mono,monospace;'>Beta</th>"
            f"<th style='padding:6px 10px;text-align:right;font-size:0.65rem;color:#636366;"
            f"text-transform:uppercase;font-family:DM Mono,monospace;'>Impacto $</th>"
            f"<th style='padding:6px 10px;text-align:right;font-size:0.65rem;color:#636366;"
            f"text-transform:uppercase;font-family:DM Mono,monospace;'>Impacto %</th>"
            f"</tr></thead><tbody>{_st_html_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Sin posiciones para calcular el stress test.")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# HELPERS: Fetch rich per-stock data
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_full(ticker: str) -> dict:
    """
    Descarga todos los datos disponibles de un ticker:
    precios 1Y, indicadores técnicos, fundamentales,
    analistas, calendario de eventos, noticias.
    """
    out = {"ticker": ticker, "ok": False}
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        # ── Identidad ────────────────────────────────────────
        out["name"]      = info.get("longName") or ticker
        out["sector"]    = info.get("sector",    "—")
        out["industry"]  = info.get("industry",  "—")
        out["country"]   = info.get("country",   "—")
        out["mktcap"]    = info.get("marketCap")
        out["employees"] = info.get("fullTimeEmployees")
        out["description"] = info.get("longBusinessSummary", "")

        # ── Precio ───────────────────────────────────────────
        out["price"]      = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        out["prev_close"] = float(info.get("regularMarketPreviousClose") or info.get("previousClose") or 0)
        out["open"]       = float(info.get("regularMarketOpen") or info.get("open") or 0)
        out["day_high"]   = float(info.get("dayHigh") or info.get("regularMarketDayHigh") or 0)
        out["day_low"]    = float(info.get("dayLow")  or info.get("regularMarketDayLow")  or 0)
        out["wk52_high"]  = float(info.get("fiftyTwoWeekHigh") or 0)
        out["wk52_low"]   = float(info.get("fiftyTwoWeekLow")  or 0)
        out["volume"]     = info.get("regularMarketVolume") or info.get("volume") or 0
        out["avg_volume"] = info.get("averageVolume") or 1
        out["beta"]       = info.get("beta")
        out["change_1d"]  = (out["price"] - out["prev_close"]) / out["prev_close"] \
                            if out["prev_close"] > 0 else 0.0

        # ── Fundamentales ─────────────────────────────────────
        def _f(k): return round(float(v), 4) if (v := info.get(k)) is not None else None
        out["pe"]             = _f("trailingPE")
        out["forward_pe"]     = _f("forwardPE")
        out["peg"]            = _f("pegRatio")
        out["ev_ebitda"]      = _f("enterpriseToEbitda")
        out["pb"]             = _f("priceToBook")
        out["ps"]             = _f("priceToSalesTrailing12Months")
        out["roe"]            = _f("returnOnEquity")
        out["roa"]            = _f("returnOnAssets")
        out["profit_margin"]  = _f("profitMargins")
        out["op_margin"]      = _f("operatingMargins")
        out["gross_margin"]   = _f("grossMargins")
        out["rev_growth"]     = _f("revenueGrowth")
        out["earn_growth"]    = _f("earningsGrowth")
        out["earn_qgrowth"]   = _f("earningsQuarterlyGrowth")
        out["debt_equity"]    = _f("debtToEquity")
        out["current_ratio"]  = _f("currentRatio")
        out["quick_ratio"]    = _f("quickRatio")
        out["dividend_yield"] = _f("dividendYield")
        out["payout_ratio"]   = _f("payoutRatio")
        out["ebitda"]         = info.get("ebitda")
        out["revenue"]        = info.get("totalRevenue")
        out["fcf"]            = info.get("freeCashflow")
        out["cash"]           = info.get("totalCash")
        out["total_debt"]     = info.get("totalDebt")
        out["eps_trailing"]   = _f("trailingEps")
        out["eps_forward"]    = _f("forwardEps")
        out["next_earn_date"] = None
        out["ex_div_date"]    = None

        # ── Calendario ─────────────────────────────────────────
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    out["next_earn_date"] = str(ed[0] if isinstance(ed, list) else ed)
                dd = cal.get("Ex-Dividend Date")
                if dd:
                    out["ex_div_date"] = str(dd)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for row_lbl in ["Earnings Date", "Earnings High", "Earnings Low"]:
                    if row_lbl in cal.index:
                        out["next_earn_date"] = str(cal.loc[row_lbl].iloc[0])
                        break
        except Exception:
            pass

        # ── Analistas ────────────────────────────────────────
        out["analyst_key"]  = info.get("recommendationKey", "—")
        out["n_analysts"]   = int(info.get("numberOfAnalystOpinions") or 0)
        out["target_mean"]  = _f("targetMeanPrice")
        out["target_high"]  = _f("targetHighPrice")
        out["target_low"]   = _f("targetLowPrice")
        out["upside"]       = (out["target_mean"] - out["price"]) / out["price"] \
                              if out["target_mean"] and out["price"] > 0 else None
        out["buy"]  = 0
        out["hold"] = 0
        out["sell"] = 0
        try:
            rec = tk.recommendations
            if rec is not None and not rec.empty:
                last = rec.tail(3)
                for col in last.columns:
                    cl = col.lower()
                    if   "strong buy" in cl or "strongbuy" in cl: out["buy"]  += int(last[col].sum())
                    elif "buy"        in cl:                       out["buy"]  += int(last[col].sum())
                    elif "hold" in cl or "neutral" in cl:          out["hold"] += int(last[col].sum())
                    elif "sell" in cl or "underperform" in cl:     out["sell"] += int(last[col].sum())
        except Exception:
            pass

        # ── Noticias ──────────────────────────────────────────
        try:
            news_raw = tk.news or []
            out["news"] = [
                {
                    "title":     n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link":      n.get("link", ""),
                    "age":       _news_age(n.get("providerPublishTime", 0)),
                }
                for n in news_raw[:8]
                if n.get("title")
            ]
        except Exception:
            out["news"] = []

        # ── Historial 1 año ───────────────────────────────────
        hist = tk.history(period="1y", auto_adjust=True)
        out["hist"] = hist
        out["hist_ok"] = not hist.empty

        # ── Indicadores técnicos ──────────────────────────────
        if not hist.empty and len(hist) >= 20:
            close = hist["Close"]
            vol_s = hist["Volume"]

            # SMA
            out["sma20"]  = close.rolling(20).mean().iloc[-1]
            out["sma50"]  = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            out["sma200"] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # RSI-14
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            out["rsi"] = float((100 - 100/(1+rs)).iloc[-1])

            # MACD (12,26,9)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            out["macd"]        = float(macd_line.iloc[-1])
            out["macd_signal"] = float(signal_line.iloc[-1])
            out["macd_hist"]   = float((macd_line - signal_line).iloc[-1])

            # Bollinger Bands (20, 2σ)
            sma20_s = close.rolling(20).mean()
            std20   = close.rolling(20).std()
            out["bb_upper"] = float((sma20_s + 2*std20).iloc[-1])
            out["bb_lower"] = float((sma20_s - 2*std20).iloc[-1])
            out["bb_pct"]   = float(
                (close.iloc[-1] - (sma20_s - 2*std20).iloc[-1]) /
                (4*std20.iloc[-1])
            ) if std20.iloc[-1] > 0 else 0.5

            # Volumen relativo
            out["vol_ratio"] = float(vol_s.iloc[-1] / vol_s.rolling(20).mean().iloc[-1]) \
                               if vol_s.rolling(20).mean().iloc[-1] > 0 else 1.0

            # Retornos
            p = close
            out["ret_1w"]  = float(p.iloc[-1]/p.iloc[-5]-1)   if len(p)>=5   else None
            out["ret_1m"]  = float(p.iloc[-1]/p.iloc[-21]-1)  if len(p)>=21  else None
            out["ret_3m"]  = float(p.iloc[-1]/p.iloc[-63]-1)  if len(p)>=63  else None
            out["ret_6m"]  = float(p.iloc[-1]/p.iloc[-126]-1) if len(p)>=126 else None
            out["ret_1y"]  = float(p.iloc[-1]/p.iloc[0]-1)

            # Max drawdown (1Y)
            cum_max = p.cummax()
            dd      = (p - cum_max) / cum_max
            out["max_dd"] = float(dd.min())

            # ATH distance
            out["from_ath"] = float(p.iloc[-1]/out["wk52_high"] - 1) if out["wk52_high"] > 0 else 0
            out["from_atl"] = float(p.iloc[-1]/out["wk52_low"]  - 1) if out["wk52_low"]  > 0 else 0

        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)

    return out


def _tech_label(rsi, sma50, sma200, macd_hist, bb_pct):
    """Genera un resumen textual de la situación técnica."""
    signals = []
    if rsi is not None:
        if   rsi < 30: signals.append("RSI en sobreventa (oportunidad potencial)")
        elif rsi > 70: signals.append("RSI en sobrecompra (precaución)")
        elif rsi > 55: signals.append("RSI con momentum positivo")
        else:          signals.append("RSI en territorio neutral/bajista")

    price = 0  # placeholder; real comparison done in main
    if sma50 is not None:
        signals.append("cotiza por encima de SMA50 (tendencia alcista corto plazo)" if True
                       else "cotiza bajo SMA50 (tendencia bajista corto plazo)")

    if sma200 is not None:
        signals.append("sobre SMA200 (tendencia alcista largo plazo)")

    if macd_hist is not None:
        if   macd_hist > 0: signals.append("MACD positivo (impulso alcista)")
        else:                signals.append("MACD negativo (impulso bajista)")

    if bb_pct is not None:
        if   bb_pct > 0.8: signals.append("cerca del techo de Bandas de Bollinger")
        elif bb_pct < 0.2: signals.append("cerca del suelo de Bandas de Bollinger")

    return signals


def _build_ai_prompt(d: dict, avg_cost: float, shares: float) -> str:
    """Construye el prompt para el análisis narrativo de IA."""
    pos_val  = d["price"] * shares
    pl_pct   = (d["price"] - avg_cost) / avg_cost if avg_cost > 0 else 0
    pl_usd   = (d["price"] - avg_cost) * shares

    def pct(v): return f"{v:.1%}" if v is not None else "N/D"
    def num(v, fmt=".1f"): return f"{v:{fmt}}" if v is not None else "N/D"
    def big(v):
        if v is None: return "N/D"
        if abs(v)>=1e12: return f"${v/1e12:.1f}T"
        if abs(v)>=1e9:  return f"${v/1e9:.1f}B"
        if abs(v)>=1e6:  return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"

    total_an = d["buy"] + d["hold"] + d["sell"]
    an_str   = f"{d['buy']} compra / {d['hold']} mantener / {d['sell']} vender" \
               if total_an > 0 else "sin datos"

    news_titles = "\n".join(f"- {n['title']} ({n['publisher']}, {n['age']})"
                            for n in d.get("news", [])[:5])

    return f"""Eres un analista financiero experto. Analiza la siguiente acción de forma EXTENSA y ÚTIL para un inversor individual en México. Responde EN ESPAÑOL en 4 secciones bien definidas:

## 1. SITUACIÓN TÉCNICA
## 2. SALUD FUNDAMENTAL DEL NEGOCIO
## 3. CATALIZADORES Y RIESGOS CLAVE
## 4. CONTEXTO PARA LA DECISIÓN DE INVERSIÓN

Sé específico, usa los datos proporcionados y da perspectivas accionables. NO repitas los datos crudos, interprétalos.

=== DATOS DE LA POSICIÓN ===
Empresa: {d['name']} ({d['ticker']})
Sector: {d['sector']} | Industria: {d['industry']}
Descripción: {d.get('description','')[:400]}

=== POSICIÓN DEL INVERSOR ===
Acciones: {shares:.4f} | Costo promedio: ${avg_cost:.2f}
Precio actual: ${d['price']:.2f} | P&L: ${pl_usd:+.2f} ({pl_pct:+.1%})
Valor de posición: ${pos_val:,.2f}

=== TÉCNICO ===
Precio: ${d['price']:.2f} | Cambio hoy: {pct(d['change_1d'])}
SMA 20: ${num(d.get('sma20'))} | SMA 50: ${num(d.get('sma50'))} | SMA 200: ${num(d.get('sma200'))}
RSI(14): {num(d.get('rsi'))} | MACD hist: {num(d.get('macd_hist'),'.3f')}
Bollinger %B: {num(d.get('bb_pct'),'.2f')} | Vol relativo: {num(d.get('vol_ratio'),'.2f')}x
Retornos: 1S={pct(d.get('ret_1w'))} | 1M={pct(d.get('ret_1m'))} | 3M={pct(d.get('ret_3m'))} | 6M={pct(d.get('ret_6m'))} | 1A={pct(d.get('ret_1y'))}
Max Drawdown 1A: {pct(d.get('max_dd'))} | Desde máx 52S: {pct(d.get('from_ath'))}
Beta: {num(d.get('beta'),'.2f')}

=== FUNDAMENTAL ===
Market Cap: {big(d.get('mktcap'))} | EBITDA: {big(d.get('ebitda'))} | Revenue: {big(d.get('revenue'))} | FCF: {big(d.get('fcf'))}
P/E trailing: {num(d.get('pe'))} | P/E forward: {num(d.get('forward_pe'))} | PEG: {num(d.get('peg'))} | EV/EBITDA: {num(d.get('ev_ebitda'))}
ROE: {pct(d.get('roe'))} | ROA: {pct(d.get('roa'))} | Margen neto: {pct(d.get('profit_margin'))} | Margen op: {pct(d.get('op_margin'))}
Crec. ingresos: {pct(d.get('rev_growth'))} | Crec. ganancias: {pct(d.get('earn_growth'))} | Crec. ganancias QoQ: {pct(d.get('earn_qgrowth'))}
Deuda/Capital: {num(d.get('debt_equity'))} | Current Ratio: {num(d.get('current_ratio'))} | EPS trailing: ${num(d.get('eps_trailing'))} | EPS forward: ${num(d.get('eps_forward'))}
Dividendo: {pct(d.get('dividend_yield'))} | Deuda total: {big(d.get('total_debt'))} | Cash: {big(d.get('cash'))}

=== CONSENSO ANALISTAS ===
Recomendación: {d.get('analyst_key','—').upper()} | {d['n_analysts']} analistas
Distribución: {an_str}
Precio objetivo: media=${num(d.get('target_mean'))} | alto=${num(d.get('target_high'))} | bajo=${num(d.get('target_low'))}
Upside vs precio actual: {pct(d.get('upside'))}
Próximo reporte de earnings: {d.get('next_earn_date','No disponible')}

=== NOTICIAS RECIENTES ===
{news_titles if news_titles else "Sin noticias disponibles"}
"""


def _call_claude(prompt: str, max_tokens: int = 2000, system: str = "") -> str:
    """
    Llama al API de Claude con la key configurada en sidebar.
    Retorna el texto de la respuesta o un mensaje de error claro.
    """
    # Siempre leer Secrets primero (fuente más fresca y segura),
    # con fallback a session_state (entrada del sidebar en local)
    api_key = (_load_anthropic_key() or st.session_state.get("anthropic_api_key", "")).strip()

    if not api_key:
        return (
            "⚠️ **Sin API Key configurada.**\n\n"
            "Para usar el análisis con IA, pega tu Anthropic API Key "
            "en el campo del sidebar (empieza con `sk-ant-...`).\n"
            "Puedes obtenerla gratis en **console.anthropic.com**."
        )

    headers = {
        "Content-Type":    "application/json",
        "x-api-key":       api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    try:
        resp = __import__("requests").post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=90,
        )
        if resp.status_code != 200:
            err = resp.json().get("error", {})
            return (f"❌ Error de API ({resp.status_code}): "
                    f"{err.get('message', resp.text[:300])}")
        data  = resp.json()
        texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return "\n\n".join(texts) if texts else "Sin respuesta del modelo."
    except Exception as e:
        return f"❌ Error de conexión: {e}"


def _get_ai_analysis(prompt: str, ticker: str) -> str:
    """Análisis narrativo de acción individual."""
    return _call_claude(prompt, max_tokens=2000)


# ─────────────────────────────────────────────────────────────
# TAB: Análisis por Acción
# ─────────────────────────────────────────────────────────────

def tab_stock_deep_dive() -> None:
    hdf = holdings_df()
    if hdf.empty:
        st.info("📂  Agrega posiciones en **Portfolio Editor** primero.")
        return

    tickers = hdf["Ticker"].unique().tolist()

    # ── Selector ──────────────────────────────────────────────
    section("ANÁLISIS PROFUNDO POR ACCIÓN")
    col_sel, col_run = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox(
            "Selecciona la acción a analizar:",
            tickers,
            key="deepdive_ticker",
        )
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔬 Analizar", type="primary",
                            use_container_width=True, key="deepdive_run")

    st.divider()

    # ── Cargar datos ──────────────────────────────────────────
    cache_key = f"deepdive_{selected}"
    if run_btn or cache_key not in st.session_state:
        with st.spinner(f"Recopilando todos los datos de {selected}…"):
            d = fetch_stock_full(selected)
        st.session_state[cache_key] = d
    else:
        d = st.session_state[cache_key]

    if not d.get("ok"):
        st.error(f"No se pudieron obtener datos: {d.get('error','desconocido')}")
        return

    # Datos de la posición del usuario
    pos_row  = hdf[hdf["Ticker"] == selected]
    shares   = float(pos_row["Shares"].iloc[0])  if not pos_row.empty else 0.0
    avg_cost = float(pos_row["AvgCost"].iloc[0]) if not pos_row.empty else 0.0
    pl_usd   = (d["price"] - avg_cost) * shares
    pl_pct   = (d["price"] - avg_cost) / avg_cost if avg_cost > 0 else 0.0
    pos_val  = d["price"] * shares
    chg_color = "#30d158" if d["change_1d"] >= 0 else "#ff453a"
    pl_color  = "#30d158" if pl_pct >= 0 else "#ff453a"

    # ══════════════════════════════════════════════════════════
    # HEADER: nombre, precio, posición
    # ══════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(10,132,255,0.08),rgba(48,209,88,0.06));
                border:1px solid rgba(255,255,255,0.08); border-radius:20px;
                padding:24px 28px; margin-bottom:20px;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
          <div style="font-size:2.2rem; font-weight:800; color:#fff;
                      font-family:'DM Mono',monospace; letter-spacing:-1px;">
            {selected}
          </div>
          <div style="font-size:1rem; color:#aeaeb2; margin-top:2px;">{d['name']}</div>
          <div style="font-size:0.8rem; color:#8e8e93; margin-top:4px;">
            {d['sector']} · {d['industry']} · {d.get('country','—')}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:2.5rem; font-weight:800; color:#fff;
                      font-family:'DM Mono',monospace;">${d['price']:,.2f}</div>
          <div style="font-size:1rem; color:{chg_color}; font-family:'DM Mono',monospace;">
            {'▲' if d['change_1d']>=0 else '▼'} {abs(d['change_1d']):.2%} hoy
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs de posición ──────────────────────────────────────
    section("TU POSICIÓN")
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    with pc1: kpi_card("Acciones",    f"{shares:.4f}")
    with pc2: kpi_card("Costo Prom.", f"${avg_cost:.2f}")
    with pc3: kpi_card("Valor",       f"${pos_val:,.2f}")
    with pc4: kpi_card("P&L ($)",     f"${pl_usd:+,.2f}", accent=pl_color)
    with pc5: kpi_card("P&L (%)",     f"{pl_pct:+.2%}",   accent=pl_color)
    with pc6:
        up = d.get("upside")
        kpi_card("Upside Analistas",
                 f"{up:+.1%}" if up is not None else "—",
                 accent="#30d158" if up and up>0 else "#ff453a")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # GRÁFICA TÉCNICA COMPLETA (candlestick + vol + SMA + BB)
    # ══════════════════════════════════════════════════════════
    section("GRÁFICA TÉCNICA (1 AÑO)")
    if d.get("hist_ok"):
        hist = d["hist"]
        close = hist["Close"]
        period_btn = st.radio("Período:", ["1M","3M","6M","1A"],
                              horizontal=True, index=3, key="dd_period")
        period_map = {"1M":21,"3M":63,"6M":126,"1A":len(hist)}
        n_bars = min(period_map[period_btn], len(hist))
        h = hist.iloc[-n_bars:].copy()
        cl = close.iloc[-n_bars:]

        # Indicadores
        sma20  = cl.rolling(20).mean()
        sma50  = cl.rolling(50).mean()
        bb_mid = cl.rolling(20).mean()
        bb_std = cl.rolling(20).std()
        bb_up  = bb_mid + 2*bb_std
        bb_lo  = bb_mid - 2*bb_std

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.03,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=h.index, open=h["Open"], high=h["High"],
            low=h["Low"], close=h["Close"],
            name="Precio",
            increasing_line_color="#30d158",
            decreasing_line_color="#ff453a",
            increasing_fillcolor="rgba(48,209,88,0.6)",
            decreasing_fillcolor="rgba(255,69,58,0.6)",
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=h.index, y=bb_up.reindex(h.index),
                                  line=dict(color="rgba(255,214,10,0.4)", width=1, dash="dot"),
                                  name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=bb_lo.reindex(h.index),
                                  fill="tonexty", fillcolor="rgba(255,214,10,0.04)",
                                  line=dict(color="rgba(255,214,10,0.4)", width=1, dash="dot"),
                                  name="Bollinger Bands"), row=1, col=1)

        # SMA lines
        if len(cl) >= 20:
            fig.add_trace(go.Scatter(x=h.index, y=sma20.reindex(h.index),
                                      line=dict(color="#0a84ff", width=1.5),
                                      name="SMA 20"), row=1, col=1)
        if len(cl) >= 50:
            fig.add_trace(go.Scatter(x=h.index, y=sma50.reindex(h.index),
                                      line=dict(color="#a78bfa", width=1.5),
                                      name="SMA 50"), row=1, col=1)

        # Costo promedio
        if avg_cost > 0:
            fig.add_hline(y=avg_cost, line_dash="dash",
                          line_color="rgba(255,214,10,0.8)",
                          annotation_text=f"Tu costo ${avg_cost:.2f}",
                          annotation_position="top left", row=1, col=1)

        # RSI
        delta_r = cl.diff()
        gain_r  = delta_r.clip(lower=0).rolling(14).mean()
        loss_r  = (-delta_r.clip(upper=0)).rolling(14).mean()
        rsi_s   = 100 - 100/(1 + gain_r/loss_r.replace(0,np.nan))

        fig.add_trace(go.Scatter(x=h.index, y=rsi_s.reindex(h.index),
                                  line=dict(color="#fb923c", width=2),
                                  name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line_color="rgba(255,69,58,0.4)",
                      line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="rgba(48,209,88,0.4)",
                      line_dash="dot", row=2, col=1)

        # MACD
        ema12_s  = cl.ewm(span=12).mean()
        ema26_s  = cl.ewm(span=26).mean()
        macd_s   = ema12_s - ema26_s
        signal_s = macd_s.ewm(span=9).mean()
        hist_s   = macd_s - signal_s

        colors_macd = ["#30d158" if v >= 0 else "#ff453a"
                       for v in hist_s.reindex(h.index).fillna(0)]
        fig.add_trace(go.Bar(x=h.index, y=hist_s.reindex(h.index),
                             marker_color=colors_macd, name="MACD Hist",
                             showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=macd_s.reindex(h.index),
                                  line=dict(color="#0a84ff", width=1.5),
                                  name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=signal_s.reindex(h.index),
                                  line=dict(color="#ff453a", width=1.5),
                                  name="Signal"), row=3, col=1)

        # Build layout without any keys we override, then merge overrides
        _base = {k: v for k, v in PLOTLY_LAYOUT.items()
                 if k not in ("xaxis", "yaxis", "legend")}
        fig.update_layout(
            **_base,
            height=620,
            xaxis_rangeslider_visible=False,
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", side="right"),
            yaxis2=dict(title="RSI",  gridcolor="rgba(255,255,255,0.04)",
                        range=[0, 100], side="right"),
            yaxis3=dict(title="MACD", gridcolor="rgba(255,255,255,0.04)", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # MÉTRICAS TÉCNICAS + FUNDAMENTALES + ANALISTAS
    # ══════════════════════════════════════════════════════════
    col_tech, col_fund, col_an = st.columns(3)

    with col_tech:
        section("TÉCNICO")
        price = d["price"]
        def _bar(label, val, lo, hi, fmt=".1f", good_high=True):
            if val is None: return
            norm = (val - lo) / (hi - lo) if hi != lo else 0.5
            norm = max(0, min(1, norm))
            c = "#30d158" if (norm > 0.6) == good_high else "#ff453a"
            if 0.4 <= norm <= 0.6: c = "#ffd60a"
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<div style='display:flex; justify-content:space-between; font-size:0.78rem;'>"
                f"<span style='color:#aeaeb2'>{label}</span>"
                f"<span style='color:#ffffff; font-family:var(--mono)'>{val:{fmt}}</span></div>"
                f"<div style='background:rgba(255,255,255,0.05); border-radius:4px; height:6px; margin-top:3px;'>"
                f"<div style='width:{norm*100:.0f}%; background:{c}; height:6px; border-radius:4px;'></div></div>"
                f"</div>", unsafe_allow_html=True)

        _bar("RSI (14)",  d.get("rsi"), 0, 100, ".1f",  good_high=False)
        _bar("Vol relativo", d.get("vol_ratio"), 0, 3, ".2f", good_high=True)
        _bar("BB %B",     d.get("bb_pct"), 0, 1, ".2f", good_high=False)

        st.markdown(f"""
        <div style='font-family:var(--mono); font-size:0.82rem; line-height:2.0;'>
        <span style='color:#8e8e93'>SMA 20</span>&emsp;
        <span style='color:{"#30d158" if d.get("sma20") and price>d.get("sma20",0) else "#ff453a"}'>
        ${d.get("sma20",0):.2f} {"▲" if d.get("sma20") and price>d.get("sma20",0) else "▼"}</span><br>
        <span style='color:#8e8e93'>SMA 50</span>&emsp;
        <span style='color:{"#30d158" if d.get("sma50") and price>d.get("sma50",0) else "#ff453a"}'>
        ${d.get("sma50",0) or 0:.2f} {"▲" if d.get("sma50") and price>d.get("sma50",0) else "▼"}</span><br>
        <span style='color:#8e8e93'>SMA 200</span>&emsp;
        <span style='color:{"#30d158" if d.get("sma200") and price>d.get("sma200",0) else "#ff453a"}'>
        ${d.get("sma200",0) or 0:.2f} {"▲" if d.get("sma200") and price>d.get("sma200",0) else "▼"}</span><br>
        <span style='color:#8e8e93'>MACD hist</span>&emsp;
        <span style='color:{"#30d158" if (d.get("macd_hist") or 0)>0 else "#ff453a"}'>
        {d.get("macd_hist",0) or 0:+.3f}</span><br>
        <span style='color:#8e8e93'>52S Máx</span>&emsp;
        <span style='color:#ffffff'>${d.get("wk52_high",0):.2f}
        ({d.get("from_ath",0)*100:+.1f}%)</span><br>
        <span style='color:#8e8e93'>52S Mín</span>&emsp;
        <span style='color:#ffffff'>${d.get("wk52_low",0):.2f}
        ({d.get("from_atl",0)*100:+.1f}%)</span><br>
        <span style='color:#8e8e93'>Beta</span>&emsp;
        <span style='color:#ffffff'>{d.get("beta") or "N/D"}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><b style='color:#aeaeb2; font-size:0.78rem;'>RETORNOS</b>", unsafe_allow_html=True)
        for label, key in [("1 Semana","ret_1w"),("1 Mes","ret_1m"),
                            ("3 Meses","ret_3m"),("6 Meses","ret_6m"),("1 Año","ret_1y")]:
            v = d.get(key)
            if v is not None:
                c = "#30d158" if v >= 0 else "#ff453a"
                st.markdown(
                    f"<span style='font-family:var(--mono);font-size:0.82rem;"
                    f"color:#8e8e93'>{label}</span>&emsp;"
                    f"<span style='color:{c};font-weight:700;font-family:var(--mono)'>"
                    f"{v:+.2%}</span><br>", unsafe_allow_html=True)

    with col_fund:
        section("FUNDAMENTAL")
        def _row(label, val, fmt=".2f", is_pct=False, good_high=True):
            if val is None: return
            c = "#aeaeb2"
            try:
                if is_pct:
                    c = "#30d158" if (val > 0) == good_high else "#ff453a"
                    disp = f"{val:.1%}"
                else:
                    disp = f"{val:{fmt}}"
            except Exception:
                disp = str(val)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:0.82rem;margin-bottom:6px;border-bottom:1px solid "
                f"rgba(255,255,255,0.04);padding-bottom:6px;'>"
                f"<span style='color:#8e8e93'>{label}</span>"
                f"<span style='color:{c};font-family:var(--mono);font-weight:600'>{disp}</span>"
                f"</div>", unsafe_allow_html=True)

        def _big_row(label, val):
            disp = _fmt_big(val) if val else "—"
            _row(label, 1, fmt="", is_pct=False)  # dummy to trigger
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:0.82rem;margin-bottom:6px;border-bottom:1px solid "
                f"rgba(255,255,255,0.04);padding-bottom:6px;'>"
                f"<span style='color:#8e8e93'>{label}</span>"
                f"<span style='color:#ffffff;font-family:var(--mono)'>{disp}</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<span style='color:#8e8e93;font-size:0.72rem;'>VALUACIÓN</span>", unsafe_allow_html=True)
        _row("P/E trailing",  d.get("pe"),         ".1f")
        _row("P/E forward",   d.get("forward_pe"), ".1f")
        _row("PEG",           d.get("peg"),         ".2f")
        _row("EV/EBITDA",     d.get("ev_ebitda"),  ".1f")
        _row("P/Book",        d.get("pb"),          ".2f")
        _row("P/Sales",       d.get("ps"),          ".2f")
        st.markdown("<br><span style='color:#8e8e93;font-size:0.72rem;'>RENTABILIDAD</span>", unsafe_allow_html=True)
        _row("ROE",           d.get("roe"),          ".1%", True, True)
        _row("ROA",           d.get("roa"),          ".1%", True, True)
        _row("Margen Bruto",  d.get("gross_margin"), ".1%", True, True)
        _row("Margen Neto",   d.get("profit_margin"),".1%", True, True)
        _row("Margen Op.",    d.get("op_margin"),    ".1%", True, True)
        st.markdown("<br><span style='color:#8e8e93;font-size:0.72rem;'>CRECIMIENTO</span>", unsafe_allow_html=True)
        _row("Crec. Ingresos",  d.get("rev_growth"),   ".1%", True, True)
        _row("Crec. Ganancias", d.get("earn_growth"),  ".1%", True, True)
        _row("Crec. Gan. QoQ",  d.get("earn_qgrowth"), ".1%", True, True)
        st.markdown("<br><span style='color:#8e8e93;font-size:0.72rem;'>SALUD FINANCIERA</span>", unsafe_allow_html=True)
        _row("Deuda/Capital",   d.get("debt_equity"),   ".1f")
        _row("Current Ratio",   d.get("current_ratio"), ".2f")
        _row("Quick Ratio",     d.get("quick_ratio"),   ".2f")
        _row("Dividendo",       d.get("dividend_yield"),".2%", True, True)
        st.markdown("<br><span style='color:#8e8e93;font-size:0.72rem;'>TAMAÑO</span>", unsafe_allow_html=True)
        for lbl, v in [("Market Cap",d.get("mktcap")),("EBITDA",d.get("ebitda")),
                       ("Revenue",d.get("revenue")),("FCF",d.get("fcf")),
                       ("Cash",d.get("cash")),("Deuda Total",d.get("total_debt"))]:
            if v:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.82rem;margin-bottom:6px;border-bottom:1px solid "
                    f"rgba(255,255,255,0.04);padding-bottom:6px;'>"
                    f"<span style='color:#8e8e93'>{lbl}</span>"
                    f"<span style='color:#ffffff;font-family:var(--mono)'>{_fmt_big(v)}</span>"
                    f"</div>", unsafe_allow_html=True)

    with col_an:
        section("ANALISTAS & EVENTOS")
        emoji_k, label_k, color_k = _analyst_badge(d.get("analyst_key",""))
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border-radius:14px;
                    padding:16px; margin-bottom:16px; text-align:center;'>
          <div style='font-size:2rem;'>{emoji_k}</div>
          <div style='font-size:1.2rem; font-weight:700; color:{color_k};
                      font-family:var(--mono); margin-top:6px;'>{label_k}</div>
          <div style='font-size:0.8rem; color:#8e8e93; margin-top:4px;'>
            {d['n_analysts']} analistas</div>
        </div>
        """, unsafe_allow_html=True)

        # Price target vs actual
        if d.get("target_mean"):
            tm, tl, th = d["target_mean"], d.get("target_low",0), d.get("target_high",0)
            up = d.get("upside", 0) or 0
            c_up = "#30d158" if up > 0 else "#ff453a"
            st.markdown(f"""
            <div style='font-family:var(--mono); font-size:0.83rem; line-height:2.0;'>
            <span style='color:#8e8e93'>Precio obj. medio</span>&emsp;
            <b style='color:#ffffff'>${tm:.2f}</b>
            <span style='color:{c_up}; font-size:0.78rem;'> ({up:+.1%})</span><br>
            <span style='color:#8e8e93'>Rango objetivo</span>&emsp;
            <span style='color:#ffffff'>${tl:.2f} – ${th:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

        # Buy/Hold/Sell bar
        total_an = d["buy"] + d["hold"] + d["sell"]
        if total_an > 0:
            bpct = d["buy"]/total_an; hpct = d["hold"]/total_an; spct = d["sell"]/total_an
            st.markdown(
                f'<div style="display:flex; height:12px; border-radius:6px; overflow:hidden; margin:12px 0;">'
                f'<div style="width:{bpct*100:.0f}%; background:#30d158;"></div>'
                f'<div style="width:{hpct*100:.0f}%; background:#ffd60a;"></div>'
                f'<div style="width:{spct*100:.0f}%; background:#ff453a;"></div></div>'
                f'<div style="font-size:0.72rem; color:#8e8e93; font-family:var(--mono);">'
                f'🟢 {d["buy"]} &nbsp; 🟡 {d["hold"]} &nbsp; 🔴 {d["sell"]}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Eventos
        st.markdown("<span style='color:#8e8e93;font-size:0.72rem;font-weight:600;'>PRÓXIMOS EVENTOS</span>", unsafe_allow_html=True)
        earn_d = d.get("next_earn_date")
        div_d  = d.get("ex_div_date")
        if earn_d:
            st.markdown(f"📅 **Earnings:** `{earn_d}`")
        else:
            st.markdown("📅 **Earnings:** fecha no disponible")
        if div_d:
            st.markdown(f"💰 **Ex-Dividendo:** `{div_d}`")

        st.markdown("<br>", unsafe_allow_html=True)
        # Noticias
        st.markdown("<span style='color:#8e8e93;font-size:0.72rem;font-weight:600;'>NOTICIAS RECIENTES</span>", unsafe_allow_html=True)
        for n in d.get("news", [])[:6]:
            if n["title"]:
                age_s = f" *{n['age']}*" if n.get("age") else ""
                pub_s = f" — {n['publisher']}" if n.get("publisher") else ""
                link  = n.get("link","")
                txt   = n['title'][:65] + ("…" if len(n['title'])>65 else "")
                if link:
                    st.markdown(f"• [{txt}]({link}){pub_s}{age_s}")
                else:
                    st.markdown(f"• {txt}{pub_s}{age_s}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ANÁLISIS NARRATIVO (IA)
    # ══════════════════════════════════════════════════════════
    section("🤖 ANÁLISIS NARRATIVO (IA)")
    st.caption(
        "Claude analiza todos los datos anteriores y genera un diagnóstico "
        "en lenguaje claro sobre qué está pasando con esta acción."
    )

    ai_key   = f"ai_analysis_{selected}"
    gen_btn  = st.button("✨ Generar Análisis con IA", type="primary",
                          key=f"gen_ai_{selected}")

    if gen_btn or ai_key in st.session_state:
        if gen_btn or ai_key not in st.session_state:
            with st.spinner("Claude está analizando la acción… (15–30 segundos)"):
                prompt = _build_ai_prompt(d, avg_cost, shares)
                analysis_text = _get_ai_analysis(prompt, selected)
            st.session_state[ai_key] = analysis_text

        analysis = st.session_state[ai_key]
        st.markdown(
            f'<div style="background:rgba(22,22,31,0.8); border:1px solid rgba(10,132,255,0.2);\n'
            f'border-radius:16px; padding:24px 28px; line-height:1.8; color:#ffffff;">\n'
            f'{analysis}\n</div>',
            unsafe_allow_html=True
        )

        if st.button("🔄 Regenerar análisis", key=f"regen_{selected}", type="secondary"):
            if ai_key in st.session_state:
                del st.session_state[ai_key]
            st.rerun()
    else:
        st.info("Presiona **Generar Análisis con IA** para obtener el diagnóstico narrativo de esta acción.")


# ─────────────────────────────────────────────────────────────
# ANIMACIONES DE CARGA
# ─────────────────────────────────────────────────────────────

def loading_animation(msg: str = "Cargando…") -> None:
    """Barra de progreso animada con mensaje personalizado."""
    import time
    bar = st.progress(0, text=f"⏳ {msg}")
    for i in range(0, 101, 5):
        bar.progress(i, text=f"⏳ {msg} ({i}%)")
        time.sleep(0.02)
    bar.empty()


# ─────────────────────────────────────────────────────────────
# ONBOARDING — pantalla de bienvenida
# ─────────────────────────────────────────────────────────────

def show_onboarding() -> None:
    """Pantalla de bienvenida cuando no hay posiciones cargadas."""
    st.markdown("""
<style>
@keyframes _ob_fade {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0);    }
}
.ob-wrap { animation: _ob_fade .6s ease-out both; }
.ob-step {
    background: rgba(22,22,31,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px 24px;
    text-align: center;
    transition: border-color .2s;
}
.ob-step:hover { border-color: rgba(10,132,255,0.4); }
.ob-num {
    width: 40px; height: 40px; border-radius: 50%;
    background: rgba(10,132,255,0.15);
    border: 1.5px solid rgba(10,132,255,0.4);
    display: flex; align-items: center; justify-content: center;
    font-family: DM Mono, monospace; font-weight: 700;
    color: #0a84ff; font-size: 1rem; margin: 0 auto 14px;
}
</style>

<div class="ob-wrap">
  <div style="text-align:center; padding: 40px 0 30px;">
    <div style="font-size:3.5rem; margin-bottom:12px;">📊</div>
    <div style="font-size:2rem; font-weight:800; color:#fff;
                font-family:DM Mono,monospace; letter-spacing:-1px;">
      Portfolio Manager
    </div>
    <div style="font-size:1rem; color:#8e8e93; margin-top:8px;">
      Gestión inteligente de portafolios · Análisis técnico y fundamental · Rebalanceo óptimo
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    steps = [
        ("1", "📂 Carga tu portafolio",
         "Ve a **Portfolio Editor**, sube un CSV de transacciones o agrégalas manualmente una a una.",
         "💼 Portfolio Editor"),
        ("2", "🎯 Define tus objetivos",
         "Configura los pesos objetivo de cada acción y tu benchmark de referencia.",
         "⚖️ Rebalanceo"),
        ("3", "📈 Analiza y decide",
         "Revisa el dashboard en tiempo real, analiza cada posición con IA y detecta oportunidades.",
         "🧬 Análisis por Acción"),
    ]
    for col, (num, title, desc, _) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
<div class="ob-step">
  <div class="ob-num">{num}</div>
  <div style="font-weight:700;color:#ffffff;margin-bottom:8px;">{title}</div>
  <div style="font-size:0.82rem;color:#8e8e93;line-height:1.6;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick-start: demo data
    st.markdown("""
<div style="background:rgba(10,132,255,0.06);border:1px solid rgba(10,132,255,0.2);
            border-radius:14px;padding:16px 20px;text-align:center;">
  <span style="font-size:0.85rem;color:#aeaeb2;">
  ✨ <b style="color:#ffffff;">Tip:</b> Puedes importar directamente desde un CSV de GBM, 
  IBKR o cualquier broker usando el importador de la pestaña <b>Portfolio Editor</b>.
  </span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SISTEMA DE ALERTAS
# ─────────────────────────────────────────────────────────────

ALERT_TYPES = {
    "price_above":  "Precio supera",
    "price_below":  "Precio cae bajo",
    "rsi_overbought": "RSI > 70 (sobrecompra)",
    "rsi_oversold":   "RSI < 30 (sobreventa)",
    "drawdown":     "Drawdown supera",
    "earnings_near": "Earnings en menos de X días",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_earnings_calendar(tickers: list[str]) -> dict:
    """Obtiene fechas de earnings para múltiples tickers."""
    out = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            earn_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and ed:
                    earn_date = pd.to_datetime(ed[0])
                elif ed:
                    earn_date = pd.to_datetime(ed)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for lbl in ["Earnings Date"]:
                    if lbl in cal.index:
                        earn_date = pd.to_datetime(cal.loc[lbl].iloc[0])
                        break
            if earn_date and pd.notna(earn_date):
                days_away = (earn_date.date() - date.today()).days
                out[t] = {"date": earn_date.strftime("%Y-%m-%d"),
                          "days_away": days_away}
        except Exception:
            pass
    return out


def check_alerts(holdings: pd.DataFrame, prices: dict,
                 pulse: dict | None = None) -> list[dict]:
    """
    Evalúa todas las alertas activas contra precios y datos actuales.
    Retorna lista de alertas disparadas: {ticker, type, message, severity}
    """
    alerts_config = st.session_state.get("alerts", [])
    tickers       = holdings["Ticker"].tolist() if not holdings.empty else []
    triggered     = []

    # Earnings próximos (siempre verificar)
    if tickers:
        earnings = fetch_earnings_calendar(tuple(tickers))
        for t, e in earnings.items():
            da = e["days_away"]
            if 0 <= da <= 7:
                triggered.append({
                    "ticker": t, "type": "earnings_near",
                    "message": f"📅 **{t}** reporta earnings el **{e['date']}** ({da} días)",
                    "severity": "warning" if da <= 3 else "info",
                })

    for alert in alerts_config:
        if not alert.get("active", True):
            continue
        t         = alert.get("ticker", "")
        atype     = alert.get("type", "")
        threshold = float(alert.get("threshold", 0))

        info  = prices.get(t, {})
        price = float(info.get("price", 0))

        if atype == "price_above" and price > 0 and price > threshold:
            triggered.append({
                "ticker": t, "type": atype,
                "message": f"📈 **{t}** superó ${threshold:.2f} → cotiza ${price:.2f}",
                "severity": "success",
            })
        elif atype == "price_below" and price > 0 and price < threshold:
            triggered.append({
                "ticker": t, "type": atype,
                "message": f"📉 **{t}** cayó bajo ${threshold:.2f} → cotiza ${price:.2f}",
                "severity": "error",
            })
        elif atype == "rsi_overbought" and pulse:
            rsi = pulse.get(t, {}).get("rsi14")
            if rsi and rsi > 70:
                triggered.append({
                    "ticker": t, "type": atype,
                    "message": f"⚠️ **{t}** RSI en sobrecompra: {rsi:.0f}",
                    "severity": "warning",
                })
        elif atype == "rsi_oversold" and pulse:
            rsi = pulse.get(t, {}).get("rsi14")
            if rsi and rsi < 30:
                triggered.append({
                    "ticker": t, "type": atype,
                    "message": f"🟢 **{t}** RSI en sobreventa: {rsi:.0f} (posible oportunidad)",
                    "severity": "info",
                })
        elif atype == "drawdown":
            # Calcular drawdown desde costo promedio
            pos = holdings[holdings["Ticker"] == t]
            if not pos.empty:
                avg_cost = float(pos["AvgCost"].iloc[0])
                if avg_cost > 0 and price > 0:
                    dd = (price - avg_cost) / avg_cost
                    if dd < -threshold / 100:
                        triggered.append({
                            "ticker": t, "type": atype,
                            "message": (f"🔴 **{t}** en drawdown de "
                                       f"{dd:.1%} (umbral: -{threshold:.0f}%)"),
                            "severity": "error",
                        })

    return triggered


def render_alerts_banner(triggered: list[dict]) -> None:
    """Muestra las alertas disparadas como banners en la parte superior."""
    if not triggered:
        return
    for alert in triggered:
        sev = alert.get("severity", "info")
        msg = alert["message"]
        if sev == "error":
            st.error(msg)
        elif sev == "warning":
            st.warning(msg)
        elif sev == "success":
            st.success(msg)
        else:
            st.info(msg)


# ─────────────────────────────────────────────────────────────
# DIARIO DE TESIS
# ─────────────────────────────────────────────────────────────

def tab_thesis() -> None:
    """Tab del diario de tesis de inversión por posición."""
    hdf = holdings_df()

    section("DIARIO DE TESIS DE INVERSIÓN")
    st.caption(
        "Documenta por qué compraste cada posición, tu precio objetivo, "
        "el catalizador esperado y cuándo planeas revisar la tesis."
    )

    if hdf.empty:
        st.info("Agrega posiciones en Portfolio Editor para gestionar tus tesis.")
        return

    tickers  = hdf["Ticker"].unique().tolist()
    thesis   = st.session_state.get("thesis", {})
    prices   = fetch_live_prices(tickers)

    # ── Indicador de cobertura ──────────────────────────────
    covered  = sum(1 for t in tickers if thesis.get(t, {}).get("thesis","").strip())
    st.markdown(f"""
<div style="background:rgba(10,132,255,0.07);border:1px solid rgba(10,132,255,0.2);
            border-radius:12px;padding:12px 18px;margin-bottom:16px;
            font-family:DM Mono,monospace;font-size:0.82rem;color:#aeaeb2;">
  📝 Tesis documentadas:
  <b style="color:{'#30d158' if covered==len(tickers) else '#ffd60a'}">
    {covered}/{len(tickers)}</b> posiciones
  {'✓ Todas documentadas' if covered==len(tickers) else '— documenta las restantes para mejor seguimiento'}
</div>""", unsafe_allow_html=True)

    # ── Una tarjeta por posición ────────────────────────────
    for t in tickers:
        info     = prices.get(t, {})
        price    = info.get("price", 0.0)
        prev     = info.get("prev_close", 0.0)
        chg      = (price - prev) / prev if prev > 0 else 0.0
        pos_row  = hdf[hdf["Ticker"] == t]
        avg_cost = float(pos_row["AvgCost"].iloc[0]) if not pos_row.empty else 0.0
        pl_pct   = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

        th = thesis.get(t, {})
        has_thesis = bool(th.get("thesis","").strip())

        clr_pl = "#30d158" if pl_pct >= 0 else "#ff453a"
        clr_chg = "#30d158" if chg >= 0 else "#ff453a"
        border = "rgba(48,209,88,0.2)" if has_thesis else "rgba(255,255,255,0.06)"

        with st.expander(
            f"{'✅' if has_thesis else '📝'} **{t}** — "
            f"${price:,.2f}  "
            f"({chg:+.2%} hoy)  ·  P&L: {pl_pct:+.1%}",
            expanded=not has_thesis,
        ):
            col_form, col_meta = st.columns([3, 1])

            with col_form:
                new_thesis = st.text_area(
                    "📌 Tesis de inversión (por qué la compraste):",
                    value=th.get("thesis", ""),
                    height=110,
                    key=f"th_text_{t}",
                    placeholder=(
                        f"Ej: Compré {t} porque... "
                        "El catalizador principal es... "
                        "La salida planificada es cuando..."
                    ),
                )
                new_catalyst = st.text_input(
                    "⚡ Catalizador esperado:",
                    value=th.get("catalyst", ""),
                    key=f"th_cat_{t}",
                    placeholder="Ej: Lanzamiento GTA 6, Q3 earnings, aprobación FDA…",
                )
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    new_target = st.number_input(
                        "🎯 Precio objetivo ($):",
                        min_value=0.0, value=float(th.get("target_price", 0)),
                        step=1.0, key=f"th_tgt_{t}", format="%.2f",
                    )
                with tc2:
                    new_stop = st.number_input(
                        "🛑 Stop loss ($):",
                        min_value=0.0, value=float(th.get("stop_loss", 0)),
                        step=1.0, key=f"th_stop_{t}", format="%.2f",
                    )
                with tc3:
                    raw_rev = th.get("review_date", "")
                    try:
                        rev_default = pd.to_datetime(raw_rev).date() if raw_rev else date.today() + timedelta(days=90)
                    except Exception:
                        rev_default = date.today() + timedelta(days=90)
                    new_review = st.date_input(
                        "📅 Fecha de revisión:",
                        value=rev_default, key=f"th_rev_{t}",
                    )

                new_notes = st.text_input(
                    "💬 Notas adicionales:",
                    value=th.get("notes", ""),
                    key=f"th_notes_{t}",
                )

                if st.button(f"💾 Guardar tesis de {t}", key=f"th_save_{t}",
                             type="primary", use_container_width=True):
                    thesis[t] = {
                        "thesis":       new_thesis,
                        "catalyst":     new_catalyst,
                        "target_price": new_target,
                        "stop_loss":    new_stop,
                        "review_date":  str(new_review),
                        "notes":        new_notes,
                        "added_date":   th.get("added_date", str(date.today())),
                        "updated_date": str(date.today()),
                    }
                    st.session_state["thesis"] = thesis
                    st.success(f"Tesis de {t} guardada.")
                    st.rerun()

            with col_meta:
                # Métricas de la posición
                if new_target > 0:
                    upside = (new_target - price) / price if price > 0 else 0
                    up_c   = "#30d158" if upside > 0 else "#ff453a"
                    kpi_card("Upside a objetivo", f"{upside:+.1%}", accent=up_c)
                if new_stop > 0 and avg_cost > 0:
                    risk = (new_stop - avg_cost) / avg_cost
                    kpi_card("Riesgo (stop)", f"{risk:.1%}", accent="#ff453a")
                kpi_card("P&L actual", f"{pl_pct:+.1%}", accent=clr_pl)

                # Estado de la tesis
                if th.get("review_date"):
                    try:
                        rev = pd.to_datetime(th["review_date"]).date()
                        days_to_rev = (rev - date.today()).days
                        if days_to_rev < 0:
                            st.warning(f"⏰ Revisión vencida hace {abs(days_to_rev)}d")
                        elif days_to_rev <= 7:
                            st.warning(f"⏰ Revisión en {days_to_rev}d")
                        else:
                            st.info(f"📅 Revisión en {days_to_rev}d")
                    except Exception:
                        pass

                # Borrar tesis
                if has_thesis:
                    if st.button(f"🗑️ Borrar tesis", key=f"th_del_{t}",
                                 type="secondary", use_container_width=True):
                        thesis.pop(t, None)
                        st.session_state["thesis"] = thesis
                        st.rerun()


# ─────────────────────────────────────────────────────────────
# ALERTAS — configuración
# ─────────────────────────────────────────────────────────────

def tab_alerts() -> None:
    """Tab de configuración y monitoreo de alertas."""
    hdf      = holdings_df()
    tickers  = hdf["Ticker"].unique().tolist() if not hdf.empty else []
    alerts   = st.session_state.get("alerts", [])

    section("CONFIGURAR ALERTAS")
    st.caption(
        "Las alertas se evalúan automáticamente cada vez que abres el Dashboard. "
        "También puedes activar Refresh para verificarlas ahora."
    )

    # ── Nueva alerta ────────────────────────────────────────
    col_f, col_act = st.columns([2, 1], gap="large")
    with col_f:
        st.markdown("**➕ Nueva alerta:**")
        al_ticker = st.selectbox("Activo:", tickers if tickers else UNIVERSE[:20], key="al_t")
        al_type   = st.selectbox(
            "Condición:", list(ALERT_TYPES.keys()),
            format_func=lambda k: ALERT_TYPES[k], key="al_type",
        )
        al_threshold = 0.0
        if al_type in ("price_above", "price_below"):
            info    = fetch_live_prices([al_ticker]).get(al_ticker, {})
            current = info.get("price", 0)
            al_threshold = st.number_input(
                f"Precio umbral ($) — actual: ${current:.2f}",
                min_value=0.0, value=float(f"{current:.2f}"),
                step=1.0, key="al_thresh",
            )
        elif al_type == "drawdown":
            al_threshold = st.number_input(
                "Drawdown máximo desde costo promedio (%):",
                min_value=1.0, max_value=100.0, value=15.0, step=1.0, key="al_dd",
            )
        elif al_type == "earnings_near":
            al_threshold = st.number_input(
                "Avisar cuando falten X días:",
                min_value=1.0, max_value=30.0, value=7.0, step=1.0, key="al_earn",
            )

        if st.button("➕ Agregar alerta", type="primary", use_container_width=True, key="al_add"):
            alerts.append({
                "ticker":    al_ticker,
                "type":      al_type,
                "label":     ALERT_TYPES[al_type],
                "threshold": al_threshold,
                "active":    True,
                "created":   str(date.today()),
            })
            st.session_state["alerts"] = alerts
            st.success("Alerta agregada.")
            st.rerun()

    with col_act:
        st.markdown("**📋 Alertas activas:**")
        if not alerts:
            st.info("Sin alertas configuradas.")
        else:
            for i, al in enumerate(alerts):
                active = al.get("active", True)
                st.markdown(
                    f"<div style='font-family:DM Mono,monospace;font-size:0.78rem;"
                    f"color:{'#ffffff' if active else '#8e8e93'};padding:4px 0;'>"
                    f"{'✅' if active else '⏸️'} <b>{al['ticker']}</b> — "
                    f"{al['label']}"
                    f"{(' @ $'+str(al['threshold'])) if al['type'] in ('price_above','price_below') else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    tog_label = "Pausar" if active else "Activar"
                    if st.button(tog_label, key=f"al_tog_{i}", use_container_width=True):
                        alerts[i]["active"] = not active
                        st.session_state["alerts"] = alerts
                        st.rerun()
                with c2:
                    if st.button("🗑️", key=f"al_del_{i}", use_container_width=True):
                        alerts.pop(i)
                        st.session_state["alerts"] = alerts
                        st.rerun()

    # ── Earnings calendar ───────────────────────────────────
    if tickers:
        st.markdown("<br>", unsafe_allow_html=True)
        section("📅 PRÓXIMOS EARNINGS")
        with st.spinner("Consultando calendario de earnings…"):
            earnings = fetch_earnings_calendar(tuple(tickers))

        if earnings:
            earn_rows = sorted(
                [{"Ticker": t, "Fecha": e["date"], "En días": e["days_away"]}
                 for t, e in earnings.items() if e["days_away"] >= -5],
                key=lambda x: x["En días"],
            )
            if earn_rows:
                df_earn = pd.DataFrame(earn_rows)
                def _earn_color(days):
                    if days <= 3:  return "🔴"
                    if days <= 7:  return "🟡"
                    if days <= 30: return "🟢"
                    return "⚪"
                df_earn["Alerta"] = df_earn["En días"].apply(_earn_color)
                st.dataframe(
                    df_earn[["Alerta","Ticker","Fecha","En días"]],
                    column_config={
                        "Alerta":  st.column_config.TextColumn("", width="small"),
                        "Ticker":  st.column_config.TextColumn("Ticker", width="small"),
                        "Fecha":   st.column_config.TextColumn("Fecha"),
                        "En días": st.column_config.NumberColumn("En días", format="%d"),
                    },
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("Sin earnings próximos en el horizonte.")
        else:
            st.info("No se encontraron fechas de earnings disponibles.")


def sidebar() -> None:
    st.sidebar.markdown("""
    <div style="padding: 16px 0 12px;">
        <div class="hero-title">Portfolio<br>Manager</div>
        <div class="hero-sub">Gestión · Rebalanceo · Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    # ── API Key de Anthropic ──────────────────────────────────
    st.sidebar.markdown("#### 🤖 Claude AI")
    _ak_from_secrets = bool(_load_anthropic_key())

    if _ak_from_secrets:
        # Viene de Streamlit Secrets — solo mostrar estado
        st.sidebar.markdown(
            "<div style='background:rgba(48,209,88,0.07);border:1px solid rgba(48,209,88,0.2);"
            "border-radius:10px;padding:8px 12px;font-size:0.78rem;color:#30d158;"
            "font-family:DM Mono,monospace;'>✓ API Key configurada via Secrets</div>",
            unsafe_allow_html=True,
        )
    else:
        # Sin Secrets — mostrar input (útil en local)
        api_key_input = st.sidebar.text_input(
            "Anthropic API Key:",
            value=st.session_state.get("anthropic_api_key", ""),
            type="password",
            placeholder="sk-ant-api03-...",
            help="Solo necesario si corres la app localmente.",
            key="_api_key_raw",
        )
        if st.sidebar.button("💾 Guardar API Key", use_container_width=True, key="_save_key"):
            cleaned = api_key_input.strip().strip('"').strip("'")
            if cleaned.startswith("sk-ant"):
                st.session_state["anthropic_api_key"] = cleaned
                _write_credential(_ANTHROPIC_KEY_FILE, cleaned)
                st.sidebar.success("✓ Key guardada")
            elif cleaned:
                st.sidebar.error("Debe empezar con 'sk-ant-...'")
            else:
                st.session_state["anthropic_api_key"] = ""
                _write_credential(_ANTHROPIC_KEY_FILE, "")
                st.sidebar.warning("Key borrada")
        st.sidebar.caption("En producción usa Streamlit Secrets.")

    st.sidebar.divider()

    # ── GitHub Gist — Persistencia en la nube ────────────────
    st.sidebar.markdown("#### ☁️ GitHub · Nube")
    _gh_token = _load_github_token()
    _gh_gist_id = _load_gist_id()

    if _gh_token and _gh_gist_id:
        # Contar portafolios en el Gist
        _gist_files = _gist_get_files()
        _pf_count = sum(1 for f in _gist_files if f.startswith(_GIST_PF_PREFIX))
        _gist_id_short = _gh_gist_id[:8] + "…"
        # Verificar si ya está guardado en Secrets (no necesita recordatorio)
        _gist_id_in_secrets = False
        try:
            _gist_id_in_secrets = bool(st.secrets.get("GITHUB_GIST_ID", ""))
        except Exception:
            pass
        _token_in_secrets = False
        try:
            _token_in_secrets = bool(st.secrets.get("GITHUB_TOKEN", ""))
        except Exception:
            pass
        _needs_secrets_tip = not (_gist_id_in_secrets and _token_in_secrets)
        st.sidebar.markdown(
            f"<div style='background:rgba(48,209,88,0.07);border:1px solid "
            f"rgba(48,209,88,0.2);border-radius:10px;padding:10px 12px;'>"
            f"<div style='font-size:0.6rem;color:#8e8e93;font-family:DM Mono,monospace;"
            f"text-transform:uppercase;letter-spacing:.8px;'>GITHUB GIST · CONECTADO</div>"
            f"<div style='font-size:1rem;font-weight:700;color:#30d158;"
            f"font-family:DM Mono,monospace;margin-top:3px;'>"
            f"✓ {_pf_count} portafolio{'s' if _pf_count != 1 else ''}</div>"
            f"<div style='font-size:0.68rem;color:#636366;font-family:DM Mono,monospace;"
            f"margin-top:2px;'>ID: {_gist_id_short}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if _needs_secrets_tip:
            st.sidebar.markdown(
                "<div style='background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.18);"
                "border-radius:8px;padding:8px 10px;margin-top:6px;font-size:0.68rem;"
                "color:#ffd60a;font-family:DM Mono,monospace;'>"
                "⚠️ <b>Para no volver a perder datos:</b><br>"
                "Agrega en Streamlit Secrets:<br>"
                "<code style='color:#aeaeb2;font-size:0.65rem;'>GITHUB_TOKEN = \"ghp_...\"</code><br>"
                f"<code style='color:#aeaeb2;font-size:0.65rem;'>GITHUB_GIST_ID = \"{_gh_gist_id}\"</code>"
                "</div>",
                unsafe_allow_html=True,
            )
        _sync_n = st.session_state.pop("_gist_sync_count", 0)
        if _sync_n:
            st.sidebar.success(f"☁️ {_sync_n} portafolio(s) restaurado(s) desde GitHub")
        if st.sidebar.button("🗑 Cambiar token GitHub", key="_del_gh",
                             use_container_width=True):
            _write_credential(_GITHUB_TOKEN_FILE, "")
            _write_credential(_GITHUB_GIST_ID_FILE, "")
            st.session_state.pop("_gist_synced", None)
            st.rerun()
    elif _gh_token and not _gh_gist_id:
        # Token OK pero sin Gist ID — buscar uno existente primero antes de crear
        with st.sidebar:
            with st.spinner("Buscando Gist existente..."):
                _found_id = _gist_find_existing()
            if _found_id:
                st.session_state.pop("_gist_synced", None)
                st.rerun()
            else:
                with st.spinner("Creando Gist nuevo..."):
                    _new_id = _gist_create()
                if _new_id:
                    st.session_state.pop("_gist_synced", None)
                    st.rerun()
                else:
                    st.error("No se pudo crear el Gist. Verifica el token.")
    else:
        st.sidebar.markdown(
            "<div style='background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.18);"
            "border-radius:10px;padding:10px 12px;font-size:0.78rem;color:#ffd60a;"
            "font-family:DM Mono,monospace;'>"
            "Sin GitHub — los datos se pierden al reiniciar.<br>"
            "<span style='color:#8e8e93;font-size:0.7rem;'>Agrega tu token para persistencia automática.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.sidebar.expander("⚙️ Configurar GitHub", expanded=False):
            _gh_input = st.text_input(
                "Personal Access Token:",
                placeholder="ghp_xxxxxxxxxxxx",
                type="password",
                key="_gh_tok_in",
                help="Necesita scope: gist",
            )
            if st.button("💾 Guardar y conectar", key="_save_gh",
                         use_container_width=True):
                _t = _gh_input.strip()
                if _t.startswith("ghp_") or _t.startswith("github_pat_"):
                    _write_credential(_GITHUB_TOKEN_FILE, _t)
                    st.session_state.pop("_gist_synced", None)
                    st.success("✓ Token guardado — creando Gist...")
                    st.rerun()
                else:
                    st.error("El token debe empezar con ghp_ o github_pat_")
            st.caption("👉 github.com → Settings → Developer settings → Personal access tokens → scope: **gist**")

    st.sidebar.divider()

    # ── Tasa libre de riesgo — CETES 28D (Banxico) ───────────
    st.sidebar.markdown("#### 🏦 Banxico · CETES 28D")
    _bnx_token = _load_banxico_token() or st.session_state.get("banxico_token", "")
    _bnx_from_secrets = bool(_load_banxico_token())

    if _bnx_token:
        _cetes_rate, _cetes_fecha = fetch_cetes_rate(_bnx_token)
        st.session_state["rf_rate"] = _cetes_rate
        _rate_ok = _cetes_fecha != "fallback"
        if _rate_ok:
            st.sidebar.markdown(
                f"<div style='background:rgba(48,209,88,0.07);border:1px solid "
                f"rgba(48,209,88,0.2);border-radius:10px;padding:10px 12px;'>"
                f"<div style='font-size:0.6rem;color:#8e8e93;font-family:DM Mono,monospace;"
                f"text-transform:uppercase;letter-spacing:.8px;'>CETES 28D · {_cetes_fecha}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:#30d158;"
                f"font-family:DM Mono,monospace;letter-spacing:-1px;'>{_cetes_rate*100:.2f}%</div>"
                f"<div style='font-size:0.68rem;color:#8e8e93;font-family:DM Mono,monospace;"
                f"margin-top:2px;'>{'via Secrets' if _bnx_from_secrets else 'Tasa libre de riesgo'} · Banxico SIE</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.warning(f"Banxico no respondió — usando {_cetes_rate*100:.1f}% por defecto.")
        # Solo mostrar botón de cambio si NO viene de Secrets
        if not _bnx_from_secrets:
            if st.sidebar.button("🗑 Cambiar token Banxico", key="_del_bnx",
                                 use_container_width=True):
                st.session_state["banxico_token"] = ""
                _write_credential(_BANXICO_TOKEN_FILE, "")
                st.rerun()
    else:
        st.session_state["rf_rate"] = 0.09
        st.sidebar.markdown(
            "<div style='background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.18);"
            "border-radius:10px;padding:10px 12px;font-size:0.78rem;color:#ffd60a;"
            "font-family:DM Mono,monospace;'>"
            "Sin token Banxico — usando 9.0% por defecto.<br>"
            "<span style='color:#8e8e93;font-size:0.7rem;'>Agrega BANXICO_TOKEN en Streamlit Secrets.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Input solo para uso local
        with st.sidebar.expander("⚙️ Configurar token (local)", expanded=False):
            _bnx_input = st.text_input(
                "Token Banxico SIE:",
                placeholder="Pega tu token aquí…",
                type="password",
                key="_bnx_raw",
            )
            if st.button("💾 Guardar", key="_save_bnx", use_container_width=True):
                _tok = _bnx_input.strip()
                if len(_tok) > 10:
                    st.session_state["banxico_token"] = _tok
                    _write_credential(_BANXICO_TOKEN_FILE, _tok)
                    fetch_cetes_rate.clear()
                    st.success("✓ Token guardado")
                    st.rerun()
                else:
                    st.error("Token demasiado corto.")
            st.caption("👉 banxico.org.mx/SieAPIRest → Obtener token (gratis)")

    fx = get_usd_mxn()
    st.sidebar.caption(f"USD/MXN: ${fx:,.2f}")
    st.sidebar.divider()

    # Info del portafolio cargado
    pname = st.session_state.get("portfolio_name", "")
    n_pos = len(holdings_df())
    bench = st.session_state.get("benchmark", "SPY")

    if pname:
        st.sidebar.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:12px 14px;margin-top:4px;">
  <div style="font-size:0.58rem;font-weight:700;letter-spacing:1.5px;color:#636366;
              text-transform:uppercase;font-family:DM Mono,monospace;margin-bottom:8px;">
    PORTAFOLIO ACTIVO</div>
  <div style="font-family:DM Mono,monospace;font-weight:800;font-size:0.95rem;
              color:#ffffff;margin-bottom:6px;">📁 {pname}</div>
  <div style="display:flex;justify-content:space-between;margin-top:2px;">
    <span style="font-size:0.75rem;color:#8e8e93;font-family:DM Mono,monospace;">
      {n_pos} posiciones</span>
    <span style="font-size:0.75rem;color:#8e8e93;font-family:DM Mono,monospace;">
      vs {bench}</span>
  </div>
</div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
<div style="background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.2);
            border-radius:10px;padding:10px 12px;font-size:0.8rem;color:#ffd60a;
            font-family:DM Mono,monospace;">
  Sin portafolio cargado.<br>
  <span style="color:#8e8e93;font-size:0.72rem;">
    Ve a <b style="color:#aeaeb2;">💼 Portfolio</b> para empezar.</span>
</div>
        """, unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-size:0.65rem;color:#48484a;font-family:DM Mono,monospace;"
        "text-align:center;'>v2.0 · Portfolio Manager</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# MARKET OVERVIEW TAB
# ─────────────────────────────────────────────────────────────

def tab_market_overview() -> None:
    """Vista rápida: índices + sector ETFs — estilo Market Pulse cards."""

    def _mkt_spark(spark: list, chg: float) -> str:
        if len(spark) < 4:
            return ""
        w, h = 180, 48
        lo, hi = min(spark), max(spark)
        rng = (hi - lo) if hi != lo else 1.0
        pts = " ".join(
            f"{i*(w/(len(spark)-1)):.1f},{h-(v-lo)/rng*(h-4):.1f}"
            for i, v in enumerate(spark)
        )
        clr = "#30d158" if chg >= 0 else "#ff453a"
        fid = f"mf{abs(hash(str(round(spark[0],2))))%9999}"
        last_y = f"{h-(spark[-1]-lo)/rng*(h-4):.1f}"
        return (
            f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="none" style="display:block;margin:8px 0 4px;">'
            f'<defs><linearGradient id="{fid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{clr}" stop-opacity=".20"/>'
            f'<stop offset="100%" stop-color="{clr}" stop-opacity="0"/>'
            f'</linearGradient></defs>'
            f'<polyline points="0,{h} {pts} {w},{h}" fill="url(#{fid})" stroke="none"/>'
            f'<polyline points="{pts}" fill="none" stroke="{clr}" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{w}" cy="{last_y}" r="3" fill="{clr}" opacity=".9"/>'
            f'</svg>'
        )

    def _mkt_card(t: str, pd_: dict, spark_override: list | None = None,
                  range_chg: float | None = None, range_label: str = "") -> str:
        price = pd_.get("price", 0.0)
        pv    = pd_.get("prev_close", 0.0)
        chg_p = (price - pv) / pv if pv > 0 else 0.0
        chg_o = pd_.get("change_vs_open", 0.0) or 0.0
        has_o = pd_.get("day_open", 0) > 0
        chg   = chg_o if has_o else chg_p
        rsi   = pd_.get("rsi14")
        sma   = pd_.get("above_sma20")
        vr    = pd_.get("vol_ratio")
        chg1w = pd_.get("change_1w")
        # spark_override: lista de precios del período seleccionado
        spark = spark_override if spark_override is not None else pd_.get("spark_prices", [])
        # Para sparkline de período usar el cambio del rango, no intraday
        spark_chg = range_chg if range_chg is not None else chg

        clr = "#30d158" if chg >= 0 else "#ff453a"
        bg  = "rgba(48,209,88,0.04)" if chg >= 0 else "rgba(255,69,58,0.04)"
        brd = "rgba(48,209,88,0.15)" if chg >= 0 else "rgba(255,69,58,0.15)"
        pc  = "#30d158" if chg_p >= 0 else "#ff453a"
        pa  = "▲" if chg_p >= 0 else "▼"

        spark_svg = _mkt_spark(spark, spark_chg)

        rsi_b = ""
        if rsi is not None:
            rc = "#ff453a" if rsi>=70 else "#30d158" if rsi<=30 else "#8e8e93"
            rt = f"RSI {rsi:.0f} {'↑ OC' if rsi>=70 else '↓ OS' if rsi<=30 else ''}"
            rsi_b = (f"<span style='background:rgba(142,142,147,.1);color:{rc};"
                     f"border:1px solid {rc}22;border-radius:5px;"
                     f"padding:1px 7px;font-size:0.65rem;font-family:DM Mono,monospace;"
                     f"font-weight:700;'>{rt}</span>")

        sma_t = ("<span style='color:#86efac;font-size:0.65rem;'>↑ SMA20</span>" if sma is True
                 else "<span style='color:#ff453a;font-size:0.65rem;'>↓ SMA20</span>" if sma is False
                 else "")

        w1_t = ""
        if chg1w is not None:
            wc = "#30d158" if chg1w >= 0 else "#ff453a"
            w1_t = (f"<span style='color:{wc};font-size:0.65rem;"
                    f"font-family:DM Mono,monospace;'>1S: {chg1w:+.1%}</span>")

        vol_t = ""
        if vr is not None and (vr >= 1.5 or vr <= 0.5):
            vc = "#ffd60a" if vr >= 1.5 else "#8e8e93"
            vol_t = (f"<span style='color:{vc};font-size:0.65rem;font-family:DM Mono,monospace;"
                     f"background:rgba(255,255,255,0.04);border-radius:5px;"
                     f"padding:1px 6px;'>Vol ×{vr:.1f}</span>")

        open_r = ""
        if has_o:
            oc = "#30d158" if chg_o >= 0 else "#ff453a"
            oa = "▲" if chg_o >= 0 else "▼"
            open_r = (f"<div style='font-size:0.62rem;color:{oc};font-family:DM Mono,monospace;"
                      f"margin-top:1px;opacity:.8;'>{oa} {abs(chg_o):.2%} "
                      f"<span style='color:#636366;font-size:.58rem;'>apertura</span></div>")

        # Cambio de período en esquina superior derecha si hay rango seleccionado
        if range_chg is not None and range_label:
            rc2 = "#30d158" if range_chg >= 0 else "#ff453a"
            ra2 = "▲" if range_chg >= 0 else "▼"
            range_badge = (
                f'<div style="font-family:DM Mono,monospace;font-size:1.3rem;font-weight:700;'
                f'color:{rc2};line-height:1;">{ra2} {abs(range_chg):.2%}</div>'
                f'<div style="font-size:0.6rem;color:#636366;margin-top:1px;">{range_label}</div>'
            )
        else:
            range_badge = (
                f'<div style="font-family:DM Mono,monospace;font-size:1.3rem;font-weight:700;'
                f'color:{pc};line-height:1;">{pa} {abs(chg_p):.2%}</div>'
                f'<div style="font-size:0.6rem;color:#636366;margin-top:1px;">vs cierre anterior</div>'
                f'{open_r}'
            )
        return (
            f'<div class="mp-card" style="background:{bg};border:1px solid {brd};'
            f'border-radius:18px;padding:16px 18px 14px;display:flex;flex-direction:column;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'<div>'
            f'<div style="font-family:DM Mono,monospace;font-weight:900;font-size:0.95rem;'
            f'color:#fff;line-height:1;">{t}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:1.5rem;font-weight:700;'
            f'color:#fff;line-height:1.1;margin-top:3px;letter-spacing:-1px;">${price:,.2f}</div>'
            f'</div>'
            f'<div style="text-align:right;">{range_badge}</div></div>'
            f'{spark_svg}'
            f'<div style="height:1px;background:rgba(255,255,255,0.05);margin:4px 0;"></div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:4px;">'
            f'{rsi_b}{w1_t}{sma_t}{vol_t}</div></div>'
        )

    MARKET_GROUPS = {
        "Índices":     ["SPY", "QQQ", "IWM", "DIA"],
        "Renta Fija":  ["TLT", "AGG", "HYG"],
        "Commodities": ["GLD", "SLV", "USO"],
        "Volatilidad": ["VXX"],
    }
    SECTOR_TICKERS = ["XLK","XLF","XLV","XLY","XLI","XLE","XLU","XLRE","XLB"]
    SECTOR_LABELS  = {
        "XLK":"Tech","XLF":"Finanzas","XLV":"Salud","XLY":"Consumo",
        "XLI":"Industrial","XLE":"Energía","XLU":"Utilities",
        "XLRE":"Real Estate","XLB":"Materials",
    }
    _all_mkt = tuple(sorted(
        {t for g in MARKET_GROUPS.values() for t in g} | set(SECTOR_TICKERS)
    ))

    # ── Selector de rango ─────────────────────────────────────
    _RANGE_OPTS  = ["1D", "1M", "6M", "YTD", "1A"]
    _RANGE_LABEL = {"1D": "hoy", "1M": "1 mes", "6M": "6 meses", "YTD": "año actual", "1A": "1 año"}
    _range_sel = st.session_state.get("mkt_range", "1D")
    _rc1, _rc2, _rc3, _rc4, _rc5, _ = st.columns([1, 1, 1, 1, 1, 5])
    for _rcol, _ropt in zip([_rc1, _rc2, _rc3, _rc4, _rc5], _RANGE_OPTS):
        with _rcol:
            _is_active = (_range_sel == _ropt)
            _btn_style = (
                "background:#0a84ff;color:#fff;border:none;border-radius:20px;"
                "padding:4px 14px;font-family:DM Mono,monospace;font-size:0.75rem;"
                "font-weight:700;cursor:pointer;width:100%;"
            ) if _is_active else (
                "background:rgba(255,255,255,0.06);color:#8e8e93;border:1px solid rgba(255,255,255,0.1);"
                "border-radius:20px;padding:4px 14px;font-family:DM Mono,monospace;"
                "font-size:0.75rem;cursor:pointer;width:100%;"
            )
            if st.button(_ropt, key=f"_mkt_range_{_ropt}",
                         use_container_width=True,
                         type="primary" if _is_active else "secondary"):
                st.session_state["mkt_range"] = _ropt
                st.rerun()

    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    with st.spinner("Cargando mercados…"):
        mkt_pulse = fetch_pulse_data(_all_mkt)

    # Datos históricos del rango si no es 1D
    _range_df    = pd.DataFrame()
    _spark_cache: dict[str, list] = {}
    _range_ret:   dict[str, float] = {}
    _rlabel = _RANGE_LABEL.get(_range_sel, "")
    if _range_sel != "1D":
        with st.spinner(f"Cargando histórico {_rlabel}…"):
            _range_df = fetch_range_data(_all_mkt, _range_sel)
        if not _range_df.empty:
            for _t in _all_mkt:
                if _t in _range_df.columns:
                    _ser = _range_df[_t].dropna()
                    if len(_ser) >= 2:
                        _spark_cache[_t] = _ser.tolist()
                        _range_ret[_t]   = float((_ser.iloc[-1] - _ser.iloc[0]) / _ser.iloc[0])

    section("MERCADO HOY" if _range_sel == "1D" else f"MERCADO · {_range_sel}")
    for gname, gtickers in MARKET_GROUPS.items():
        st.markdown(
            f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
            f"color:#636366;text-transform:uppercase;font-family:DM Mono,monospace;"
            f"margin:14px 0 8px;'>{gname}</div>",
            unsafe_allow_html=True,
        )
        _gcols = st.columns(len(gtickers), gap="small")
        for _gc, _gt in zip(_gcols, gtickers):
            with _gc:
                _spark_ov  = _spark_cache.get(_gt) if _range_sel != "1D" else None
                _range_chg = _range_ret.get(_gt)   if _range_sel != "1D" else None
                st.markdown(
                    _mkt_card(_gt, mkt_pulse.get(_gt, {}),
                              spark_override=_spark_ov,
                              range_chg=_range_chg,
                              range_label=_rlabel),
                    unsafe_allow_html=True,
                )

    # ── Gráfica comparativa de rendimiento por período ─────────
    if _range_sel != "1D" and not _range_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section(f"RENDIMIENTO COMPARADO · {_range_sel}")

        # Colores por ticker (palette Apple)
        _COLORS = ["#0a84ff","#30d158","#bf5af2","#ffd60a","#ff453a",
                   "#ff9f0a","#64d2ff","#ff375f","#30d158","#5e5ce6"]

        # Selector de grupo a mostrar en la gráfica
        _chart_groups = {g: ts for g, ts in MARKET_GROUPS.items()}
        _chart_groups["Sectores"] = SECTOR_TICKERS
        _grp_opts = list(_chart_groups.keys())
        _grp_sel  = st.session_state.get("mkt_chart_group", "Índices")
        if _grp_sel not in _grp_opts:
            _grp_sel = _grp_opts[0]

        _gc1, _gc2, _gc3, _gc4, _gc5, _ = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 3])
        for _gcol2, _gopt in zip([_gc1, _gc2, _gc3, _gc4, _gc5], _grp_opts):
            with _gcol2:
                if st.button(_gopt, key=f"_mkt_grp_{_gopt}", use_container_width=True,
                             type="primary" if _grp_sel == _gopt else "secondary"):
                    st.session_state["mkt_chart_group"] = _gopt
                    st.rerun()

        _chart_tickers = _chart_groups.get(_grp_sel, [])
        _fig_rng = go.Figure()
        _ci = 0
        for _ct in _chart_tickers:
            if _ct not in _range_df.columns:
                continue
            _ser = _range_df[_ct].dropna()
            if len(_ser) < 2:
                continue
            _norm = (_ser / _ser.iloc[0] - 1) * 100  # % vs primer día
            _clr  = _COLORS[_ci % len(_COLORS)]
            _ci  += 1
            _fig_rng.add_trace(go.Scatter(
                x=_norm.index, y=_norm.values,
                name=_ct, mode="lines",
                line=dict(color=_clr, width=2),
                hovertemplate=f"<b>{_ct}</b>: %{{y:+.2f}}%<extra></extra>",
            ))
        # Línea de cero
        _fig_rng.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        _fig_rng.update_layout(
            **_pl("xaxis", "yaxis"),
            height=340,
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", showgrid=True,
                zerolinecolor="rgba(255,255,255,0.08)",
                tickformat="%b %d" if _range_sel in ("1M",) else "%b '%y",
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", showgrid=True,
                zerolinecolor="rgba(255,255,255,0.08)",
                ticksuffix="%",
            ),
            legend=dict(
                bgcolor="rgba(28,28,30,0.85)", bordercolor="rgba(255,255,255,0.08)",
                borderwidth=1, font=dict(color="#aeaeb2", size=11),
                orientation="h", y=-0.18,
            ),
        )
        st.plotly_chart(_fig_rng, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    section("SECTORES S&P 500")
    _seccols = st.columns(3, gap="small")
    for _si, _stk in enumerate(SECTOR_TICKERS):
        _sd  = mkt_pulse.get(_stk, {})
        _sp  = _sd.get("price", 0.0)
        _spv = _sd.get("prev_close", 0.0)
        # Usar retorno del período si hay rango activo
        if _range_sel != "1D" and _stk in _range_ret:
            _sch = _range_ret[_stk]
            _sec_sub = f"<span style='color:#48484a;font-size:0.62rem;'> {_range_sel}</span>"
        else:
            _sch = (_sp - _spv) / _spv if _spv > 0 else 0.0
            _sec_sub = ""
        _rgb = "48,209,88" if _sch >= 0 else "255,69,58"
        _sc  = "#30d158" if _sch >= 0 else "#ff453a"
        _sar = "▲" if _sch >= 0 else "▼"
        with _seccols[_si % 3]:
            st.markdown(
                f"<div style='background:rgba({_rgb},0.05);border:1px solid rgba({_rgb},0.18);"
                f"border-radius:10px;padding:10px 14px;margin-bottom:8px;"
                f"display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:0.8rem;color:#aeaeb2;'>{SECTOR_LABELS[_stk]}"
                f"<span style='color:#48484a;font-size:0.7rem;'> {_stk}</span></span>"
                f"<span style='font-family:DM Mono,monospace;font-weight:700;color:{_sc};'>"
                f"{_sar} {abs(_sch):.2%}{_sec_sub}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Watchlist ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("WATCHLIST")

    # Input para agregar tickers
    _wl_c1, _wl_c2 = st.columns([3, 1])
    with _wl_c1:
        _wl_new = st.text_input("Agregar ticker:", placeholder="AAPL, MSFT, GOOGL…",
                                 key="_wl_add_input", label_visibility="collapsed")
    with _wl_c2:
        if st.button("➕ Agregar", key="_wl_add_btn", use_container_width=True):
            _wl_items = [t.strip().upper() for t in _wl_new.replace(",", " ").split() if t.strip()]
            _wl_curr  = st.session_state.get("watchlist", [])
            _added = 0
            for _wt in _wl_items:
                if _wt and _wt not in _wl_curr:
                    _wl_curr.append(_wt)
                    _added += 1
            st.session_state["watchlist"] = _wl_curr
            if _added:
                # Persistir en portafolio actual
                _wl_pname = st.session_state.get("portfolio_name", "")
                if _wl_pname:
                    save_portfolio(_wl_pname,
                                   st.session_state.get("transactions", []),
                                   st.session_state.get("target_weights", {}),
                                   st.session_state.get("benchmark", "SPY"))
                st.rerun()

    _watchlist = st.session_state.get("watchlist", [])

    if _watchlist:
        with st.spinner("Cargando watchlist…"):
            _wl_prices = fetch_live_prices(tuple(sorted(_watchlist)))

        _wl_cols = st.columns(min(len(_watchlist), 4))
        _to_remove = []
        for _wi, _wt in enumerate(_watchlist):
            _winf  = _wl_prices.get(_wt, {})
            _wpx   = _winf.get("price",      0.0)
            _wpv   = _winf.get("prev_close", 0.0)
            _wchg  = (_wpx - _wpv) / _wpv if _wpv > 0 else 0.0
            _wclr  = "#30d158" if _wchg >= 0 else "#ff453a"
            _warr  = "▲" if _wchg >= 0 else "▼"
            with _wl_cols[_wi % 4]:
                st.markdown(
                    f"<div style='background:#000;border:1px solid rgba(255,255,255,0.08);"
                    f"border-radius:14px;padding:14px;margin-bottom:8px;'>"
                    f"<div style='font-family:DM Mono,monospace;font-weight:700;"
                    f"font-size:0.88rem;color:#fff;margin-bottom:4px;'>{_wt}</div>"
                    f"<div style='font-family:DM Mono,monospace;font-size:1.3rem;"
                    f"font-weight:700;color:#fff;'>${_wpx:,.2f}</div>"
                    f"<div style='font-family:DM Mono,monospace;font-size:0.8rem;"
                    f"color:{_wclr};margin-top:2px;'>{_warr} {abs(_wchg):.2%}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("✕", key=f"_wl_rm_{_wt}", use_container_width=True):
                    _to_remove.append(_wt)

        if _to_remove:
            st.session_state["watchlist"] = [t for t in _watchlist if t not in _to_remove]
            _wl_pname = st.session_state.get("portfolio_name", "")
            if _wl_pname:
                save_portfolio(_wl_pname,
                               st.session_state.get("transactions", []),
                               st.session_state.get("target_weights", {}),
                               st.session_state.get("benchmark", "SPY"))
            st.rerun()
    else:
        st.markdown(
            "<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);"
            "border-radius:12px;padding:16px;text-align:center;color:#48484a;"
            "font-size:0.82rem;font-family:DM Mono,monospace;'>"
            "Tu watchlist está vacía. Agrega tickers arriba para monitorearlos.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    init_state()
    sidebar()

    # ── Onboarding cuando no hay posiciones ──────────────────
    hdf_check     = holdings_df()
    has_portfolio = not hdf_check.empty

    if not has_portfolio:
        show_onboarding()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.62rem;font-weight:700;letter-spacing:2px;"
            "color:#636366;text-transform:uppercase;font-family:DM Mono,monospace;"
            "margin-bottom:12px;'>EMPEZAR — AÑADE TU PRIMERA POSICIÓN</div>",
            unsafe_allow_html=True)
        tab_editor()
        return

    # ── Navegación reorganizada ────────────────────────────
    tabs = st.tabs([
        "📡 Dashboard",      # Monitor
        "🌍 Mercado",        # Monitor
        "🧬 Por Acción",     # Análisis
        "📈 Performance",    # Análisis
        "🔍 Técnico",        # Análisis
        "💼 Portfolio",      # Gestión
        "⚖️ Rebalanceo",    # Gestión
        "📝 Tesis",          # Gestión
        "🔔 Alertas",        # Gestión
    ])

    with tabs[0]: tab_dashboard()
    with tabs[1]: tab_market_overview()
    with tabs[2]: tab_stock_deep_dive()
    with tabs[3]: tab_performance()
    with tabs[4]: tab_analytics()
    with tabs[5]: tab_editor()
    with tabs[6]: tab_rebalance()
    with tabs[7]: tab_thesis()
    with tabs[8]: tab_alerts()


if __name__ == "__main__":
    main()

    ########    python -m streamlit run portfolio_manager.py