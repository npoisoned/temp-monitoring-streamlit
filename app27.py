import time
import asyncio

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import requests
import aiohttp
from joblib import Parallel, delayed


def season_from_month(m):
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def normalize_season(x):
    if not isinstance(x, str):
        return x
    s = x.strip().lower()
    ru = {"зима": "winter", "весна": "spring", "лето": "summer", "осень": "autumn"}
    return ru.get(s, s)


def get_weather_sync(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    return r.status_code, r.json()


async def get_weather_async(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return resp.status, data


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    if not {"city", "timestamp", "temperature"}.issubset(df.columns):
        raise ValueError("Нужны колонки: city, timestamp, temperature (season опционально)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature"])

    df["city"] = df["city"].astype(str).str.strip()

    if "season" not in df.columns:
        df["season"] = df["timestamp"].dt.month.map(season_from_month)
    else:
        df["season"] = df["season"].apply(normalize_season)
        miss = df["season"].isna()
        if miss.any():
            df.loc[miss, "season"] = df.loc[miss, "timestamp"].dt.month.map(season_from_month)

    return df


def analyze_city(df_city, window=30):
    d = df_city.sort_values("timestamp").copy()

    d["roll_mean"] = d["temperature"].rolling(window, min_periods=10).mean()
    d["roll_std"] = d["temperature"].rolling(window, min_periods=10).std(ddof=0)

    d["upper"] = d["roll_mean"] + 2 * d["roll_std"]
    d["lower"] = d["roll_mean"] - 2 * d["roll_std"]

    d["is_anomaly"] = False
    ok = d["roll_std"].notna()
    d.loc[ok, "is_anomaly"] = (d.loc[ok, "temperature"] > d.loc[ok, "upper"]) | (d.loc[ok, "temperature"] < d.loc[ok, "lower"])
    return d


st.set_page_config(page_title="Температуры", layout="wide")
st.title("Температуры: сезонность, аномалии, тренд и текущая погода")

with st.sidebar:
    st.header("Параметры")
    file = st.file_uploader("Загрузите temperature_data.csv", type=["csv"])

    api_key = st.text_input("Ключ OpenWeatherMap", type="password")

    mode = st.selectbox("Способ запроса погоды", ["Обычный", "Асинхронный"], index=0)

    st.divider()
    st.subheader("Ускорение расчётов")
    bench_btn = st.button("Сравнить скорость: обычный и параллельный расчёт")

if not file:
    st.info("Загрузите CSV файл, чтобы продолжить.")
    st.stop()

try:
    df = load_data(file)
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

cities = sorted(df["city"].unique())
city = st.selectbox("Город", cities)

if bench_btn:
    st.write("Сравнение скорости расчёта по всем городам:")

    t0 = time.perf_counter()
    parts = []
    for c, g in df.groupby("city", sort=False):
        parts.append(analyze_city(g))
    _ = pd.concat(parts, ignore_index=True)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    groups = [g for _, g in df.groupby("city", sort=False)]
    try:
        res = Parallel(n_jobs=-1)(delayed(analyze_city)(g) for g in groups)
        _ = pd.concat(res, ignore_index=True)
        t_par = time.perf_counter() - t0
    except Exception as e:
        t_par = None
        st.warning(f"Параллельный расчёт не запустился: {e}")

    c1, c2 = st.columns(2)
    c1.metric("Обычный расчёт", f"{t_seq:.3f} сек")
    c2.metric("Параллельный расчёт", f"{t_par:.3f} сек" if t_par is not None else "н/д")

    st.caption("На небольшом объёме данных параллельный режим может быть медленнее из-за накладных расходов.")

df_city = df[df["city"] == city].copy()
df_city = analyze_city(df_city, window=30)

season_stats = (
    df_city.groupby("season", as_index=False)["temperature"]
    .agg(mean="mean", std="std", min="min", max="max", count="count")
)
season_stats["std"] = season_stats["std"].fillna(0.0)

st.subheader(f"{city}. Описательная статистика")
a, b, c, d = st.columns(4)
a.metric("Количество дней", f"{len(df_city):,}")
b.metric("Средняя температура", f"{df_city['temperature'].mean():.2f} °C")
c.metric("Минимум", f"{df_city['temperature'].min():.2f} °C")
d.metric("Максимум", f"{df_city['temperature'].max():.2f} °C")

st.dataframe(season_stats, use_container_width=True, hide_index=True)

st.subheader("Временной ряд и аномалии (окно 30 дней)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_city["timestamp"], y=df_city["temperature"], mode="lines", name="Температура"))
fig.add_trace(go.Scatter(x=df_city["timestamp"], y=df_city["roll_mean"], mode="lines", name="Скользящее среднее"))
fig.add_trace(go.Scatter(x=df_city["timestamp"], y=df_city["upper"], mode="lines", name="Верхняя граница", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df_city["timestamp"], y=df_city["lower"], mode="lines", name="Нижняя граница", line=dict(dash="dot")))

anom = df_city[df_city["is_anomaly"]]
fig.add_trace(go.Scatter(x=anom["timestamp"], y=anom["temperature"], mode="markers", name="Аномалии"))

fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Сезонные показатели (среднее и разброс)")
tmp = season_stats.copy()
tmp["up"] = tmp["mean"] + tmp["std"]
tmp["lo"] = tmp["mean"] - tmp["std"]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=tmp["season"], y=tmp["mean"], name="Среднее"))
fig2.add_trace(go.Scatter(x=tmp["season"], y=tmp["up"], mode="lines+markers", name="Плюс 1 стандартное отклонение"))
fig2.add_trace(go.Scatter(x=tmp["season"], y=tmp["lo"], mode="lines+markers", name="Минус 1 стандартное отклонение"))
fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Долгосрочный тренд (месячные средние)")
m = (
    df_city.sort_values("timestamp")
    .set_index("timestamp")["temperature"]
    .resample("MS").mean()
    .to_frame("temp_m")
    .reset_index()
)

x = (m["timestamp"] - m["timestamp"].min()).dt.days.astype(float).to_numpy()
y = m["temp_m"].to_numpy()
mask = np.isfinite(x) & np.isfinite(y)

if mask.sum() >= 2:
    k, b0 = np.polyfit(x[mask], y[mask], 1)
    m["trend"] = k * x + b0
else:
    m["trend"] = np.nan

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=m["timestamp"], y=m["temp_m"], mode="lines+markers", name="Месячная средняя"))
fig3.add_trace(go.Scatter(x=m["timestamp"], y=m["trend"], mode="lines", name="Линейный тренд"))
fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Текущая температура и проверка по норме сезона")
today = pd.Timestamp.today()
cur_season = season_from_month(int(today.month))

if not api_key:
    st.info("Введите ключ, чтобы получить текущую температуру.")
else:
    row = season_stats[season_stats["season"] == cur_season]
    if row.empty:
        st.warning(f"Для города {city} нет данных по сезону: {cur_season}.")
    else:
        mean_s = float(row["mean"].iloc[0])
        std_s = float(row["std"].iloc[0])
        low = mean_s - 2 * std_s
        high = mean_s + 2 * std_s

        with st.spinner("Получение данных о погоде..."):
            t0 = time.perf_counter()
            if mode == "Обычный":
                code, payload = get_weather_sync(city, api_key)
            else:
                code, payload = run_async(get_weather_async(city, api_key))
            ms = (time.perf_counter() - t0) * 1000

        if isinstance(payload, dict) and str(payload.get("cod")) == "401":
            st.error(payload)
        elif code != 200:
            st.error(f"Ошибка запроса: HTTP {code}")
            st.code(payload)
        else:
            temp_now = float(payload["main"]["temp"])
            ok = (low <= temp_now <= high) if std_s > 0 else (temp_now == mean_s)

            c1, c2, c3 = st.columns(3)
            c1.metric("Текущая температура", f"{temp_now:.2f} °C")
            c2.metric("Сезон", cur_season)
            c3.metric("Время запроса", f"{ms:.0f} мс")

            st.write(f"Нормальный диапазон для сезона {cur_season}: [{low:.2f}; {high:.2f}] °C (по историческим данным).")
            if ok:
                st.success("Текущая температура в пределах нормы.")
            else:
                st.warning("Текущая температура выходит за пределы нормы.")

with st.expander("Распределение температур по сезонам"):
    fig4 = px.box(df_city, x="season", y="temperature", points="outliers")
    fig4.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)