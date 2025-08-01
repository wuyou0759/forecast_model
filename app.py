# app.py
import streamlit as st
import altair as alt
import pandas as pd
from model import run_pipeline, THRESHOLD, EVENT_WINDOW, series_info

st.set_page_config(page_title="Forecast Model", layout="wide")
st.title("Interactive Forecast Model")

#hardcoded data
SCENARIOS = {
    "Competitive Coexistence": {"driver_1": -0.5, "driver_2": 0.5, "driver_3": 0.5, "driver_4": 0.5},
    "Separate Silos": {"driver_1": -1.0, "driver_2": 1.5, "driver_3": 1.5, "driver_4": 1.5},
    "Renaissance of Democracies": {"driver_1": -0.5, "driver_2": -0.5, "driver_3": -1.0, "driver_4": -0.5},
}

FORECAST_DESCRIPTIONS = {
    "forecast_6": "the U.S. effective tariff rate on China will average over 25% from 2026–2030",
    "forecast_7": "U.S. reshoring and FDI job announcements in 2026 will NOT exceed the 2022 record level",
    "forecast_8": "NASA’s budget will exceed 0.50% of the U.S. federal budget in 2030",
    "forecast_9": "U.S. industrial production for defense and space equipment will average above its all-time high across 2028–2032",
    "forecast_10": "U.S. investment policy measures deemed 'less favourable to investors' will average above the elevated post-2020 level over 2026–2030",
    "forecast_11": "the global count of all active sanctions will exceed 1000 by the end of 2027",
    "forecast_12": "U.S. critical technology exports to China will average higher over 2026–2030 than it did during 2020–2024",
    "forecast_13": "the share of critical sectors in China's total BRI engagement will exceed its long-run average of 76% by 2033",
    "forecast_14": "annual requests for WTO dispute consultations will remain below their long-run average from 2026–2030",
    "forecast_15": "the Asian Infrastructure Investment Bank (AIIB) will surpass 105 members by its 15th anniversary in 2031",
    "forecast_16": "at least half of Americans will perceive foreign trade as an economic threat in 2032",
    "forecast_17": "Apple's annual market share in the Chinese Mobile, Tablet and Console market will average lower over 2026-2030 than it did during 2020-2024",
    "forecast_18": "the Internet Freedom Score for China will average at or below 9 from 2026-2030",
    "forecast_19": "the average annual F1 visa refusal rate for China from 2026-2030 will exceed the 2017-2023 historical average",
    "forecast_20": "the share of new H-1B visas granted to the Professional, Scientific, and Technical Services sector will exceed 50% by fiscal year 2030",
}

REMARKS = {
    "forecast_6": "",
    "forecast_7": "",
    "forecast_8": "When NASA’s budget share is expressed as a raw value (e.g. 0.005 = 0.5 %), the standard deviation is very small (10-³), causing the drivers' effects to be too small and rounded off to 0. Will be fixed in update by rescaling the forecast.",
    "forecast_9": "",
    "forecast_10": "",
    "forecast_11": 'The confidence intervals for this forecast is "poorly shaped" because the model assumes that ETS error is added additively (constant variance), while in this case, the forecast data increased exponentially from 2020 onwards. This means that earlier years will cause the mean-squared-error to be small, resulting in an error band that is far too tight relative to recent levels.',
    "forecast_12": 'Forecast 12 is the only forecast that generates an "unintuitive" result despite my "Theory-driven Model Selection" (see Section 3.3, "Model Methodology"). This is because for all possible candidate models (driver + lag combinations), none of the models satisfied the constraint that the probability for forecast 12 (U.S. critical technology exports to China) should theoretically be lower under "Separate Silos" and higher under “Renaissance of Democracies” as compared to “Competitive Coexistence”. As such, the model with the best AIC score was chosen instead. This is reflective of limitations in my model selection process, where it is not always possible to reconcile data-driven inference with theory-led constraints given the assumptions that were made (see Section 3.3.2, "Model Assumptions").',
    "forecast_13": "",
    "forecast_14": "",
    "forecast_15": "",
    "forecast_16": "",
    "forecast_17": "",
    "forecast_18": "",
    "forecast_19": 'Forecast 19 relies solely on ETS as its time series data was insufficient for meaningful regression analysis. This is because F19 only has 6 data points (see "F1 - F20 Forecast Data").',
    "forecast_20": "",
}

SPACE = "\u00A0"

FORECAST_TAGS = {
    "forecast_6": "",
    "forecast_7": "",
    "forecast_8": f"{SPACE*19}(Remarks)",   # ← 3 × em-spaces, then tag
    "forecast_9": "",
    "forecast_10": "",
    "forecast_11": f"{SPACE*16}(Remarks)",
    "forecast_12": f"{SPACE*16}(Remarks)",
    "forecast_13": "",
    "forecast_14": "",
    "forecast_15": "",
    "forecast_16": "",
    "forecast_17": "",
    "forecast_18": "",
    "forecast_19": f"{SPACE*16}(Remarks)",
    "forecast_20": "",
}

def fmt_forecast(key: str) -> str:
    tag = FORECAST_TAGS.get(key, "")
    return f"{key}{tag}" if tag else key

#sidebar
keys = list(THRESHOLD.keys())           
forecast_key = st.sidebar.selectbox(
    "Select forecast",
    list(THRESHOLD.keys()),
    format_func=fmt_forecast
)
scenario = st.sidebar.selectbox("Choose scenario", list(SCENARIOS.keys()) + ["Custom"])

if scenario == "Custom":
    st.sidebar.markdown("---")
    st.sidebar.write("Customize Driver Ramps (z-score targets by 2035)")
    targets = {d: st.sidebar.slider(f"{d}", -2.0, 2.0, 0.0, 0.1) for d in SCENARIOS["Competitive Coexistence"]}
else:
    targets = SCENARIOS[scenario]

if st.sidebar.button("▶️ Run Forecast", use_container_width=True):
    with st.spinner("Running forecast simulation..."):
        prob, sims, effects, fan = run_pipeline(forecast_key, targets, seed=42)
    
    INVERT_PROB = {"forecast_7", "forecast_14", "forecast_17", "forecast_18"}
    prob_display = 1 - prob if forecast_key in INVERT_PROB else prob
    
    #main page
    st.header(f"{forecast_key}")
    st.markdown(f"#### There is a **{prob_display:.1%}** chance that {FORECAST_DESCRIPTIONS.get(forecast_key, 'the event occurs')}.")
    
    #Build Altair Fan Chart
    chart_df = fan.join(effects[["baseline_level"]]).reset_index()
    future_start = series_info[forecast_key]["end"]

    #Base chart for layering
    base_chart = alt.Chart(chart_df).encode(x=alt.X('year:O', title="Year")).properties(width=700, height=400)

    #Layered areas for uncertainty bands
    #90 % band
    band_90 = base_chart.mark_area(opacity=0.20, color="#4c78a8").encode(
        y=alt.Y('p05:Q', title='Value'),
        y2='p95:Q',
        tooltip=[
            alt.Tooltip('year:O',  title='Year'),
            alt.Tooltip('p05:Q',   title='5th percentile',  format='.5f'),
            alt.Tooltip('p95:Q',   title='95th percentile', format='.5f')
        ]
    ).transform_filter(f"datum.year >= {future_start}")

    #50 % band
    band_50 = base_chart.mark_area(opacity=0.40, color="#4c78a8").encode(
        y='p25:Q',
        y2='p75:Q',
        tooltip=[
            alt.Tooltip('year:O',  title='Year'),
            alt.Tooltip('p25:Q',   title='25th percentile', format='.5f'),
            alt.Tooltip('p75:Q',   title='75th percentile', format='.5f')
        ]
    ).transform_filter(f"datum.year >= {future_start}")


    line_df = chart_df.melt(
        id_vars='year',
        value_vars=['p50', 'baseline_level'],
        var_name='Series',
        value_name='value'
    )
    line_df['Series'] = line_df['Series'].map({
        'p50': 'Adjusted prediction',
        'baseline_level': 'Baseline ETS prediction'
    })

    line_chart = (
        alt.Chart(line_df)
        .mark_line(size=2.5)
        .encode(
            x='year:O',
            y='value:Q',
            color=alt.Color(
                'Series:N',
                scale=alt.Scale(
                    domain=['Adjusted prediction', 'Baseline ETS prediction'],
                    range=['#e45756', '#54a24b']
                ),
                legend=alt.Legend(title='')
            ),
            strokeDash=alt.StrokeDash(
                'Series:N',
                scale=alt.Scale(
                    domain=['Adjusted prediction', 'Baseline ETS prediction'],
                    range=[[0], [5, 5]]
                ),
                legend=None
            )
        )
        .transform_filter(f"datum.year >= {future_start}")
    )

    hist = base_chart.mark_line(color="#666666").encode(
        y='baseline_level:Q'
    ).transform_filter(f"datum.year <= {future_start}")

    # Combine chart layers
    st.altair_chart(
        (hist + band_90 + band_50 + line_chart),
        use_container_width=True
    )
    #st.altair_chart(
    #    (band_90 + band_50 + median + baseline),
    #    use_container_width=True
    #)

    st.subheader("Driver Adjustment Details")
    # Extract and format the table
    table = effects.loc[2026:2035].copy()
    table.index = table.index.astype(str)
    table.index.name = "Year"

    # Custom column renaming logic
    def rename_column(col):
        if col.startswith("driver_") and "_L" in col:
            parts = col.split("_")
            driver_num = parts[1]
            lag = parts[2][1:]
            if parts[3] == "z":
                return f"Driver {driver_num}, Lag {lag}, Δ Z-Score"
            elif parts[3] == "effect":
                return f"Driver {driver_num}, Lag {lag}, Raw Adjustment"
        elif col == "baseline_level":
            return "ETS Baseline"
        elif col == "total_driver_effect":
            return "Total Driver Adjustment"
        elif col == "adjusted_level":
            return "Adjusted Level"
        return col

    #Apply renaming
    table.columns = [rename_column(col) for col in table.columns]

    #Display the renamed and rounded table
    st.dataframe(table.round(3), use_container_width=True)

    remark = REMARKS.get(forecast_key)
    if remark:
        st.markdown("#### Remarks")
        st.markdown(remark)

else:
    st.info("Select a forecast and scenario, then click 'Run Forecast' to begin.")
