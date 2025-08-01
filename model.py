# model.py
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

#data
driver_1 = [21.1328754776383,28.5601936492049,25.7845926473305,35.1325154549707,52.0410301439847,50.2665621014441,55.2374343812225,56.5146838433062,61.381362014625,58.9726494801639,62.2312433273391,65.8917606626368,66.6520703713157,72.3683947089178,72.1433279921242,68.413980818487,69.9597407824991,68.6202333784948,70.2065671780679,71.0590149062401,68.5614741945107,67.2527545050931,65.7757534710021,62.0102376029941,59.7643332052565,58.6403091845112,58.7728942427915,56.6875343157862,58.233237560013,61.3159082537406,60.0028271730544,59.0664552776805,63.4844994833622,61.6692689502803,55.2773623064643,53.8073183824442,55.382536457332,48.6379980120923,50.7785426634822,52.2623635747032]
driver_2 = [0,0,0,0,0,0,1.33333333333333,10.9166666666667,14.75,5,7.33333333333333,8.25,31.6666666666667,24.25,17,13.5,26.0833333333333,13.5833333333333,16.4166666666667,15.6666666666667,24.25,39.1428571428571]
driver_3 = [49.5703508175,55.2519307166667,56.5468198708333,57.0621623633333,74.0596690008333,71.07381682,79.7199218266667,73.3307311033333,73.3882596883333,59.1917649425,72.9339220816667,69.4717002983333,80.8895889416667,82.6740445533333,99.6199824175,116.56814585,92.2093571091667,92.8476058966667,97.0843245733333,112.151191458333,91.8400749091667,96.0067419616667,114.588775734167,125.134380833333,128.547622025,151.765046116667,168.8233893,199.538643258333,149.852054458333,153.14790065,150.630132075,126.8796898]
driver_4 = [47346552670,49879770605,54650942605,54561215775,53432327058,54561791263,66442751527,78398442241,84329031218,84990165431,83407993005,78237979893,80708070966,81469794408,89278920340,92080928754,94715251080,104665000000,113382000000,126880000000,143688000000,176559000000,221674000000,223427000000,245149000000,272163000000,295546000000,304087000000,309661000000,321867000000,325129000000,299373000000,325034000000,316719000000,308084000000,295853000000,287961000000,293168000000,290996000000,298095000000,320086000000,331806000000,378463000000,440532000000,492999000000,533203000000,558335000000,589586000000,656756000000,705917000000,738005000000,752288000000,725205000000,679229000000,647789000000,633830000000,639856000000,646753000000,682491000000,734344000000,778397000000,806230000000,860692000000,916015000000]

forecast_6 = [3.51291169482192,3.20890025710232,2.86599844722779,3.05521397522535,3.0047058659705,3.04626390935626,3.00074592204549,3.20671868923621,3.27066328123962,3.17929565616567,3.09233665266154,3.06789214960608,2.97600553384586,2.98603750499126,2.88603132092713,2.68408649820258,3.93050223689287,8.95139207099094,9.98242319376683,11.25884737202,10.9286955590019,10.5659683551149,10.9151140198873,27.4273244864498]
forecast_7 = [10868,23355,52299,66683,98758,73494,114217,177772,148482,98587,155791,248353,349408,263583,244940]
forecast_8 = [0.0099,0.0105,0.0101,0.0101,0.0094,0.0088,0.0089,0.009,0.0086,0.008,0.0075,0.0076,0.0072,0.0068,0.0066,0.0063,0.0057,0.0058,0.006,0.0057,0.0052,0.0051,0.005,0.0049,0.005,0.0049,0.005,0.0047,0.005,0.0047,0.0048,0.0049,0.005,0.0052,0.005]
forecast_9 = [67.254725,73.878425,74.7745,79.575475,78.1143583333333,85.6417166666667,85.2814083333333,101.570483333333,109.744966666667,105.116058333333,113.108241666667,109.815708333333,111.840475,108.186133333333,104.27615,101.302833333333,97.7371833333333,100,103.655366666667,114.111841666667,107.695375,109.2814,107.2944,114.468775,119.731758333333,123.15306]
forecast_10 = [1,0,2,0,2,1,3,0,5,5,17,12,7]
forecast_11 = [18,25,27,26,34,30,30,33,35,33,37,39,46,59,57,62,64,65,72,72,70,67,61,78,84,76,76,90,98,106,90,97,115,107,103,93,102,113,114,124,136,136,161,167,158,143,149,152,155,149,149,148,135,155,141,146,159,147,157,173,181,217,239,247,276,257,261,242,252,271,305,400,500,615]
forecast_12 = [19757942,19723432,27085492,35866555,34716440,35734708,40994266,34991305,45265764,100466649,59214870,78729317,121660680,121787727,156976479,285468751,302580697,373498028,7691381458,7434807636,5787143669,8143589198,6162390735,4855360301,5892200065,6905678657,7893542777,8010325428,8737484417,10680838866,12300126183,15481098251,19112247949,14622504641,10431867501,14540801057]
forecast_13 = [0.7928232905,0.7326657129,0.7924929178,0.7372286079,0.6843039773,0.7856637794,0.7551956562,0.8048245614,0.7620008649,0.731579734,0.7883567862,0.7802431025]
forecast_14 = [25,39,50,41,30,34,23,37,26,19,12,20,13,19,14,17,8,27,20,14,13,17,17,38,20,5,9,8,6,10]
forecast_15 = [18,50,61,69,76,83,88,92,95,98,103]
forecast_16 = [36,37,39,41,43.5,46,48,50,52,47,46,45,46,35,38,33,34,23,25,21,18,32,35]
forecast_17 = [29.47,30.11,33.25,29.1,25.9,26.98,23.99,22.74,19.72,21.62,25.06,24.82,24.38,23.61]
forecast_18 = [12,12,10,10,10,10,9,9,9]
forecast_19 = [0.3497,0.2538,0.3119,0.1984,0.3493,0.3626]
forecast_20 = [43.33,44.56,50.76,55.66,55.84,61.06,56.31,53.96,51.36,43.26,47.58,51.21,52.09,51.94,49.8]


#config
series_info = {"driver_1":{"start":1986,"end":2025},"driver_2":{"start":2004,"end":2025},"driver_3":{"start":1993,"end":2024},"driver_4":{"start":1960,"end":2023},"forecast_6":{"start":2002,"end":2025},"forecast_7":{"start":2010,"end":2024},"forecast_8":{"start":1990,"end":2024},"forecast_9":{"start":2000,"end":2025},"forecast_10":{"start":2012,"end":2024},"forecast_11":{"start":1950,"end":2023},"forecast_12":{"start":1989,"end":2024},"forecast_13":{"start":2013,"end":2024},"forecast_14":{"start":1995,"end":2024},"forecast_15":{"start":2015,"end":2025},"forecast_16":{"start":2000,"end":2022},"forecast_17":{"start":2012,"end":2025},"forecast_18":{"start":2016,"end":2024},"forecast_19":{"start":2018,"end":2023},"forecast_20":{"start":2009,"end":2023}}
DRIVER_MAP = {
 'forecast_6': ['driver_1_L1','driver_2_L2','driver_3_L1','driver_4_L0'],
 'forecast_7': ['driver_1_L1','driver_2_L0','driver_3_L2','driver_4_L0'],
 'forecast_8': ['driver_1_L2','driver_3_L2','driver_4_L0'],
 'forecast_9': ['driver_4_L0'],
 'forecast_10': ['driver_1_L2','driver_2_L0','driver_3_L2','driver_4_L2'],
 'forecast_11': ['driver_1_L2','driver_3_L2','driver_4_L0'],
 'forecast_12': ['driver_1_L0','driver_2_L2','driver_4_L2'],
 'forecast_13': ['driver_3_L2','driver_4_L0'],
 'forecast_14': ['driver_1_L0','driver_2_L2','driver_3_L2'],
 'forecast_15':['driver_1_L0','driver_2_L2','driver_3_L2','driver_4_L2'],
 'forecast_16':['driver_1_L0','driver_4_L1'],
 'forecast_17':['driver_1_L0','driver_2_L0','driver_3_L1','driver_4_L0'],
 'forecast_18':['driver_1_L2','driver_2_L2','driver_3_L2','driver_4_L0'],
 'forecast_19': None,
 'forecast_20':['driver_2_L0','driver_3_L2','driver_4_L2']
}

THRESHOLD = {'forecast_6': 25, 'forecast_7': 349408, 'forecast_8': 0.005, 'forecast_9': 124.336, 'forecast_10': 9,'forecast_11': 1000, 'forecast_12': 14837703880, 'forecast_13': 0.76, 'forecast_14': 21, 'forecast_15': 105, 'forecast_16': 50, 'forecast_17': 23.12, 'forecast_18': 9, 'forecast_19': 0.3043, 'forecast_20': 50}
EVENT_WINDOW = {'forecast_6': '2026-2030', 'forecast_7': '2026', 'forecast_8':'2030', 'forecast_9': '2028-2032', 'forecast_10': '2026-2030','forecast_11': '2027','forecast_12': '2026-2030','forecast_13': '2033', 'forecast_14':'2026-2030', 'forecast_15':'2031', 'forecast_16':'2032', 'forecast_17':'2026-2030', 'forecast_18':'2026-2030', 'forecast_19':'2026-2030', 'forecast_20':'2030'}

#helper functions

def make_series(name: str) -> pd.Series:
    vals = globals()[name]
    start = series_info[name]["start"]
    index = pd.RangeIndex(start, start + len(vals), name="year")
    return pd.Series(vals, index=index, name=name)

def fit_ets(y: pd.Series):
    return ExponentialSmoothing(
        y, trend="add", damped_trend=True, seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

def standardize(df: pd.DataFrame):
    mu = df.mean()
    sd = df.std(ddof=1)
    # Add a small epsilon to standard deviation to avoid division by zero
    return (df - mu) / (sd + 1e-9), mu, sd

def get_driver_z_series(drv: str, lag: int, idx: pd.RangeIndex, targets: dict):
    z = pd.Series(0.0, index=idx)
    target_z = targets.get(drv, 0.0)
    ramp_years = range(2026, 2036)
    span = len(ramp_years)
    
    ramp = pd.Series(np.linspace(0, target_z, span), index=ramp_years)
    z.update(ramp)
    z = z.ffill().fillna(0.0)
    
    if lag > 0:
        z = z.shift(lag)
        
    return z.fillna(0.0)

#main

def run_pipeline(key: str, targets: dict[str, float], *, seed: int = 42):
    """
    Runs the full driver-adjusted forecast pipeline.
    This version uses window-consistent standardization and a fixed driver start date.
    """
    np.random.seed(seed)

    DRIVER_COMMON_START = 2004
    all_series = {name: make_series(name) for name in series_info}
    if DRIVER_COMMON_START is not None:
        for name, s in all_series.items():
            if name.startswith("driver_"):
                all_series[name] = s.loc[s.index >= DRIVER_COMMON_START]
    
    # 1. Generate baseline ETS forecast
    y_hist = all_series[key]
    ets_fit = fit_ets(y_hist)
    fut_idx = pd.RangeIndex(y_hist.index[-1] + 1, 2036, name="year")
    y_base = pd.concat([y_hist, ets_fit.forecast(len(fut_idx)).set_axis(fut_idx)])

    # 2. Check for drivers and prepare for simulation
    x_terms = DRIVER_MAP.get(key)
    N_MC = 5000
    hist_len, tot_len = len(y_hist), len(y_base)
    
    # Prepare ETS error band
    mse = np.mean(ets_fit.resid**2)
    steps = np.arange(1, len(fut_idx) + 1)
    ets_err_band = pd.concat([pd.Series(0.0, y_hist.index), pd.Series(1.96 * np.sqrt(mse*steps), fut_idx)])

    # 3. Handle driver-adjusted vs. ETS-only cases
    if x_terms:
        # --- Driver-Adjusted Path ---
        # a. Build regression dataframe with correct lags
        series_list = [y_hist]
        for term in x_terms:
            drv, lag_str = term.rsplit('_L', 1)
            # Use the (potentially truncated) driver series
            s_shifted = all_series[drv].shift(int(lag_str))
            s_shifted.name = term
            series_list.append(s_shifted)
        
        df_reg = pd.concat(series_list, axis=1, join='inner').dropna()

        # b. Standardize on the regression window & fit OLS
        z_df, mu, sd = standardize(df_reg)
        mu_y, sd_y = mu[key], sd[key]
        ols_fit = sm.OLS(z_df[key], z_df[x_terms]).fit()

        # c. Calculate future driver adjustments
        total_adj = pd.Series(0.0, index=y_base.index)
        effects_df = pd.DataFrame(index=y_base.index)
        for term in x_terms:
            drv, lag_str = term.rsplit('_L', 1)
            lag = int(lag_str)
            
            z_ramp = get_driver_z_series(drv, lag, y_base.index, targets)
            adj_for_term = ols_fit.params[term] * z_ramp
            total_adj += adj_for_term

            effects_df[f"{term}_z"] = z_ramp
            effects_df[f"{term}_effect"] = adj_for_term * sd_y

        # d. Run Monte Carlo simulation with two noise sources
        base_z = (y_base - mu_y) / (sd_y + 1e-9)
        adj_z = base_z + total_adj
        
        band_z = (ets_err_band / 1.96) / (sd_y + 1e-9)
        band_z.loc[y_hist.index] = 0
        
        noise_ets = np.random.normal(0, band_z.values, size=(N_MC, tot_len))
        resid_smp = np.random.choice(ols_fit.resid, size=(N_MC, tot_len - hist_len), replace=True)
        noise_res = np.hstack([np.zeros((N_MC, hist_len)), resid_smp])
        
        sims_z = adj_z.values + noise_ets + noise_res
        sims = sims_z * sd_y + mu_y
    
    else:
        #ETS-Only Path (for forecasts with no drivers)
        effects_df = pd.DataFrame(index=y_base.index) # Empty effects
        ets_std_err = ets_err_band / 1.96
        ets_std_err.loc[y_hist.index] = 0
        noise = np.random.normal(0, ets_std_err.values, size=(N_MC, tot_len))
        sims = y_base.values + noise

    # 4. Calculate final probability
    thr, win = THRESHOLD[key], EVENT_WINDOW[key]
    if '-' in win:
        y0, y1 = map(int, win.split("-"))
        mask = (y_base.index >= y0) & (y_base.index <= y1)
        metric = sims[:, mask].mean(axis=1)
    else:
        metric = sims[:, y_base.index.get_loc(int(win))]
    
    probability = float((metric > thr).mean())

    # 5. Prepare outputs for the app
    q = lambda p: np.percentile(sims, p, axis=0)
    fan_df = pd.DataFrame({"p05": q(5), "p25": q(25), "p50": q(50), "p75": q(75), "p95": q(95)}, index=y_base.index)

    effects_df["baseline_level"] = y_base
    effects_df["total_driver_effect"] = effects_df.filter(like="_effect").sum(axis=1)
    effects_df["adjusted_level"] = effects_df["baseline_level"] + effects_df["total_driver_effect"]

    return probability, sims, effects_df, fan_df