# ============================================================
# HAWKES + INTRA-NUAGES + MODÈLE HYBRIDE (XGBOOST)
# VERSION COMPLÈTE, STRUCTURÉE ET SAUVEGARDÉE
# ============================================================

import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------
DATA_PATH = "data_train_databattle2026/segment_alerts_all_airports_train.csv"
OUTPUT_DIR = "hawkes_hybrid_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paramètres Hawkes / simulation
GAMMA = 0.03                 # extinction globale
N_SIM = 60                   # mettre 30 pour debug rapide, 60/100 pour final
MAX_TIME_MIN = 180           # horizon max pour la simulation
INTENSITY_THRESHOLD = 5e-5   # seuil d'extinction
TRAIN_QUANTILE = 0.80        # split temporel par date de début d'alerte

# ------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ------------------------------------------------------------
print("=" * 60)
print("1. CHARGEMENT DES DONNÉES")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"], low_memory=False)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

# Normalisation des booléens
for col in ["icloud", "is_last_lightning_cloud_ground"]:
    if col in df.columns:
        df[col] = df[col].map({
            "VRAI": True, "FAUX": False,
            "True": True, "False": False,
            True: True, False: False,
            1: True, 0: False
        })

# Exclusion optionnelle de Pise 2016
mask_pise_2016 = (df["airport"] == "Pise") & (df["date"].dt.year == 2016)
df = df[~mask_pise_2016].copy()

# On garde tous les événements appartenant à une alerte
# (si des intra-nuages sont présents avec airport_alert_id, ils sont conservés)
df = df[
    (df["airport_alert_id"].notna()) &
    (df["date"].notna())
].copy()

df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)

print(f"Dataset size: {df.shape}")
print(f"Nombre d'alertes: {df.groupby(['airport', 'airport_alert_id']).ngroups}")

# ------------------------------------------------------------
# 2. CONSTRUCTION DE tau RÉEL ET DES FEATURES DE BASE
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. CONSTRUCTION DES SÉQUENCES TEMPORELLES")
print("=" * 60)

# Temps du dernier événement de chaque alerte
last_time = df.groupby(["airport", "airport_alert_id"])["date"].transform("max")
first_time = df.groupby(["airport", "airport_alert_id"])["date"].transform("min")

# τ réel = temps restant jusqu'au dernier événement de l'alerte
df["tau_true_min"] = (last_time - df["date"]).dt.total_seconds() / 60.0

# temps relatif dans l'alerte
df["t_min"] = (df["date"] - first_time).dt.total_seconds() / 60.0

# index dans l'alerte
df["event_idx"] = df.groupby(["airport", "airport_alert_id"]).cumcount()
df["n_events"] = df.groupby(["airport", "airport_alert_id"])["event_idx"].transform("max") + 1
df["progress_ratio"] = np.where(df["n_events"] > 1, df["event_idx"] / (df["n_events"] - 1), 1.0)

# inter-temps
df["dt_prev"] = df.groupby(["airport", "airport_alert_id"])["t_min"].diff().fillna(0)

# rolling mean / std des inter-temps
df["rolling_mean_dt"] = (
    df.groupby(["airport", "airport_alert_id"])["dt_prev"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

df["rolling_std_dt"] = (
    df.groupby(["airport", "airport_alert_id"])["dt_prev"]
    .transform(lambda x: x.rolling(5, min_periods=1).std())
).fillna(0)

# intra-nuages
df["icloud_int"] = df["icloud"].fillna(False).astype(int)

df["icloud_ratio"] = (
    df.groupby(["airport", "airport_alert_id"])["icloud_int"]
    .transform(lambda x: x.expanding().mean())
)

df["icloud_recent"] = (
    df.groupby(["airport", "airport_alert_id"])["icloud_int"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

# nombre cumulé d'événements
df["cum_strikes"] = df["event_idx"] + 1

# amplitude / polarité
df["abs_amplitude"] = df["amplitude"].abs()
df["is_positive"] = (df["amplitude"] > 0).astype(int)

# géométrie
az = np.radians(df["azimuth"].fillna(0))
df["x_pos"] = df["dist"].fillna(0) * np.sin(az)
df["y_pos"] = df["dist"].fillna(0) * np.cos(az)
df["dist_amp_interaction"] = df["dist"].fillna(0) * df["abs_amplitude"].fillna(0)

# temps cyclique
df["hour"] = df["date"].dt.hour
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# encodage aéroport
df["airport_code"] = df["airport"].astype("category").cat.codes

# utile pour le modèle log
df["log_tau_true"] = np.log1p(df["tau_true_min"])

# ------------------------------------------------------------
# 3. SPLIT TEMPOREL PAR ALERTE
# ------------------------------------------------------------
alerts_df = (
    df.groupby(["airport", "airport_alert_id"], as_index=False)
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_events=("event_idx", "count"),
        duration_min=("t_min", "max")
    )
    .rename(columns={"airport_alert_id": "alert_id"})
)

split_date = alerts_df["start_date"].quantile(TRAIN_QUANTILE)
alerts_df["set"] = np.where(alerts_df["start_date"] <= split_date, "train", "test")

df = df.rename(columns={"airport_alert_id": "alert_id"})
df = df.merge(alerts_df[["airport", "alert_id", "set"]], on=["airport", "alert_id"], how="left")

print(f"Date de split: {split_date}")
print(f"Alertes train: {(alerts_df['set'] == 'train').sum()}")
print(f"Alertes test : {(alerts_df['set'] == 'test').sum()}")
print(f"Événements train: {(df['set'] == 'train').sum()}")
print(f"Événements test : {(df['set'] == 'test').sum()}")

# ------------------------------------------------------------
# 4. DÉFINITION DES FONCTIONS HAWKES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DÉFINITION DES FONCTIONS HAWKES")
print("=" * 60)

def hawkes_loglik_sequence(times, ic, mu, alpha, beta, delta):
    """
    Log-vraisemblance négative pour une séquence Hawkes exponentielle.
    - g_all[i] = somme_{j<i} exp(-beta*(t_i-t_j))
    - g_ic[i]  = somme_{j<i, ic_j=1} exp(-beta*(t_i-t_j))
    λ_i = mu + alpha*g_all[i] + delta*g_ic[i]
    """
    n = len(times)
    if n < 2:
        return 0.0

    g_all = np.zeros(n)
    g_ic = np.zeros(n)

    for i in range(1, n):
        dt = times[i] - times[i - 1]
        decay = np.exp(-beta * dt)
        g_all[i] = decay * (g_all[i - 1] + 1.0)
        g_ic[i] = decay * (g_ic[i - 1] + float(ic[i - 1]))

    lambdas = mu + alpha * g_all + delta * g_ic
    if np.any(lambdas <= 1e-12):
        return 1e12

    # loglik sur les événements observés
    ll = np.sum(np.log(lambdas))

    # compensateur
    T = times[-1]
    comp_all = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    comp_ic = (delta / beta) * np.sum(ic * (1.0 - np.exp(-beta * (T - times))))
    compensator = mu * T + comp_all + comp_ic

    # pénalité douce stationnarité
    penalty = 0.0
    if alpha >= beta:
        penalty += 1e6 * (alpha - beta + 1e-6) ** 2

    return -(ll - compensator) + penalty


def hawkes_total_negloglik(params, sequences):
    mu, alpha, beta, delta = params

    if mu <= 1e-8 or alpha < 0 or beta <= 0 or delta < 0:
        return 1e18

    total = 0.0
    for times, ic in sequences:
        val = hawkes_loglik_sequence(times, ic, mu, alpha, beta, delta)
        if not np.isfinite(val):
            return 1e18
        total += val
    return total


def optimize_hawkes_params_robust(sequences):
    print("Optimisation des paramètres Hawkes...")

    initial_guesses = [
        (0.005, 0.05, 0.10, 0.01),
        (0.010, 0.08, 0.15, 0.02),
        (0.010, 0.10, 0.20, 0.03),
        (0.008, 0.12, 0.25, 0.05),
        (0.003, 0.06, 0.12, 0.01),
        (0.020, 0.15, 0.30, 0.05),
        (0.001, 0.03, 0.08, 0.005),
        (0.015, 0.09, 0.18, 0.02),
    ]

    best_ll = np.inf
    best_params = None

    for i, init in enumerate(initial_guesses, start=1):
        try:
            res = minimize(
                hawkes_total_negloglik,
                x0=init,
                args=(sequences,),
                method="L-BFGS-B",
                bounds=[(1e-6, 0.10), (1e-6, 1.0), (1e-6, 1.5), (0.0, 1.0)],
                options={"maxiter": 400, "ftol": 1e-8}
            )

            if res.success and res.fun < best_ll:
                best_ll = res.fun
                best_params = res.x
                print(f"  Départ {i}: mu={res.x[0]:.5f}, alpha={res.x[1]:.5f}, beta={res.x[2]:.5f}, delta={res.x[3]:.5f} -> LL={res.fun:.2f} ✓")
            elif res.success:
                print(f"  Départ {i}: mu={res.x[0]:.5f}, alpha={res.x[1]:.5f}, beta={res.x[2]:.5f}, delta={res.x[3]:.5f} -> LL={res.fun:.2f}")
            else:
                print(f"  Départ {i}: Échec de convergence")
        except Exception as e:
            print(f"  Départ {i}: Erreur - {str(e)[:60]}")

    if best_params is None:
        print("\n⚠️ Aucune optimisation n'a convergé, valeurs par défaut utilisées.")
        best_params = np.array([0.01, 0.15, 0.30, 0.02])

    mu_opt, alpha_opt, beta_opt, delta_opt = best_params
    if alpha_opt >= beta_opt:
        alpha_opt = 0.95 * beta_opt

    return float(mu_opt), float(alpha_opt), float(beta_opt), float(delta_opt)


def compute_post_event_state(times, ic, beta):
    """
    État juste après chaque événement :
    z_all_post[i] = somme_{j<=i} exp(-beta*(t_i-t_j))
    z_ic_post[i]  = somme_{j<=i, ic_j=1} exp(-beta*(t_i-t_j))
    """
    n = len(times)
    z_all = np.zeros(n)
    z_ic = np.zeros(n)

    if n == 0:
        return z_all, z_ic

    z_all[0] = 1.0
    z_ic[0] = float(ic[0])

    for i in range(1, n):
        dt = times[i] - times[i - 1]
        decay = np.exp(-beta * dt)
        z_all[i] = decay * z_all[i - 1] + 1.0
        z_ic[i] = decay * z_ic[i - 1] + float(ic[i])

    return z_all, z_ic


def simulate_hawkes_with_extinction(mu, alpha, beta, delta, gamma, z_all0, z_ic0,
                                    p_ic_next,
                                    max_time_min=180,
                                    intensity_threshold=5e-5,
                                    n_sim=60,
                                    seed=None):
    """
    Simule le futur depuis l'état courant.
    - z_all0 : excitation totale courante
    - z_ic0  : excitation intra-nuage courante
    - p_ic_next : proba qu'un futur événement soit IC (approchée par icloud_recent)
    """
    rng = np.random.default_rng(seed)
    tau_samples = np.zeros(n_sim)

    p_ic_next = float(np.clip(p_ic_next, 0.0, 1.0))

    for s in range(n_sim):
        t = 0.0
        z_all = float(z_all0)
        z_ic = float(z_ic0)

        last_event_time = 0.0
        has_event = False

        while True:
            lam = (mu + alpha * z_all + delta * z_ic) * np.exp(-gamma * t)

            if lam < intensity_threshold:
                break

            w = rng.exponential(1.0 / max(lam, 1e-12))
            t_candidate = t + w

            if t_candidate > max_time_min:
                break

            decay = np.exp(-beta * w)
            z_all_decay = z_all * decay
            z_ic_decay = z_ic * decay

            lam_candidate = (mu + alpha * z_all_decay + delta * z_ic_decay) * np.exp(-gamma * t_candidate)
            accept_prob = min(1.0, lam_candidate / max(lam, 1e-12))

            if rng.uniform() <= accept_prob:
                t = t_candidate

                # type du futur événement
                is_ic_future = rng.uniform() < p_ic_next

                z_all = z_all_decay + 1.0
                z_ic = z_ic_decay + float(is_ic_future)

                last_event_time = t
                has_event = True
            else:
                t = t_candidate
                z_all = z_all_decay
                z_ic = z_ic_decay

        tau_samples[s] = last_event_time if has_event else 0.0

    return {
        "tau_mean": float(np.mean(tau_samples)),
        "tau_median": float(np.median(tau_samples)),
        "tau_q05": float(np.percentile(tau_samples, 5)),
        "tau_q95": float(np.percentile(tau_samples, 95)),
        "tau_std": float(np.std(tau_samples)),
        "p_finished_30min": float(np.mean(tau_samples < 30.0)),
        "p_finished_60min": float(np.mean(tau_samples < 60.0)),
    }

# ------------------------------------------------------------
# 5. AJUSTEMENT DES PARAMÈTRES HAWKES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. AJUSTEMENT DES PARAMÈTRES HAWKES")
print("=" * 60)

train_sequences = []
for (airport, alert_id), g in df[df["set"] == "train"].groupby(["airport", "alert_id"], sort=False):
    g = g.sort_values("t_min")
    times = g["t_min"].values
    ic = g["icloud_int"].values
    if len(times) >= 3:
        train_sequences.append((times, ic))

print(f"Nombre de séquences d'entraînement (>=3 événements): {len(train_sequences)}")

t0_opt = time.time()
mu_opt, alpha_opt, beta_opt, delta_opt = optimize_hawkes_params_robust(train_sequences)
t1_opt = time.time()

print(f"\n✅ Paramètres optimaux trouvés en {t1_opt - t0_opt:.2f} secondes:")
print(f"   μ (taux de base)      = {mu_opt:.6f} /min")
print(f"   α (auto-excitation)  = {alpha_opt:.6f} /min")
print(f"   β (décroissance)     = {beta_opt:.6f} /min")
print(f"   δ (effet intra-nuage)= {delta_opt:.6f} /min")
print(f"   γ (extinction)       = {GAMMA:.6f} /min")
print(f"   α/β = {alpha_opt / beta_opt:.4f}")

# ------------------------------------------------------------
# 6. CALCUL DE L'ÉTAT D'EXCITATION POST-ÉVÉNEMENT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. CALCUL DE L'ÉTAT D'EXCITATION POST-ÉVÉNEMENT")
print("=" * 60)

df["z_all_post"] = 0.0
df["z_ic_post"] = 0.0
df["lambda_now"] = 0.0

for (airport, alert_id), g in df.groupby(["airport", "alert_id"], sort=False):
    g = g.sort_values("t_min")
    times = g["t_min"].values
    ic = g["icloud_int"].values
    z_all, z_ic = compute_post_event_state(times, ic, beta_opt)

    idx = g.index
    df.loc[idx, "z_all_post"] = z_all
    df.loc[idx, "z_ic_post"] = z_ic
    df.loc[idx, "lambda_now"] = mu_opt + alpha_opt * z_all + delta_opt * z_ic

print(f"État d'excitation calculé pour {len(df)} événements")
print(f"  z_all moyen: {df['z_all_post'].mean():.2f}")
print(f"  z_ic moyen : {df['z_ic_post'].mean():.2f}")
print(f"  λ moyen    : {df['lambda_now'].mean():.4f} /min")

# ------------------------------------------------------------
# 7. PRÉDICTION HAWKES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. PRÉDICTION AVEC SIMULATIONS HAWKES")
print("=" * 60)

print("Paramètres de simulation:")
print(f"  - Nombre de simulations: {N_SIM}")
print(f"  - Temps max: {MAX_TIME_MIN} min")
print(f"  - Seuil d'extinction: {INTENSITY_THRESHOLD}")

predictions = []
n_total = len(df)

print("\nCalcul des prédictions...")
for i, row in enumerate(df.itertuples(index=False), start=1):
    pred = simulate_hawkes_with_extinction(
        mu=mu_opt,
        alpha=alpha_opt,
        beta=beta_opt,
        delta=delta_opt,
        gamma=GAMMA,
        z_all0=row.z_all_post,
        z_ic0=row.z_ic_post,
        p_ic_next=row.icloud_recent,
        max_time_min=MAX_TIME_MIN,
        intensity_threshold=INTENSITY_THRESHOLD,
        n_sim=N_SIM,
        seed=RANDOM_STATE + i
    )
    predictions.append(pred)

    if i % 5000 == 0:
        print(f"  Progression: {i}/{n_total} événements traités")

pred_df = pd.DataFrame(predictions).add_prefix("tau_pred_")
df_events = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

print(f"\n✅ Prédictions Hawkes calculées pour {len(df_events)} événements")

# ------------------------------------------------------------
# 8. MODÈLE HYBRIDE XGBOOST
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("7. MODÈLE HYBRIDE XGBOOST")
print("=" * 60)

hybrid_features = [
    # sorties Hawkes
    "tau_pred_tau_mean",
    "tau_pred_tau_median",
    "tau_pred_tau_q05",
    "tau_pred_tau_q95",
    "tau_pred_tau_std",
    "tau_pred_p_finished_30min",
    "tau_pred_p_finished_60min",
    # états Hawkes
    "z_all_post",
    "z_ic_post",
    "lambda_now",
    # dynamique récente
    "event_idx",
    "n_events",
    "progress_ratio",
    "dt_prev",
    "rolling_mean_dt",
    "rolling_std_dt",
    "cum_strikes",
    # intra-nuages
    "icloud_int",
    "icloud_ratio",
    "icloud_recent",
    # physique
    "dist",
    "azimuth",
    "maxis",
    "abs_amplitude",
    "is_positive",
    "dist_amp_interaction",
    "x_pos",
    "y_pos",
    # temporel
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dayofyear",
    # contexte
    "airport_code",
]

train_mask = df_events["set"] == "train"
test_mask = df_events["set"] == "test"

X_train = df_events.loc[train_mask, hybrid_features].fillna(0)
X_test = df_events.loc[test_mask, hybrid_features].fillna(0)

# Régression sur log(1+tau) pour mieux modéliser l'asymétrie
y_train_log = df_events.loc[train_mask, "log_tau_true"].values
y_test = df_events.loc[test_mask, "tau_true_min"].values

if HAS_XGB:
    model_xgb = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    model_xgb = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

model_xgb.fit(X_train, y_train_log)

# prédictions XGB sur toute la base
X_all = df_events[hybrid_features].fillna(0)
df_events["tau_pred_xgb"] = np.expm1(model_xgb.predict(X_all))
df_events["tau_pred_xgb"] = np.clip(df_events["tau_pred_xgb"], 0, 300)

# Blend Hawkes + XGB
# Poids choisi sur train uniquement
best_w = None
best_mae_blend = np.inf

train_true = df_events.loc[train_mask, "tau_true_min"].values
train_hawkes = df_events.loc[train_mask, "tau_pred_tau_mean"].values
train_xgb = df_events.loc[train_mask, "tau_pred_xgb"].values

for w in np.linspace(0, 1, 21):
    pred_blend = w * train_hawkes + (1 - w) * train_xgb
    mae_blend = mean_absolute_error(train_true, pred_blend)
    if mae_blend < best_mae_blend:
        best_mae_blend = mae_blend
        best_w = w

df_events["tau_pred_final"] = best_w * df_events["tau_pred_tau_mean"] + (1 - best_w) * df_events["tau_pred_xgb"]
df_events["tau_pred_final"] = np.clip(df_events["tau_pred_final"], 0, 300)

print(f"✅ Modèle hybride entraîné")
print(f"   Poids optimal blend (train) : Hawkes={best_w:.2f}, XGB={1-best_w:.2f}")

# ------------------------------------------------------------
# 9. MÉTRIQUES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("8. ÉVALUATION DES PERFORMANCES")
print("=" * 60)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    rel_err = np.median(np.abs(y_true - y_pred) / (y_true + 1.0))
    corr, pval = pearsonr(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "MedianRelativeError": rel_err,
        "Corr": corr,
        "Corr_pvalue": pval
    }

metrics_hawkes = compute_metrics(
    df_events.loc[test_mask, "tau_true_min"].values,
    df_events.loc[test_mask, "tau_pred_tau_mean"].values
)

metrics_xgb = compute_metrics(
    df_events.loc[test_mask, "tau_true_min"].values,
    df_events.loc[test_mask, "tau_pred_xgb"].values
)

metrics_final = compute_metrics(
    df_events.loc[test_mask, "tau_true_min"].values,
    df_events.loc[test_mask, "tau_pred_final"].values
)

# couverture Hawkes
coverage_90_hawkes = np.mean(
    (df_events.loc[test_mask, "tau_true_min"] >= df_events.loc[test_mask, "tau_pred_tau_q05"]) &
    (df_events.loc[test_mask, "tau_true_min"] <= df_events.loc[test_mask, "tau_pred_tau_q95"])
)

# classification fin<30
df_events["true_finished_30"] = df_events["tau_true_min"] < 30
df_events["pred_hawkes_finished_30"] = df_events["tau_pred_p_finished_30min"] > 0.5
df_events["pred_xgb_finished_30"] = df_events["tau_pred_xgb"] < 30
df_events["pred_final_finished_30"] = df_events["tau_pred_final"] < 30

acc30_hawkes = np.mean(df_events.loc[test_mask, "pred_hawkes_finished_30"] == df_events.loc[test_mask, "true_finished_30"])
acc30_xgb = np.mean(df_events.loc[test_mask, "pred_xgb_finished_30"] == df_events.loc[test_mask, "true_finished_30"])
acc30_final = np.mean(df_events.loc[test_mask, "pred_final_finished_30"] == df_events.loc[test_mask, "true_finished_30"])

print("\n📊 HAWKES (TEST)")
print(f"  MAE                  : {metrics_hawkes['MAE']:.2f} min")
print(f"  RMSE                 : {metrics_hawkes['RMSE']:.2f} min")
print(f"  Bias                 : {metrics_hawkes['Bias']:.2f} min")
print(f"  Erreur relative méd. : {metrics_hawkes['MedianRelativeError']:.2%}")
print(f"  Corrélation          : {metrics_hawkes['Corr']:.3f} (p={metrics_hawkes['Corr_pvalue']:.2e})")
print(f"  Coverage 90%         : {coverage_90_hawkes:.2%}")
print(f"  Accuracy fin<30min   : {acc30_hawkes:.2%}")

print("\n📊 XGBOOST (TEST)")
print(f"  MAE                  : {metrics_xgb['MAE']:.2f} min")
print(f"  RMSE                 : {metrics_xgb['RMSE']:.2f} min")
print(f"  Bias                 : {metrics_xgb['Bias']:.2f} min")
print(f"  Erreur relative méd. : {metrics_xgb['MedianRelativeError']:.2%}")
print(f"  Corrélation          : {metrics_xgb['Corr']:.3f} (p={metrics_xgb['Corr_pvalue']:.2e})")
print(f"  Accuracy fin<30min   : {acc30_xgb:.2%}")

print("\n📊 MODÈLE FINAL (TEST)")
print(f"  MAE                  : {metrics_final['MAE']:.2f} min")
print(f"  RMSE                 : {metrics_final['RMSE']:.2f} min")
print(f"  Bias                 : {metrics_final['Bias']:.2f} min")
print(f"  Erreur relative méd. : {metrics_final['MedianRelativeError']:.2%}")
print(f"  Corrélation          : {metrics_final['Corr']:.3f} (p={metrics_final['Corr_pvalue']:.2e})")
print(f"  Accuracy fin<30min   : {acc30_final:.2%}")

# ------------------------------------------------------------
# 10. STATISTIQUES PAR TYPE D'ÉVÉNEMENT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("9. STATISTIQUES PAR TYPE D'ÉVÉNEMENT")
print("=" * 60)

test_events = df_events.loc[test_mask].copy()
first_events = test_events[test_events["event_idx"] == 0].copy()
last_events = test_events[test_events["event_idx"] == (test_events["n_events"] - 1)].copy()

print("\nPremier événement de l'alerte:")
print(f"  Tau réel moyen     : {first_events['tau_true_min'].mean():.2f} min")
print(f"  Hawkes moyen       : {first_events['tau_pred_tau_mean'].mean():.2f} min")
print(f"  XGB moyen          : {first_events['tau_pred_xgb'].mean():.2f} min")
print(f"  Final moyen        : {first_events['tau_pred_final'].mean():.2f} min")

print("\nDernier événement de l'alerte:")
print(f"  Tau réel moyen     : {last_events['tau_true_min'].mean():.2f} min")
print(f"  Hawkes moyen       : {last_events['tau_pred_tau_mean'].mean():.2f} min")
print(f"  XGB moyen          : {last_events['tau_pred_xgb'].mean():.2f} min")
print(f"  Final moyen        : {last_events['tau_pred_final'].mean():.2f} min")

# ------------------------------------------------------------
# 11. STATISTIQUES PAR AÉROPORT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("10. STATISTIQUES PAR AÉROPORT")
print("=" * 60)

airport_stats = []
for airport, g in test_events.groupby("airport"):
    airport_stats.append({
        "airport": airport,
        "n_events": len(g),
        "n_alerts": g["alert_id"].nunique(),
        "tau_true_mean": g["tau_true_min"].mean(),
        "tau_hawkes_mean": g["tau_pred_tau_mean"].mean(),
        "tau_xgb_mean": g["tau_pred_xgb"].mean(),
        "tau_final_mean": g["tau_pred_final"].mean(),
        "mae_hawkes": mean_absolute_error(g["tau_true_min"], g["tau_pred_tau_mean"]),
        "mae_xgb": mean_absolute_error(g["tau_true_min"], g["tau_pred_xgb"]),
        "mae_final": mean_absolute_error(g["tau_true_min"], g["tau_pred_final"]),
        "rmse_final": np.sqrt(mean_squared_error(g["tau_true_min"], g["tau_pred_final"])),
    })

airport_df = pd.DataFrame(airport_stats)
print(airport_df.round(3).to_string(index=False))

# ------------------------------------------------------------
# 12. VISUALISATIONS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("11. GÉNÉRATION DES VISUALISATIONS")
print("=" * 60)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. distributions
ax = axes[0, 0]
sns.histplot(test_events["tau_true_min"], bins=60, kde=True, ax=ax, label="true", alpha=0.35, color="steelblue")
sns.histplot(test_events["tau_pred_tau_mean"], bins=60, kde=True, ax=ax, label="hawkes", alpha=0.35, color="orange")
sns.histplot(test_events["tau_pred_final"], bins=60, kde=True, ax=ax, label="final", alpha=0.35, color="green")
ax.set_title("Distribution du temps restant (τ)")
ax.set_xlabel("tau_true_min")
ax.set_xlim(0, 250)
ax.legend()

# 2. scatter
ax = axes[0, 1]
sample_plot = test_events.sample(min(6000, len(test_events)), random_state=RANDOM_STATE)
max_val = min(180, max(sample_plot["tau_true_min"].max(), sample_plot["tau_pred_final"].max()))
ax.scatter(sample_plot["tau_true_min"], sample_plot["tau_pred_final"], s=6, alpha=0.20, label="final")
ax.plot([0, max_val], [0, max_val], "r--", alpha=0.6, label="prédiction parfaite")
ax.set_title(f"Scatter plot final (corr = {metrics_final['Corr']:.3f})")
ax.set_xlabel("Tau réel (min)")
ax.set_ylabel("Tau prédit (min)")
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend()

# 3. erreur
ax = axes[0, 2]
error = test_events["tau_pred_final"] - test_events["tau_true_min"]
sns.histplot(error, bins=60, kde=True, ax=ax, color="purple")
ax.axvline(0, color="red", linestyle="--")
ax.set_title("Distribution de l'erreur de prédiction")
ax.set_xlabel("Erreur (min)")
ax.set_xlim(-100, 100)

# 4. MAE par aéroport
ax = axes[1, 0]
airport_plot = airport_df.melt(
    id_vars="airport",
    value_vars=["mae_hawkes", "mae_xgb", "mae_final"],
    var_name="model",
    value_name="MAE"
)
sns.barplot(data=airport_plot, x="airport", y="MAE", hue="model", ax=ax)
ax.set_title("MAE par aéroport")
ax.set_xlabel("Aéroport")
ax.set_ylabel("MAE (min)")
ax.tick_params(axis="x", rotation=20)

# 5. erreur relative médiane par intervalle de τ
ax = axes[1, 1]
test_events["tau_bin"] = pd.cut(test_events["tau_true_min"], bins=[0, 10, 30, 60, 120, 500], include_lowest=True)
rel_hawkes = test_events.groupby("tau_bin").apply(
    lambda x: np.median(np.abs(x["tau_true_min"] - x["tau_pred_tau_mean"]) / (x["tau_true_min"] + 1))
)
rel_final = test_events.groupby("tau_bin").apply(
    lambda x: np.median(np.abs(x["tau_true_min"] - x["tau_pred_final"]) / (x["tau_true_min"] + 1))
)
x = np.arange(len(rel_hawkes))
ax.bar(x - 0.2, rel_hawkes.values, width=0.4, label="hawkes")
ax.bar(x + 0.2, rel_final.values, width=0.4, label="final")
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in rel_hawkes.index], rotation=20)
ax.set_title("Erreur relative médiane par intervalle de τ")
ax.set_xlabel("Tau réel (min)")
ax.set_ylabel("Erreur relative médiane")
ax.legend()

# 6. probabilité de fin < 30 min
ax = axes[1, 2]
sample_plot2 = test_events.sample(min(6000, len(test_events)), random_state=RANDOM_STATE)
ax.scatter(sample_plot2["tau_true_min"], sample_plot2["tau_pred_p_finished_30min"], alpha=0.25, s=6)
ax.axvline(30, color="red", linestyle="--", alpha=0.6)
ax.set_title("Calibration de la probabilité de fin")
ax.set_xlabel("Tau réel (min)")
ax.set_ylabel("P(fin < 30 min) prédite")
ax.set_ylim(0, 1)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "hawkes_hybrid_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 13. IMPORTANCE DES VARIABLES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("12. IMPORTANCE DES VARIABLES")
print("=" * 60)

if hasattr(model_xgb, "feature_importances_"):
    feat_imp = pd.DataFrame({
        "feature": hybrid_features,
        "importance": model_xgb.feature_importances_
    }).sort_values("importance", ascending=False)

    print(feat_imp.head(20).to_string(index=False))

    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_imp.head(15), x="importance", y="feature")
    plt.title("Top variables - modèle XGBoost")
    plt.tight_layout()
    feat_plot_path = os.path.join(OUTPUT_DIR, "xgb_feature_importance.png")
    plt.savefig(feat_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
else:
    feat_imp = pd.DataFrame(columns=["feature", "importance"])

# ------------------------------------------------------------
# 14. SAUVEGARDE DES MODÈLES ET RÉSULTATS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("13. SAUVEGARDE DES RÉSULTATS")
print("=" * 60)

# tables
pred_path = os.path.join(OUTPUT_DIR, "hawkes_hybrid_predictions.csv")
airport_path = os.path.join(OUTPUT_DIR, "hawkes_hybrid_airport_stats.csv")
feat_imp_path = os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv")

metrics_df = pd.DataFrame([
    {
        "model": "hawkes",
        "MAE": metrics_hawkes["MAE"],
        "RMSE": metrics_hawkes["RMSE"],
        "Bias": metrics_hawkes["Bias"],
        "MedianRelativeError": metrics_hawkes["MedianRelativeError"],
        "Corr": metrics_hawkes["Corr"],
        "AccuracyFinished30": acc30_hawkes,
        "Coverage90": coverage_90_hawkes
    },
    {
        "model": "xgb",
        "MAE": metrics_xgb["MAE"],
        "RMSE": metrics_xgb["RMSE"],
        "Bias": metrics_xgb["Bias"],
        "MedianRelativeError": metrics_xgb["MedianRelativeError"],
        "Corr": metrics_xgb["Corr"],
        "AccuracyFinished30": acc30_xgb,
        "Coverage90": np.nan
    },
    {
        "model": "final",
        "MAE": metrics_final["MAE"],
        "RMSE": metrics_final["RMSE"],
        "Bias": metrics_final["Bias"],
        "MedianRelativeError": metrics_final["MedianRelativeError"],
        "Corr": metrics_final["Corr"],
        "AccuracyFinished30": acc30_final,
        "Coverage90": np.nan
    }
])
metrics_path = os.path.join(OUTPUT_DIR, "hawkes_hybrid_metrics.csv")

df_events.to_csv(pred_path, index=False)
airport_df.to_csv(airport_path, index=False)
feat_imp.to_csv(feat_imp_path, index=False)
metrics_df.to_csv(metrics_path, index=False)

# modèles
hawkes_model = {
    "mu": mu_opt,
    "alpha": alpha_opt,
    "beta": beta_opt,
    "delta": delta_opt,
    "gamma": GAMMA,
    "n_sim": N_SIM,
    "max_time_min": MAX_TIME_MIN,
    "intensity_threshold": INTENSITY_THRESHOLD
}

with open(os.path.join(OUTPUT_DIR, "hawkes_model.pkl"), "wb") as f:
    pickle.dump(hawkes_model, f)

with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(model_xgb, f)

with open(os.path.join(OUTPUT_DIR, "blend_config.pkl"), "wb") as f:
    pickle.dump({"hawkes_weight": best_w, "xgb_weight": 1 - best_w}, f)

with open(os.path.join(OUTPUT_DIR, "feature_list.pkl"), "wb") as f:
    pickle.dump(hybrid_features, f)

print(f"✅ Fichiers sauvegardés dans : {OUTPUT_DIR}")
print("   - hawkes_hybrid_predictions.csv")
print("   - hawkes_hybrid_airport_stats.csv")
print("   - hawkes_hybrid_metrics.csv")
print("   - xgb_feature_importance.csv")
print("   - hawkes_model.pkl")
print("   - xgb_model.pkl")
print("   - blend_config.pkl")
print("   - feature_list.pkl")
print("   - hawkes_hybrid_results.png")
print("   - xgb_feature_importance.png")

# ------------------------------------------------------------
# 15. SYNTHÈSE FINALE
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("14. SYNTHÈSE FINALE")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                RÉSULTATS HAWKES + HYBRIDE XGBOOST                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ Paramètres Hawkes :                                                ║
║   μ = {mu_opt:.6f} /min                                               ║
║   α = {alpha_opt:.6f} /min                                               ║
║   β = {beta_opt:.6f} /min                                               ║
║   δ = {delta_opt:.6f} /min                                               ║
║   γ = {GAMMA:.6f} /min                                               ║
║   Blend : Hawkes={best_w:.2f} | XGB={1-best_w:.2f}                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ HAWKES (TEST)                                                      ║
║   MAE   = {metrics_hawkes['MAE']:.2f} min                                             ║
║   RMSE  = {metrics_hawkes['RMSE']:.2f} min                                             ║
║   Corr  = {metrics_hawkes['Corr']:.3f}                                                  ║
║   Acc30 = {acc30_hawkes:.2%}                                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ XGBOOST (TEST)                                                     ║
║   MAE   = {metrics_xgb['MAE']:.2f} min                                             ║
║   RMSE  = {metrics_xgb['RMSE']:.2f} min                                             ║
║   Corr  = {metrics_xgb['Corr']:.3f}                                                  ║
║   Acc30 = {acc30_xgb:.2%}                                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ FINAL (TEST)                                                       ║
║   MAE   = {metrics_final['MAE']:.2f} min                                             ║
║   RMSE  = {metrics_final['RMSE']:.2f} min                                             ║
║   Corr  = {metrics_final['Corr']:.3f}                                                  ║
║   Acc30 = {acc30_final:.2%}                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")