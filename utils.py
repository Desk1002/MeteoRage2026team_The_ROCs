"""
utils.py — Data Battle 2026 : fonctions utilitaires pour la modélisation des alertes orage
Contient : feature engineering, clustering, évaluation, métriques métier, visualisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, log_loss,
    average_precision_score, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# ──────────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING — CG intra-alerte (inter-temps, amplitude, fenêtres,
#    trajectoire, centroïde + vecteur de déplacement)
# ──────────────────────────────────────────────────────────────────────────────

def sliding_count_vec(dates_ns, window_ns):
    """Sliding count vectorisé — O(n log n), zéro boucle Python."""
    n = len(dates_ns)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    starts = np.searchsorted(dates_ns, dates_ns - window_ns, side='left')
    return (np.arange(n, dtype=np.int32) - starts)


def compute_features(df):
    """
    Features CG intra-alerte sans fuite de données :
      - Inter-temps (stats glissantes)
      - Contexte alerte (n_cg_cumul, durée)
      - Fenêtres glissantes 5/10/15 min
      - Amplitude
      - Distance / azimuth
      - Trajectoire : dist_trend, az_dispersion
      - [NEW] Centroïde XY + vecteur de déplacement du cœur d'orage
      - Encodage temporel et aéroport
    """
    df = df.sort_values(['airport', 'airport_alert_id', 'date']).copy()
    grp = df.groupby(['airport', 'airport_alert_id'], sort=False)

    # ── INTER-TEMPS ─────────────────────────────────────────────────────────
    df['inter_time_s']   = grp['date'].diff().dt.total_seconds().fillna(0)
    df['inter_time_log'] = np.log1p(df['inter_time_s'])

    df['median_inter'] = grp['inter_time_s'].transform(
        lambda x: x.expanding().median().shift(1).fillna(0))
    df['mean_inter']   = grp['inter_time_s'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0))
    df['std_inter']    = grp['inter_time_s'].transform(
        lambda x: x.expanding().std().shift(1).fillna(0))
    df['max_inter']    = grp['inter_time_s'].transform(
        lambda x: x.expanding().max().shift(1).fillna(0))

    df['ratio_median']  = df['inter_time_s'] / (df['median_inter'] + 1e-9)
    df['ratio_mean']    = df['inter_time_s'] / (df['mean_inter']   + 1e-9)
    df['z_score_inter'] = (df['inter_time_s'] - df['mean_inter']) / (df['std_inter'] + 1e-9)

    df['inter_trend'] = grp['inter_time_s'].transform(
        lambda x: x.rolling(5, min_periods=2).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True
        ).shift(1).fillna(0))
    df['prev_inter']  = grp['inter_time_s'].transform(lambda x: x.shift(1).fillna(0))
    df['inter_accel'] = df['inter_time_s'] - df['prev_inter']

    # ── CONTEXTE DE L'ALERTE ────────────────────────────────────────────────
    df['n_cg_cumul']    = grp.cumcount()
    df['alert_dur_min'] = grp['date'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60)

    # ── AMPLITUDE ───────────────────────────────────────────────────────────
    df['amp_abs']       = df['amplitude'].abs()
    df['is_positive']   = (df['amplitude'] > 0).astype(np.int8)
    df['pct_pos_cumul'] = grp['is_positive'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0))
    df['pct_pos_last5'] = grp['is_positive'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(0))
    df['amp_trend']     = grp['amplitude'].transform(
        lambda x: x.rolling(5, min_periods=2).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True
        ).shift(1).fillna(0))

    # ── FENÊTRES GLISSANTES (vectorisées) ───────────────────────────────────
    for w_min, w_s in [(5, 300), (10, 600), (15, 900)]:
        w_ns = int(w_s * 1e9)
        col  = f'n_cg_{w_min}m'
        df[col] = grp['date'].transform(
            lambda x: pd.Series(
                sliding_count_vec(x.sort_values().values.astype(np.int64), w_ns),
                index=x.sort_values().index
            ).reindex(x.index)
        )

    # ── DISTANCE / AZIMUTH ──────────────────────────────────────────────────
    # maxis = erreur de localisation théorique en km (≠ azimuth)
    df['mean_dist_5']   = grp['dist'].transform(
        lambda x: x.rolling(5,  min_periods=1).mean().shift(1).bfill().fillna(0))
    df['mean_dist_10']  = grp['dist'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1).bfill().fillna(0))
    df['dist_vs_mean5'] = df['dist'] - df['mean_dist_5']
    df['dist_trend']    = grp['dist'].transform(
        lambda x: x.rolling(5, min_periods=2).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True
        ).shift(1).fillna(0))

    # azimuth = direction en degrés (0=N, 90=E, 180=S, 270=O)
    # maxis   = erreur de localisation théorique en km (feature séparée)
    df['az_dispersion'] = grp['azimuth'].transform(
        lambda x: x.rolling(10, min_periods=2).std().shift(1).fillna(0))

    # ── [NEW] CENTROÏDE XY + VECTEUR DE DÉPLACEMENT ─────────────────────────
    # Projection polaire → cartésienne (origine = aéroport)
    # x = Est, y = Nord  (convention géographique standard)
    az_rad      = np.radians(df['azimuth'])
    df['x_pos'] = df['dist'] * np.sin(az_rad)
    df['y_pos'] = df['dist'] * np.cos(az_rad)

    # Centroïde des 5 et 10 derniers éclairs (causal : shift(1))
    for w in [5, 10]:
        df[f'centroid_x_{w}'] = grp['x_pos'].transform(
            lambda x: x.rolling(w, min_periods=1).mean().shift(1).fillna(0))
        df[f'centroid_y_{w}'] = grp['y_pos'].transform(
            lambda x: x.rolling(w, min_periods=1).mean().shift(1).fillna(0))

    # Distance du centroïde 5 à l'aéroport (+ le centre se rapproche = fin ?)
    df['centroid_dist_ap'] = np.sqrt(
        df['centroid_x_5']**2 + df['centroid_y_5']**2)

    # Vecteur de déplacement centroïde 5→10 (vitesse apparente du cœur)
    df['centroid_dx']    = df['centroid_x_5'] - df['centroid_x_10']
    df['centroid_dy']    = df['centroid_y_5'] - df['centroid_y_10']
    df['centroid_speed'] = np.sqrt(df['centroid_dx']**2 + df['centroid_dy']**2)

    # Projection radiale : < 0 → le cœur se rapproche, > 0 → il s'éloigne
    denom = df['centroid_dist_ap'] + 1e-9
    df['centroid_approach'] = (
        df['centroid_x_10'] * df['centroid_dx'] +
        df['centroid_y_10'] * df['centroid_dy']
    ) / denom

    # ── TEMPOREL & ENCODAGE ─────────────────────────────────────────────────
    df['month']       = df['date'].dt.month
    df['hour']        = df['date'].dt.hour
    df['season']      = ((df['month'] % 12) // 3).astype(np.int8)
    airport_enc       = {'Ajaccio': 0, 'Bastia': 1, 'Biarritz': 2, 'Nantes': 3, 'Pise': 4}
    df['airport_enc'] = df['airport'].map(airport_enc).astype(np.int8)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. FEATURES IC ET CG ENTOURANT (zone 0–30 km)
# ──────────────────────────────────────────────────────────────────────────────

def compute_surrounding_counts(alert_t_ns, ref_t_ns, window_ns):
    """Pour chaque T dans alert_t_ns, compte ref_t_ns dans [T-window, T). Arrays triés."""
    ends   = np.searchsorted(ref_t_ns, alert_t_ns,              side='right')
    starts = np.searchsorted(ref_t_ns, alert_t_ns - window_ns,  side='left')
    return (ends - starts).astype(np.int32)


def compute_surrounding_features(df_alert, df_all):
    """
    Ajoute des features IC et CG entourant pour chaque éclair CG d'alerte.

    - n_ic_{w}m       : éclairs IC dans 0–30 km, fenêtre w min
    - n_cg_surr_{w}m  : éclairs CG dans 20–30 km, fenêtre w min
    - ratio_surr_{w}m : IC / (IC + CG_surr) → signature intra-nuage
    """
    df_alert = df_alert.sort_values(['airport', 'date']).copy()

    df_ic   = df_all[df_all['icloud'] == True].sort_values(['airport', 'date'])
    df_cg_s = df_all[
        (~df_all['icloud']) & (df_all['dist'].between(20, 30))
    ].sort_values(['airport', 'date'])

    ic_by_ap  = {ap: g['date'].values.astype(np.int64)
                 for ap, g in df_ic.groupby('airport')}
    cgs_by_ap = {ap: g['date'].values.astype(np.int64)
                 for ap, g in df_cg_s.groupby('airport')}

    WINDOWS = {'5m': 300, '15m': 900, '30m': 1800}

    for col_pfx, by_ap in [('n_ic', ic_by_ap), ('n_cg_surr', cgs_by_ap)]:
        for w_name, w_s in WINDOWS.items():
            w_ns   = int(w_s * 1e9)
            result = np.zeros(len(df_alert), dtype=np.int32)
            for ap in df_alert['airport'].unique():
                mask  = (df_alert['airport'] == ap).values
                t_ns  = df_alert.loc[mask, 'date'].values.astype(np.int64)
                ref   = by_ap.get(ap, np.array([], dtype=np.int64))
                result[mask] = compute_surrounding_counts(t_ns, ref, w_ns)
            df_alert[f'{col_pfx}_{w_name}'] = result

    for w_name in WINDOWS:
        ic  = df_alert[f'n_ic_{w_name}']
        cgs = df_alert[f'n_cg_surr_{w_name}']
        df_alert[f'ratio_surr_{w_name}'] = ic / (ic + cgs + 1.0)

    return df_alert


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING DES TYPES D'ORAGES (K-Means 4 clusters)
# ──────────────────────────────────────────────────────────────────────────────

def cluster_storms(df, n_clusters=4, seed=42):
    """
    K-Means sur features agrégées par alerte.
    Entraîné sur 2016–2020 pour éviter le leakage temporel.
    Clusters typiques : frontal intense | orographique court | convectif long | mixte
    """
    alert_level = df.groupby(['airport', 'airport_alert_id']).agg(
        duration_min   = ('alert_dur_min',  'max'),
        n_cg           = ('n_cg_cumul',     'max'),
        median_inter_s = ('inter_time_s',   'median'),
        max_amp        = ('amplitude',       lambda x: x.abs().max()),
        mean_dist      = ('dist',           'mean'),
        std_dist       = ('dist',           'std'),
        hour_start     = ('hour',           'first'),
        month          = ('month',          'first'),
        year           = ('date',           lambda x: x.dt.year.iloc[0]),
    ).reset_index().fillna(0)

    CLUST_FEATS = ['duration_min', 'n_cg', 'median_inter_s',
                   'max_amp', 'mean_dist', 'std_dist', 'hour_start', 'month']

    X_all   = alert_level[CLUST_FEATS].values
    tr_mask = alert_level['year'] <= 2020

    scaler = StandardScaler().fit(X_all[tr_mask])
    X_sc   = scaler.transform(X_all)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    km.fit(X_sc[tr_mask])
    alert_level['storm_cluster'] = km.predict(X_sc)

    df = df.merge(
        alert_level[['airport', 'airport_alert_id', 'storm_cluster']],
        on=['airport', 'airport_alert_id'], how='left'
    )
    df['storm_cluster'] = df['storm_cluster'].fillna(0).astype(np.int8)

    print('=== Clusters d\'orages (train ≤ 2020) ===')
    for c in range(n_clusters):
        mask = (alert_level['year'] <= 2020) & (alert_level['storm_cluster'] == c)
        sub  = alert_level[mask]
        print(f'  Cluster {c} (n={mask.sum():3d}) : '
              f'durée={sub["duration_min"].mean():.0f}min, '
              f'n_CG={sub["n_cg"].mean():.0f}, '
              f'inter_med={sub["median_inter_s"].mean():.0f}s, '
              f'dist={sub["mean_dist"].mean():.1f}km, '
              f'heure={sub["hour_start"].mean():.0f}h')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. ÉVALUATION ET MÉTRIQUES MÉTIER
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, y_true, y_proba):
    """Calcule toutes les métriques d'évaluation."""
    y_true  = np.array(y_true)
    y_proba = np.clip(np.array(y_proba), 1e-7, 1 - 1e-7)
    return {
        'Modèle':     name,
        'Brier ↓':    round(brier_score_loss(y_true, y_proba), 4),
        'AUC-ROC ↑':  round(roc_auc_score(y_true, y_proba), 4),
        'Log-loss ↓': round(log_loss(y_true, y_proba), 4),
        'AP ↑':       round(average_precision_score(y_true, y_proba), 4),
    }


def business_metric(df_eval, y_proba, threshold=0.5):
    """
    Métrique métier : minutes gagnées en moyenne vs règle 30 min.
    Pour chaque alerte :
      - t_real_end : date du vrai dernier CG
      - t_pred     : date du 1er éclair avec P >= threshold
      - gain       : max(t_real_end - t_pred, 0) en minutes
    """
    df_eval = df_eval.copy()
    df_eval['proba'] = y_proba
    df_eval['flag']  = df_eval['proba'] >= threshold

    savings, false_alarms = [], []
    for (airport, alert_id), grp in df_eval.groupby(['airport', 'airport_alert_id']):
        grp = grp.sort_values('date')
        last_true = grp[grp['is_last'] == True]
        if len(last_true) == 0:
            continue
        t_real_end = last_true['date'].values[0]

        flagged = grp[grp['flag']]
        if len(flagged) == 0:
            false_alarms.append(0)
            continue
        t_pred = flagged['date'].values[0]
        saving_min = (pd.Timestamp(t_real_end) - pd.Timestamp(t_pred)).total_seconds() / 60
        savings.append(max(saving_min, 0))
        false_alarms.append(1 if t_pred < t_real_end else 0)

    return {
        'Gain moyen (min)':  round(np.mean(savings),   1) if savings else 0,
        'Gain médian (min)': round(np.median(savings),  1) if savings else 0,
        'Faux positifs (%)': round(np.mean(false_alarms) * 100, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_calibration_roc(y_true, probas_dict, title=''):
    """Courbes de calibration, ROC et Précision-Rappel pour tous les modèles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(probas_dict)))

    for (name, y_prob), c in zip(probas_dict.items(), colors):
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

        frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        axes[0].plot(mean_pred, frac, 's-', color=c, label=name)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, color=c, label=f'{name} ({auc:.3f})')

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[2].plot(rec, prec, color=c, label=f'{name} (AP={ap:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Parfait')
    axes[0].set(xlabel='P prédite', ylabel='Fraction observée',
                title='Courbes de calibration')
    axes[0].legend(fontsize=8)

    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set(xlabel='FPR', ylabel='TPR', title='Courbes ROC')
    axes[1].legend(fontsize=8)

    axes[2].set(xlabel='Rappel', ylabel='Précision', title='Courbes Précision-Rappel')
    axes[2].legend(fontsize=8)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
