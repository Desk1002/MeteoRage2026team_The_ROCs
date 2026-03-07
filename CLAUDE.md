# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexte du projet

Data Battle 2026 — Prédiction probabiliste de la fin des alertes orage autour d'aéroports (Météorage).

**Objectif** : estimer la probabilité qu'un éclair CG soit le dernier d'une alerte (`is_last_lightning_cloud_ground == True`), afin de lever les alertes plus tôt que la règle fixe de 30 min.

**Métrique principale** : Brier Score (calibration). Secondaire : AUC-ROC. Ne jamais utiliser l'accuracy.

## Environnement Python

Utiliser l'environnement Anaconda `anaconda3_henoc` :
```bash
/c/Users/user/anaconda3_henoc/python script.py
/c/Users/user/anaconda3_henoc/python -m jupyter notebook
```

Librairies principales : `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `lightgbm`, `xgboost`, `optuna`.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `segment_alerts_all_airports_train.csv` | Données d'entraînement (507 071 éclairs) |
| `utils.py` | Fonctions utilitaires — feature engineering, clustering, évaluation, visualisation |
| `analyse.ipynb` | EDA complet (exploration, stats, graphiques) |
| `modelisation.ipynb` | Pipeline de modélisation v2 (NHPP, LogReg, LightGBM, XGBoost, Stacking) |
| `Info data.docx` | Documentation officielle des colonnes et des zones |
| `MeteoRage2026team_The_ROCs/` | Dépôt Git de l'équipe (miroir partagé, `figures/` contient les graphiques exportés) |

**Note** : `utils.py` à la racine et `MeteoRage2026team_The_ROCs/utils.py` doivent rester synchronisés.

## Architecture des données

**Zones** :
- **Zone alerte** : < 20 km, éclairs CG uniquement → `airport_alert_id` non nul
- **Zone contexte** : 20–30 km, tous éclairs (IC + CG) → `airport_alert_id` nul
- Le dataset couvre **30 km** au total (pas 50 km)

**Colonnes du CSV** (noms exacts vérifiés sur le fichier réel) :

| Colonne | Type | Description |
|---|---|---|
| `lightning_id` | int64 | Identifiant global d'un éclair dans le fichier |
| `lightning_airport_id` | int64 | Identifiant d'un éclair **par aéroport** (pas global) |
| `date` | object→datetime | Horodatage UTC de l'éclair |
| `lon` / `lat` | float64 | Position WGS84 de l'éclair |
| `amplitude` | float64 | Polarité + intensité max du courant (kA) |
| `maxis` | float64 | Erreur de localisation théorique en km |
| `icloud` | bool | `False` = CG (nuage-sol) ; `True` = IC (intra-nuage) |
| `dist` | float64 | Distance à l'aéroport en km |
| `azimuth` | float64 | Direction depuis l'aéroport (0°=N, 90°=E, 180°=S, 270°=O) |
| `airport` | object | Nom de l'aéroport |
| `airport_alert_id` | float64 | Numéro d'alerte **par aéroport** (NaN pour zone contexte) |
| `is_last_lightning_cloud_ground` | object | Cible : `True` = dernier CG de l'alerte (type `object`, pas `bool`) |

> ⚠️ `airport_alert_id` et `is_last_lightning_cloud_ground` ne sont remplis que pour les éclairs < 20 km (zone alerte). Toujours utiliser `== True` et non `is True` sur `is_last_lightning_cloud_ground`.

> ⚠️ Le docx officiel nomme cette colonne `alert_airport_id` — le nom réel dans le CSV est `airport_alert_id`.

**Aéroports** — 5 dans le training set (Bron absent du CSV train, probablement dans les données de test) :

| Aéroport | Encodage | Longitude | Latitude |
|---|---|---|---|
| Ajaccio | 0 | 8.8029 | 41.9236 |
| Bastia | 1 | 9.4837 | 42.5527 |
| Biarritz | 2 | -1.524 | 43.4683 |
| Nantes | 3 | -1.6107 | 47.1532 |
| Pise | 4 | 10.399 | 43.695 |
| **Bron** | — | 4.9389 | 45.7294 |

Biarritz est le plus difficile à modéliser (LightGBM par aéroport).

**Distinctions critiques** :
- **56 599** = éclairs CG en zone alerte (lignes avec `airport_alert_id` non nul)
- **2 627** = nombre d'alertes distinctes (`df[df["is_last_lightning_cloud_ground"]==True].shape[0]`)
- Déséquilibre 1:20 (4.6% de True)
- Training set couvre **2016–2022** (le docx mentionne 2016–2025, mais les données de test/compétition couvrent 2023+)

**Anomalie connue** : Pise 2016 a un système d'enregistrement différent pour les IC. Exclure Pise 2016 pour les features basées sur le ratio IC/CG.

**Discordance documentaire** : le sujet officiel mentionne 50 km et 230K éclairs ; le dataset réel couvre 30 km et contient 507 071 éclairs.

## Architecture de utils.py

**`compute_features(df)`** — features CG intra-alerte (sans fuite de données, tout en causal via `shift(1)`) :
- Inter-temps (stats glissantes, ratio, z-score, trend, accélération)
- Contexte alerte (`n_cg_cumul`, `alert_dur_min`)
- Fenêtres glissantes 5/10/15 min (`n_cg_5m`, `n_cg_10m`, `n_cg_15m`)
- Amplitude (`amp_abs`, `is_positive`, `pct_pos_cumul`, `amp_trend`)
- Distance/azimuth (`mean_dist_5/10`, `dist_trend`, `az_dispersion`)
- Centroïde XY + vecteur de déplacement du cœur d'orage

**`compute_surrounding_features(df_alert, df_all)`** — features IC et CG entourant :
- `df_alert` = éclairs CG zone alerte uniquement ; `df_all` = **dataset complet** (toutes les 507 071 lignes, zone alerte + contexte)
- `n_ic_{5m/15m/30m}` : éclairs IC dans 0–30 km
- `n_cg_surr_{w}m` : éclairs CG dans 20–30 km
- `ratio_surr_{w}m` : signature IC/(IC+CG)

**`cluster_storms(df)`** — K-Means 4 clusters par alerte (entraîné sur 2016–2020 uniquement).

**`evaluate_model(name, y_true, y_proba)`** — renvoie Brier, AUC-ROC, Log-loss, AP.

**`business_metric(df_eval, y_proba, threshold)`** — minutes gagnées vs règle 30 min. `df_eval` doit avoir une colonne `is_last` (booléen), pas `is_last_lightning_cloud_ground`.

## Pipeline de modélisation (modelisation.ipynb)

**Split temporel strict** : Train 2016–2020 | Val 2021 | Test 2022. Ne jamais mélanger.

Modèles dans l'ordre :
1. Baseline (seuil 30 min)
2. NHPP (Non-Homogeneous Poisson Process)
3. Logistic Regression calibrée
4. LightGBM tuné Optuna (50 essais)
5. XGBoost tuné Optuna (50 essais)
6. LightGBM par aéroport (5 modèles spécialisés)
7. Stacking v2 (méta-learner sur 5 bases)

## Contexte physique (issu de la présentation Météorage)

**Fonctionnement de l'alerte opérationnelle** :
1. Déclenchement dès le premier CG dans la zone 20 km
2. Maintien actif tant qu'un CG survient dans les 30 min (**TTC = Time To Clear**)
3. Levée automatique 30 min après le dernier CG → c'est ce délai qu'on cherche à réduire

**Physique des éclairs IC vs CG** :
- Les IC émettent en **VHF** (haute fréquence), les CG en **LF** (1–400 kHz)
- L'activité IC **précède et succède** l'arc en retour → un ratio IC/(IC+CG) élevé puis décroissant est un signal de fin de cellule
- C'est la justification physique de la feature `ratio_surr_{w}m`
- Précision réseau Météorage : localisation médiane **100 m** pour CG, détection CG **>98%**, différenciation CG/IC **>90%** → données très fiables

**Types d'orages dans le dataset** :
- **Frontaux** (majorité) : systèmes organisés, trajectoire lisible dans les éclairs → bien modélisés
- **Orographiques** (minorité) : formation rapide sous l'influence des reliefs (Alpes, Pyrénées, Corse) → plus difficiles à anticiper → Ajaccio, Bastia, Biarritz concernés

**Métriques métier Météorage** (référence pour interpréter `business_metric`) :

| Métrique | Définition |
|---|---|
| **TTC** | Time To Clear — durée de maintien (30 min dans notre cas) |
| **LRE** | Last Related Event — dernier éclair dans la zone environnante |
| **FTWR** | False Threat Warning Rate — taux de fausses levées d'alerte (= sécurité) |
| **POD** | Probability of Detection = 1 − FTWR |
| **POD20'** | % d'alertes avec ≥ 20 min d'anticipation sur la fin réelle → métrique la plus proche de notre `business_metric` |

Résultats observés par Météorage (zone montagne) : POD typiquement **94–100%**, POD20' entre **71–97%**. Notre modèle doit maintenir un POD proche de 100% pour ne pas lever l'alerte avant la fin réelle.

## Règles absolues

- Toutes les features doivent être calculées avec uniquement les éclairs **antérieurs** à l'éclair courant (pas de fuite de données).
- `airport_alert_id` n'est **jamais** utilisé seul pour `nunique()` — toujours groupé par `airport`.
- La "Proportion alerte" à 11.2% correspond à la proportion d'**éclairs CG en zone alerte** sur le total, pas au nombre d'alertes.
- Ne jamais lever l'alerte avant le dernier CG réel : le coût d'un faux négatif (avion frappé) est bien supérieur au coût d'un faux positif (attente inutile).
