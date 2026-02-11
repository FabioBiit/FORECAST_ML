# 📊 Model Analysis Guide

Questa guida spiega come utilizzare gli strumenti di analisi del modello ML creati per il progetto di forecasting.

---

## 📁 File di Analisi

### 1. `analyze_model.py`
Analizza il modello addestrato e genera **8 visualizzazioni** dettagliate.

**Cosa genera:**
- ✅ Predizioni vs valori reali nel tempo (validation & test)
- ✅ Feature importance (top 15 feature più importanti)
- ✅ Analisi dei residui (4 plot: timeline, istogramma, scatter, Q-Q plot)
- ✅ Scatter plot actual vs predicted con R²
- ✅ Errori per ora del giorno (per identificare pattern temporali)

### 2. `test_horizons.py`
Testa il modello con **diversi orizzonti di previsione** (6h, 12h, 24h, 48h, 72h, 168h).

**Cosa fa:**
- ✅ Addestra modelli per ogni orizzonte
- ✅ Confronta performance naive vs XGBoost
- ✅ Identifica l'orizzonte ottimale
- ✅ Salva risultati in CSV

### 3. `run_analysis.py`
Script wrapper per eseguire rapidamente l'analisi completa.

---

## 🚀 Come Eseguire le Analisi

### Prerequisito: Modello Addestrato
Prima di eseguire le analisi, assicurati di aver addestrato il modello:

```bash
cd ml-forecasting-service
python function_app.py
```

Questo genererà il file `models/xgb_power_h24.joblib`.

---

## 🎯 Uso Pratico

### Analisi Standard (Raccomandato)

```bash
cd ml-forecasting-service
python run_analysis.py
```

Questo genererà tutti i plot nella cartella `plots/`:
- `val_predictions_time.png`
- `test_predictions_time.png`
- `feature_importance.png`
- `val_residuals.png`
- `test_residuals.png`
- `val_scatter.png`
- `test_scatter.png`
- `error_by_hour.png`

---

### Test di Orizzonti Multipli

Per trovare l'orizzonte di previsione ottimale:

```bash
cd ml-forecasting-service
python -m training.test_horizons
```

Output esempio:
```
SUMMARY: PERFORMANCE ACROSS HORIZONS
================================================================================

Horizon      Val MAE      Test MAE     Val Improv.     Test Improv.
--------------------------------------------------------------------------------
  6h      0.4123       0.3856       25.3%           18.7%
 12h      0.4567       0.4012       23.1%           16.2%
 24h      0.5177       0.4256       22.0%           13.5%
 48h      0.6234       0.5123       18.9%           11.8%
 72h      0.7012       0.6234       15.2%            9.3%
168h      0.8901       0.7845       10.1%            5.2%

BEST HORIZONS
================================================================================
Best on Validation: 6h (MAE=0.4123)
Best on Test:       6h (MAE=0.3856)
```

Risultati salvati in: `models/horizon_comparison.csv`

---

### Esecuzione Singola (Avanzato)

Se vuoi eseguire solo una specifica analisi:

```bash
# Solo visualizzazioni
python -m training.analyze_model

# Solo test orizzonti
python -m training.test_horizons
```

---

## 📈 Interpretazione dei Risultati

### 1. Predictions Over Time
**File:** `val_predictions_time.png`, `test_predictions_time.png`

**Cosa guardare:**
- ✅ Le linee si sovrappongono? → Buona predizione
- ❌ Grandi discrepanze? → Modello non cattura pattern
- 🔍 Pattern sistematici di errore? → Possibili miglioramenti

### 2. Feature Importance
**File:** `feature_importance.png`

**Interpretazione:**
- Le **lag features** (lag_24, lag_168) sono solitamente le più importanti
- **Rolling statistics** catturano trend e volatilità
- **Calendar features** (hour, weekday) catturano stagionalità

**Esempio:**
```
Top Features:
1. lag_24       → Valore 24h fa (autocorrelazione forte)
2. lag_168      → Valore 1 settimana fa (stagionalità settimanale)
3. roll_mean_24 → Media mobile 24h (trend recente)
4. hour         → Ora del giorno (pattern giornaliero)
```

### 3. Residuals Analysis
**File:** `val_residuals.png`, `test_residuals.png`

**4 Sub-plot:**

**A) Residuals Over Time**
- ✅ Random intorno a zero → Buon modello
- ❌ Pattern temporali → Modello manca qualcosa

**B) Histogram**
- ✅ Distribuzione normale centrata su zero → Ottimo
- ❌ Asimmetrica o bimodale → Problemi

**C) Predicted vs Residuals**
- ✅ Nuvola uniforme → Buon modello
- ❌ Pattern a imbuto → Eteroschedasticità (errore varia con predizione)

**D) Q-Q Plot**
- ✅ Punti su linea retta → Residui normali
- ❌ Deviazioni → Outliers o distribuzione non normale

### 4. Actual vs Predicted Scatter
**File:** `val_scatter.png`, `test_scatter.png`

**Interpretazione:**
- **R² ≈ 1.0** → Perfetto
- **R² > 0.8** → Molto buono
- **R² > 0.6** → Accettabile
- **R² < 0.5** → Modello debole

I punti dovrebbero essere vicini alla linea rossa (perfect prediction).

### 5. Error by Hour
**File:** `error_by_hour.png`

**2 Sub-plot:**

**A) MAE by Hour**
- Identifica **ore difficili da predire**
- Ore di picco (mattina/sera) hanno spesso errori maggiori

**B) Bias by Hour**
- Valori sopra zero → Modello **sottostima** in quelle ore
- Valori sotto zero → Modello **sovrastima** in quelle ore
- Vicino a zero → **Non biased**

---

## 🎓 Best Practices

### Dopo l'analisi, considera:

1. **Se feature importance mostra feature inutili:**
   - Rimuovile per semplificare il modello
   - Riduce overfitting e migliora velocità

2. **Se residui mostrano pattern:**
   - Aggiungi nuove feature (es: festività, meteo)
   - Prova modelli più complessi (LSTM, Prophet)

3. **Se errori variano per ora:**
   - Considera modelli separati per diversi periodi
   - Aggiungi feature specifiche per quelle ore

4. **Se test >> validation:**
   - Possibile overfitting
   - Riduci complessità modello o aggiungi regolarizzazione

---

## 📞 Troubleshooting

### Errore: "Model not found"
```bash
# Soluzione: Addestra prima il modello
python function_app.py
```

### Errore: "No module named 'scipy'"
```bash
# Soluzione: Installa scipy
pip install scipy
```

### Plot non vengono salvati
```bash
# Verifica che la cartella plots/ sia creata
# Controlla i permessi di scrittura
```

---

## 🔄 Workflow Completo

```bash
# 1. Pipeline completa (ingestion → training)
python function_app.py

# 2. Analisi e visualizzazioni
python run_analysis.py

# 3. (Opzionale) Test orizzonti multipli
python -m training.test_horizons

# 4. Rivedi i plot in plots/
explorer plots  # Windows
open plots      # Mac
```

---

## 📚 Risorse Utili

- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Time Series Forecasting:** https://otexts.com/fpp3/
- **Feature Engineering:** https://scikit-learn.org/stable/modules/preprocessing.html

---

**Buona analisi! 📊🚀**
