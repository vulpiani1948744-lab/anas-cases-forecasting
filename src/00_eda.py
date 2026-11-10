"""
00 - Diagnostica statistica del target (per giustificare il modello nel report).

Caratterizza la natura matematica di Y = num_richieste (un CONTEGGIO mensile):
- % di zeri (zero-inflation)
- media, varianza e indice di dispersione var/media (sovradispersione di Poisson)
- asimmetria (skewness) e quantili
- esito: se var >> media e molti zeri -> la loss corretta e' la Tweedie deviance
  (compound Poisson-Gamma, 1<p<2), non l'MSE gaussiana.

Richiede output/target_mensile.csv (lanciare prima 01_target.py).
Output: output/eda_target.json  (numeri pronti per il report)
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import config as C

y = pd.read_csv(C.OUT / "target_mensile.csv")["num_richieste"].astype(float)

mean, var = y.mean(), y.var()
disp = var / mean
diag = {
    "n_osservazioni": int(len(y)),
    "pct_zeri": round((y == 0).mean() * 100, 2),
    "media": round(mean, 3),
    "varianza": round(var, 3),
    "indice_dispersione_var_su_media": round(disp, 2),
    "skewness": round(float(stats.skew(y)), 2),
    "kurtosis": round(float(stats.kurtosis(y)), 2),
    "max": int(y.max()),
    "quantili": {q: float(np.quantile(y, q)) for q in [0.5, 0.9, 0.95, 0.99]},
}

print("=" * 60)
print("DIAGNOSTICA DEL TARGET (num_richieste = conteggio mensile)")
print("=" * 60)
for k, v in diag.items():
    print(f"  {k}: {v}")

print("\nINTERPRETAZIONE STATISTICA:")
print(f"  - {diag['pct_zeri']}% di zeri -> dati ZERO-INFLATED")
if disp > 1.2:
    print(f"  - var/media = {disp:.1f} >> 1 -> SOVRADISPERSIONE (no Poisson puro)")
    print("  - distribuzione asimmetrica e a coda pesante")
    print("  => loss corretta: TWEEDIE DEVIANCE (compound Poisson-Gamma, 1<p<2)")
    print("     o Binomiale Negativa. NO MSE gaussiana (predirrebbe valori negativi).")
else:
    print(f"  - var/media = {disp:.1f} ~ 1 -> Poisson adeguato")

json.dump(diag, open(C.OUT / "eda_target.json", "w"), indent=2)
print(f"\n-> output/eda_target.json")
