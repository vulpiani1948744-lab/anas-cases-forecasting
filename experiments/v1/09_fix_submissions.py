"""
09 - Riformatta tutte le submission per matchare l'esempio:
- nessun header
- una riga per predizione
- CRLF line endings (come example_submission.txt)
- 15048 righe esatte

Sovrascrive i file submission_*.txt esistenti.
"""

from pathlib import Path
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_DIR / "output"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"  # dati grezzi della competizione (non versionati)

EXPECTED_ROWS = 15048

# verifica l'esempio
with open(DATA_DIR / "example_submission.txt", "rb") as f:
    example_bytes = f.read()
print(f"Esempio: {len(example_bytes)} byte, prime 50 chars: {example_bytes[:50]!r}")
newline = b"\n"
print(f"Linee esempio: {example_bytes.count(newline)}")

for sub_file in sorted(OUT_DIR.glob("submission_*.txt")):
    df = pd.read_csv(sub_file)
    # se c'è la colonna num_richieste, prendi quella; altrimenti assumi singola colonna
    if "num_richieste" in df.columns:
        values = df["num_richieste"].values
    else:
        values = df.iloc[:, 0].values

    assert len(values) == EXPECTED_ROWS, f"{sub_file.name}: ha {len(values)} righe invece di {EXPECTED_ROWS}"

    # scrivi senza header, con CRLF, una riga per valore
    lines = [f"{v}" for v in values]
    content = "\r\n".join(lines) + "\r\n"
    sub_file.write_text(content)

    # ricontrollo
    n_lines = sub_file.read_bytes().count(b"\n")
    print(f"OK {sub_file.name}: {n_lines} linee, {sub_file.stat().st_size} byte")

print("\nDone. Tutte le submission sono ora nel formato dell'esempio.")
