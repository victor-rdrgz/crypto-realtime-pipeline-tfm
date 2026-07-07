import requests
import json
import csv

url = "https://api.binance.com/api/v3/exchangeInfo"
data = requests.get(url, timeout=30).json()

pairs = []
bases = set()
quotes = set()

for s in data.get("symbols", []):
    # Filtra solo pares spot que estén activos
    perms = s.get("permissions") or []
    if s.get("status") != "TRADING":
        continue
    if "SPOT" not in perms and not s.get("isSpotTradingAllowed", False):
        continue

    sym  = s.get("symbol")
    base = s.get("baseAsset")
    quote= s.get("quoteAsset")
    if not (sym and base and quote):
        continue

    pairs.append({"symbol": sym, "baseAsset": base, "quoteAsset": quote})
    bases.add(base)
    quotes.add(quote)

# Ordena por símbolo para estabilidad
pairs.sort(key=lambda x: x["symbol"])

# JSON reducido (ligero y fácil de subir)
with open("binance_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

# CSV opcional
with open("binance_pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["symbol", "baseAsset", "quoteAsset"])
    w.writeheader()
    w.writerows(pairs)

print(f"Pares spot TRADING: {len(pairs)}")
print(f"Bases únicas: {len(bases)}")
print(f"Quotes únicas: {len(quotes)}")
