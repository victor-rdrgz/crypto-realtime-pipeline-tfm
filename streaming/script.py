#!/usr/bin/env python3
"""
Requisitos:
  pip install websockets ujson mlflow pandas python-dotenv
"""

import os, time, math, asyncio, collections
from datetime import datetime, timezone
try:
    import ujson as json
except Exception:
    import json
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed
import mlflow, mlflow.pyfunc
from dotenv import load_dotenv

# -----------------------------
# Carga .env y config
# -----------------------------
load_dotenv()
BINANCE_WS_BASE = os.getenv("BINANCE_WS_BASE", "wss://fstream.binance.com")

# Acepta símbolos con espacios y mayúsculas en .env; internamente WS usa minúsculas
SYMBOLS_RAW = os.getenv("SYMBOLS", "")
SYMBOLS = [s.strip().lower() for s in SYMBOLS_RAW.split(",") if s.strip()]

# MIN_NOTIONAL viene como JSON (claves pueden venir en MAYÚS); lo hacemos case-insensitive
MIN_NOTIONAL_RAW = os.getenv("MIN_NOTIONAL", "{}")
try:
    _tmp = json.loads(MIN_NOTIONAL_RAW)
    MIN_NOTIONAL = {str(k).lower(): float(v) for k, v in _tmp.items()}
except Exception as e:
    print(f"[env] MIN_NOTIONAL parse error: {e}; usando mapa vacío")
    MIN_NOTIONAL = {}

THRESH = float(os.getenv("DECISION_THRESHOLD", "0.5"))

def get_min_notional(sym_lower: str) -> float:
    return MIN_NOTIONAL.get(sym_lower, 0.0)

# -----------------------------
# Utilidades de ventanas
# -----------------------------
class TimedSum:
    """Suma deslizante sobre ventana (ms)."""
    def __init__(self, horizon_ms: int):
        self.h = horizon_ms
        self.q = collections.deque()  # (ts_ms, value)
        self.sum = 0.0
    def add(self, ts_ms: int, value: float):
        self.q.append((ts_ms, value))
        self.sum += value
        self._evict(ts_ms)
    def value(self, now_ms: int) -> float:
        self._evict(now_ms)
        return self.sum
    def _evict(self, now_ms: int):
        h = self.h
        q = self.q
        s = self.sum
        while q and (now_ms - q[0][0] > h):
            _, v = q.popleft()
            s -= v
        self.sum = s

class ReturnTracker:
    """Guarda mid por segundo para returns 1s y 5s."""
    def __init__(self):
        self.by_sec = {}
        self.secs = collections.deque()
    def update(self, ts_ms: int, mid: float):
        sec = ts_ms // 1000
        if self.secs and self.secs[-1] == sec:
            self.by_sec[sec] = mid
            return
        self.secs.append(sec)
        self.by_sec[sec] = mid
        cutoff = sec - 120
        while self.secs and self.secs[0] < cutoff:
            old = self.secs.popleft()
            self.by_sec.pop(old, None)
    def ret(self, ts_ms: int, horizon_s: int) -> float:
        sec = ts_ms // 1000
        p_now = self.by_sec.get(sec)
        p_past = self.by_sec.get(sec - horizon_s)
        if p_now is None or p_past is None or p_past == 0:
            return 0.0
        return (p_now / p_past) - 1.0

# -----------------------------
# Estado por símbolo y features (modelo v5 pyfunc: 11 columnas)
# -----------------------------
class SymbolState:
    def __init__(self, symbol_lower: str):
        self.sym_upper = symbol_lower.upper()

        # Top-of-book (bookTicker)
        self.bid_p = None; self.bid_q = None
        self.ask_p = None; self.ask_q = None

        # Rolling sums para trades (5s)
        self.buy_5s  = TimedSum(5000)
        self.sell_5s = TimedSum(5000)

        # Returns con mid
        self.rt = ReturnTracker()

    def on_book_ticker(self, ts_ms: int, bid_p: float, bid_q: float, ask_p: float, ask_q: float):
        self.bid_p, self.bid_q = bid_p, bid_q
        self.ask_p, self.ask_q = ask_p, ask_q
        if (bid_p is not None) and (ask_p is not None):
            mid = (bid_p + ask_p) / 2.0
            if mid and mid > 0:
                self.rt.update(ts_ms, mid)

    def on_agg_trade(self, ts_ms: int, price: float, qty: float, is_buyer_maker: bool):
        # is_buyer_maker=True → el vendedor es el agresor → venta agresora
        if is_buyer_maker:
            self.sell_5s.add(ts_ms, qty)
        else:
            self.buy_5s.add(ts_ms, qty)

    def build_features(self, ts_ms: int) -> dict:
        buy_qty_5s  = float(self.buy_5s.value(ts_ms))
        sell_qty_5s = float(self.sell_5s.value(ts_ms))
        z_r_1s = float(self.rt.ret(ts_ms, 1))
        z_r_5s = float(self.rt.ret(ts_ms, 5))

        # Imbalance en top-of-book actual (qty_imb_5s)
        if (self.bid_q is not None) and (self.ask_q is not None):
            qty_imb_5s = (self.bid_q - self.ask_q) / (self.bid_q + self.ask_q + 1e-9)
        else:
            qty_imb_5s = 0.0

        max_qty_5s = max(buy_qty_5s, sell_qty_5s)
        log_max_qty_5s = math.log1p(max_qty_5s)
        whale_side = 1 if buy_qty_5s > sell_qty_5s else 0

        # Interacciones
        x1 = whale_side * z_r_1s
        x2 = log_max_qty_5s * qty_imb_5s
        x3 = whale_side * qty_imb_5s
        x4 = z_r_5s * whale_side
        x5 = log_max_qty_5s * z_r_5s

        return {
            "buy_qty_5s": buy_qty_5s,
            "sell_qty_5s": sell_qty_5s,
            "z_r_1s": z_r_1s,
            "z_r_5s": z_r_5s,
            "qty_imb_5s": float(qty_imb_5s),
            "log_max_qty_5s": float(log_max_qty_5s),
            "whale_side": int(whale_side),
            "x1": float(x1), "x2": float(x2), "x3": float(x3), "x4": float(x4), "x5": float(x5)
        }

# -----------------------------
# MLflow: carga modelo (Databricks UC)
# -----------------------------
def _normalize_model_uri(uri: str) -> str:
    if not uri:
        raise RuntimeError("MODEL_URI vacío en .env")
    # Acepta 'models:/...', 'runs:/...', 'file:/...'
    if uri.startswith(("models:/", "runs:/", "file:/")):
        return uri
    # Si te pasaron solo el nombre del modelo, apúntalo a Production por defecto
    return f"models:/{uri}/Production"

def load_model():
    # Fuerza destinos Databricks/UC
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    print("[mlflow] tracking_uri =", mlflow.get_tracking_uri())
    print("[mlflow] registry_uri =", mlflow.get_registry_uri())

    uri = _normalize_model_uri(os.getenv("MODEL_URI", ""))
    t0 = time.time()
    m = mlflow.pyfunc.load_model(uri)
    print(f"[model] loaded {uri} in {1000*(time.time()-t0):.1f} ms")
    try:
        md = m.metadata
        print(f"[model] run_id={getattr(md,'run_id',None)} model_uuid={getattr(md,'model_uuid',None)}")
    except Exception:
        pass
    return m

MODEL = load_model()

def predict_one(feats: dict):
    df = pd.DataFrame([feats]).astype({"whale_side": "int32"})
    y = MODEL.predict(df)
    pred, p1 = None, None
    try:
        if hasattr(y, "to_dict"):
            yd = y.to_dict(orient="list")
            if "prediction" in yd and yd["prediction"]:
                pred = float(yd["prediction"][0])
            if "probability" in yd and yd["probability"]:
                v = yd["probability"][0]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    p1 = float(v[1])
        else:
            val = y[0] if hasattr(y, "__len__") else y
            pred = float(val)
    except Exception:
        pass
    if pred is None and p1 is not None:
        pred = 1.0 if p1 >= THRESH else 0.0
    if p1 is None and pred is not None:
        p1 = float(pred)
    return pred, p1

# -----------------------------
# WS helpers
# -----------------------------
def build_url(symbols_lower):
    parts = []
    for s in symbols_lower:
        parts.append(f"{s}@aggTrade")
        parts.append(f"{s}@bookTicker")
    return f"{BINANCE_WS_BASE}/stream?streams={'/'.join(parts)}"

STATES = {s.lower(): SymbolState(s.lower()) for s in SYMBOLS}
URL = build_url(SYMBOLS)

# -----------------------------
# Loop principal
# -----------------------------
async def run_ws():
    print("[boot] symbols =", SYMBOLS)
    print("[boot] min_notional =", MIN_NOTIONAL)
    backoff = 1.0
    while True:
        try:
            print(f"[ws] connecting {URL}")
            async with websockets.connect(URL, max_queue=4096, ping_interval=180, ping_timeout=600) as ws:
                print("[ws] connected")
                backoff = 1.0
                async for raw in ws:
                    try:
                        received_at = int(time.time() * 1000)
                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        data = msg.get("data") or {}
                        if not stream:
                            continue

                        sym_lower = stream.split("@", 1)[0]
                        st = STATES.get(sym_lower)
                        if st is None:
                            continue

                        # bookTicker: b/B (bid price/qty), a/A (ask price/qty)
                        if "bookTicker" in stream:
                            b = data.get("b"); B = data.get("B")
                            a = data.get("a"); A = data.get("A")
                            ts = int(data.get("E") or time.time()*1000)
                            try:
                                st.on_book_ticker(
                                    ts_ms=ts,
                                    bid_p=float(b) if b is not None else None,
                                    bid_q=float(B) if B is not None else None,
                                    ask_p=float(a) if a is not None else None,
                                    ask_q=float(A) if A is not None else None,
                                )
                            except Exception as e:
                                print(f"[bookTicker] parse_error {st.sym_upper}: {e}")

                        # aggTrade: p (price), q (qty), m (isBuyerMaker), T (ts)
                        elif "aggTrade" in stream:
                            ts = int(data.get("T") or data.get("E") or time.time()*1000)
                            p = float(data.get("p"))
                            q = float(data.get("q"))
                            mflag = bool(data.get("m"))
                            notional = p * q

                            # Actualiza rolling siempre (mantiene features)
                            st.on_agg_trade(ts_ms=ts, price=p, qty=q, is_buyer_maker=mflag)

                            # Gate por MIN_NOTIONAL[símbolo]
                            min_thr = get_min_notional(sym_lower)
                            if notional < min_thr:
                                continue
                                print(f'''under_notional: 
                                            - {st.sym_upper}
                                            - {notional}'''
                                )
                                continue

                            feats = st.build_features(ts_ms=ts)
                            pred, p1 = predict_one(feats)

                            processed_at = int(time.time() * 1000)
                            out = {
                                "ts_msg": ts,
                                "ts_msg_str": datetime.fromtimestamp(
                                    ts / 1000, tz=timezone.utc
                                ).strftime('%Y-%m-%d %H:%M:%S.%f'),
                                "processed_at": processed_at,
                                "processed_at_str": datetime\
                                    .fromtimestamp(
                                        processed_at / 1000, tz=timezone.utc
                                    )\
                                    .strftime('%Y-%m-%d %H:%M:%S.%f'),
                                "symbol": st.sym_upper,
                                "notional": notional,
                                "prediction": pred,
                            }
                            out['end_to_end_time_ms'] = (
                                out['processed_at'] - out['ts_msg']
                            )
                            out["network_latency_ms"] = (
                                received_at - out["ts_msg"]
                            )
                            out["processing_latency_ms"] = (
                                out["processed_at"] - received_at
                            )
                            print(json.dumps(
                                out, 
                                ensure_ascii=False, 
                                indent=2, 
                                sort_keys=True), 
                                  flush=True
                            )

                    except Exception as e:
                        print(f"[ws] msg_error: {e}", flush=True)

        except ConnectionClosed as e:
            print(f"[ws] closed: {e}", flush=True)
        except Exception as e:
            print(f"[ws] error: {e}", flush=True)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, 60.0)
        print(f"[ws] reconnecting in {backoff:.1f}s", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(run_ws())
    except KeyboardInterrupt:
        print("[ws] stopped", flush=True)
