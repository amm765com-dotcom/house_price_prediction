"""
House Price API.
Install: pip install fastapi uvicorn scikit-learn pandas numpy pydantic joblib
Run:     uvicorn app:app --reload --port 8000
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DB_PATH = Path(__file__).with_name('house_prices.db')
MIN_REAL_ROWS = 50
CAT = ['location', 'property_type']
NUM = ['area', 'bedrooms', 'bathrooms', 'number_floors', 'year_built']
LOC_RATE = {'Dha': 20000, 'Gulberg': 18000, 'Model Town': 16000,
            'Bahria Town': 14000, 'Johar Town': 13000, 'Other': 10000}
TYPE_MULT = {'House': 1.0, 'Apartment': 0.85, 'Villa': 1.3}


def load_real_rows():
  if not DB_PATH.exists():
    return pd.DataFrame(columns=CAT + NUM + ['price'])
  conn = sqlite3.connect(DB_PATH)
  try:
    df = pd.read_sql('SELECT * FROM properties', conn)
  except Exception as e:  # missing table / bad DB -> fall back to placeholder data
    print(f'WARNING: could not read table "properties" ({e}). Run 01_EXTRACT.py first.')
    return pd.DataFrame(columns=CAT + NUM + ['price'])
  finally:
    conn.close()
  missing = [c for c in CAT + NUM + ['price'] if c not in df.columns]
  if missing:
    print(f'WARNING: table is missing columns {missing}. Re-run 01_EXTRACT.py.')
    return pd.DataFrame(columns=CAT + NUM + ['price'])
  df = df[CAT + NUM + ['price']].copy()
  for c in NUM + ['price']:
    df[c] = pd.to_numeric(df[c], errors='coerce')  # text/blank -> NaN
  df = df.dropna()
  df = df[(df['area'] > 0) & (df['price'] > 0)]
  df['location'] = df['location'].astype(str).str.strip().str.title()
  df['property_type'] = df['property_type'].astype(str).str.strip().str.title()
  return df


def synthetic_rows(n=3000, seed=42):
  rng = np.random.default_rng(seed)
  loc = rng.choice(list(LOC_RATE), n)
  typ = rng.choice(list(TYPE_MULT), n, p=[0.6, 0.25, 0.15])
  area = rng.uniform(300, 8000, n)
  bed, bath = rng.integers(1, 8, n), rng.integers(1, 7, n)
  floors, year = rng.integers(1, 4, n), rng.integers(1985, 2027, n)
  rate = np.array([LOC_RATE[l] for l in loc])
  mult = np.array([TYPE_MULT[t] for t in typ])
  age_f = np.maximum(0.8, 1 - 0.006 * (2026 - year))
  price = (area * rate * mult * age_f * (1 + 0.04 * (floors - 1))
           + bed * 250000 + bath * 150000) * rng.normal(1.0, 0.05, n)
  return pd.DataFrame({'location': loc, 'property_type': typ, 'area': area,
                       'bedrooms': bed, 'bathrooms': bath,
                       'number_floors': floors, 'year_built': year,
                       'price': price})


def make_model(**kw):
  return Pipeline([
      ('prep', ColumnTransformer(
          [('cat', OneHotEncoder(handle_unknown='ignore'), CAT)],
          remainder='passthrough')),
      ('gbr', GradientBoostingRegressor(random_state=0, **kw)),
  ])


def train():
  real = load_real_rows()
  if len(real) >= MIN_REAL_ROWS:
    data, source = real, 'database'
  else:
    data = pd.concat([synthetic_rows(), real], ignore_index=True)
    source = 'database + placeholder data'
  X, y = data[CAT + NUM], data['price']
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
  mid = make_model().fit(Xtr, ytr)
  pred = mid.predict(Xte)
  metrics = {
      'source': source, 'real_rows': len(real), 'train_rows': len(Xtr),
      'test_rows': len(Xte),
      'mae': round(float(mean_absolute_error(yte, pred))),
      'r2': round(float(r2_score(yte, pred)), 3),
      'mape': round(float(np.mean(np.abs((yte - pred) / yte)) * 100), 1),
  }
  # Real prediction range: 10th and 90th percentile models
  lo = make_model(loss='quantile', alpha=0.1).fit(X, y)
  hi = make_model(loss='quantile', alpha=0.9).fit(X, y)
  mid_full = make_model().fit(X, y)
  metrics['importance'] = {}
  try:  # optional extra: skipped if this scikit-learn version cannot do it
    names = mid_full.named_steps['prep'].get_feature_names_out()
    imp = mid_full.named_steps['gbr'].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
      key = n.split('__')[-1]
      for p in ('location', 'property_type'):
        if key.startswith(p + '_'):
          key = p
      agg[key] = agg.get(key, 0) + float(v)
    metrics['importance'] = dict(sorted(
        ((k, round(v, 3)) for k, v in agg.items()), key=lambda t: -t[1]))
  except Exception as e:
    print(f'NOTE: feature importance skipped ({e})')
  return mid_full, lo, hi, metrics


MODEL, LO, HI, METRICS = train()

app = FastAPI(title='House Price API')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])


class HouseIn(BaseModel):
  area: float = Field(..., gt=0, description='Area in sq ft')
  bedrooms: int = Field(..., ge=0)
  bathrooms: float = Field(..., ge=0)
  location: str = Field(..., min_length=1)
  property_type: str = 'House'
  number_floors: int = Field(1, ge=1)
  year_built: int = Field(2000, ge=1800)


@app.get('/health')
def health():
  return {'status': 'ok', 'trained_on': METRICS['source'],
          'real_rows': METRICS['real_rows']}


@app.get('/metrics')
def metrics():
  return METRICS


@app.get('/meta')
def meta():
  return {'locations': list(LOC_RATE), 'types': list(TYPE_MULT),
          'avg_rate': LOC_RATE}


@app.post('/predict')
def predict(h: HouseIn):
  row = pd.DataFrame([{
      'location': h.location.strip().title(),
      'property_type': h.property_type.strip().title(),
      'area': h.area, 'bedrooms': h.bedrooms, 'bathrooms': h.bathrooms,
      'number_floors': h.number_floors, 'year_built': h.year_built}])[CAT + NUM]
  try:
    p = float(MODEL.predict(row)[0])
    lo, hi = float(LO.predict(row)[0]), float(HI.predict(row)[0])
  except Exception as e:
    raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
  lo, hi = min(lo, p), max(hi, p)
  return {'price': round(p), 'low': round(lo), 'high': round(hi),
          'trained_on': METRICS['source'], 'real_rows': METRICS['real_rows']}


if __name__ == '__main__':
  import uvicorn
  print('Starting server on http://127.0.0.1:8000  (training takes a few seconds)')
  uvicorn.run(app, host='127.0.0.1', port=8000)
