import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Run from the folder containing main_df.csv.
df = pd.read_csv('main_df.csv')
df = df.dropna(subset=['actual_delivery_days', 'order_approved_at'])
df = df[df['actual_delivery_days'] >= 0].copy()

# Extract only month/day/weekday: no year and no product category.
dates = pd.to_datetime(df['order_approved_at'])
df['month'] = dates.dt.month
df['day'] = dates.dt.day
df['weekday'] = dates.dt.dayofweek
df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
for col in ['customer_zip', 'seller_zip']:
    df[col + '_region'] = df[col].astype('string').str.zfill(5).str[:2].fillna('missing')

numeric = ['order_total_fv', 'N_items', 'distance',
           'month', 'day', 'weekday', 'is_weekend']
regions = ['customer_zip_region', 'seller_zip_region']
features = numeric + regions
# Keep only rows with all necessary features.
df = df.dropna(subset=numeric)
X, y = df[features], df['actual_delivery_days']

# Later approved orders are held out for a more realistic evaluation.
indices = df.sort_values('order_approved_at').index
cut = int(len(indices) * 0.8)
train_ids, test_ids = indices[:cut], indices[cut:]

preprocess = ColumnTransformer([
    ('numbers', 'passthrough', numeric),
    ('zip_regions', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), regions)
])
model = HistGradientBoostingRegressor(
    loss='absolute_error', learning_rate=0.05, max_iter=300,
    max_leaf_nodes=31, min_samples_leaf=30,
    categorical_features=[7, 8], random_state=42
)
pipe = Pipeline([('preprocess', preprocess), ('model', model)])
pipe.fit(X.loc[train_ids], y.loc[train_ids])
predictions = pipe.predict(X.loc[test_ids])
print('MAE (days):', round(mean_absolute_error(y.loc[test_ids], predictions), 2))
print('R2:', round(r2_score(y.loc[test_ids], predictions), 3))
joblib.dump(pipe, 'olist_delivery_model.joblib')
print('Model saved!')
