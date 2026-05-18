import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import nltk
from nltk.tokenize import sent_tokenize
from tabulate import tabulate
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('punkt_tab')

#Load csv
Data = pd.read_csv("train_data.csv")

# delete duplicate
Data = Data.drop_duplicates()

for col in Data.select_dtypes(include='object').columns:
    Data[col] = Data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# to solve problems for empty cells
Data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
Data.replace(['?', 'None', 'n/a'], np.nan, inplace=True)

cols_to_drop = [
    'DRMNotice', 'ExtUserAcctNotice', 'LinuxRecReqsText',
    'MacRecReqsText', 'ResponseID', 'QueryID','HeaderImage','LegalNotice'
]
Data.drop(columns=cols_to_drop, inplace=True, errors='ignore') # drop feature not useful

# Handeling date
Data['ReleaseDate'] = pd.to_datetime(
    Data['ReleaseDate'],
    format='%b %d %Y',
    errors='coerce'
)

Data.drop(columns=['PriceCurrency'], inplace=True, errors='ignore')# conestant coulmn

#handling URL cols
Data['HasSupportURL'] = (Data['SupportURL'].notnull() & (Data['SupportURL'] != 'Unknown')).astype(int)
Data['HasBackground'] = (Data['Background'].notnull() & (Data['Background'] != 'Unknown')).astype(int)
Data['HasSupportEmail'] = (Data['SupportEmail'].notnull() & (Data['SupportEmail'] != 'Unknown')).astype(int)
Data['HasWebsite'] = (Data['Website'].notnull() & (Data['Website'] != 'Unknown')).astype(int)

Data.drop(columns=['SupportURL', 'Background', 'SupportEmail', 'Website'], inplace=True)

#Handing landuages
Data['SupportedLanguages'] = Data['SupportedLanguages'].str.replace('*', '', regex=False)
valid_langs = [
    "English", "French", "German", "Italian", "Spanish",
    "Simplified Chinese", "Traditional Chinese",
    "Korean", "Russian", "Dutch", "Danish", "Finnish",
    "Japanese", "Norwegian", "Polish", "Portuguese",
    "Swedish", "Thai"
]

for lang in valid_langs:
    Data[f'Lang_{lang}'] = Data['SupportedLanguages'].fillna('').apply(
        lambda x: 1 if lang in x else 0
    )

Data['Num_Supported_Languages'] = Data['SupportedLanguages'].fillna('').apply(
    lambda x: len(x.split())
)

Data.drop(columns=['SupportedLanguages'], inplace=True)

# ==============
# VADER
# ==============

analyzer = SentimentIntensityAnalyzer()

def vader_features(text):
    if not isinstance(text, str):
        return 0, 0, 0

    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return 0, 0, 0

    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences]

    return (
        sum(scores) / len(scores),
        max(scores),
        min(scores)
    )

Data[['Sent_avg', 'Sent_max', 'Sent_min']] = Data['Reviews'].fillna('').apply(
    lambda x: pd.Series(vader_features(x))
)

Data['Reviews_Length'] = Data['Reviews'].fillna('').apply(len)
Data['Sent_x_Length'] = Data['Sent_avg'] * Data['Reviews_Length']
Data.drop(columns=['Reviews'], inplace=True)

print("VADER numeric features ready")

# ====================================================
#numirical data
numeric_cols = Data.select_dtypes(include=[np.number]).columns
#categorical data
object_cols = Data.select_dtypes(include=['object']).columns
Data[object_cols] = Data[object_cols].fillna("Unknown")

Data.loc[Data['PriceFinal'] == 0, 'IsFree'] = True
Data.loc[Data['IsFree'] == True, ['PriceInitial', 'PriceFinal']] = 0.0
mask = Data['PriceInitial'] < Data['PriceFinal']
Data.loc[mask, 'PriceInitial'] = Data.loc[mask, 'PriceFinal'] # logical features

#log transformation
for col in ['SteamSpyOwners', 'AchievementCount', 'PriceFinal']:
    Data[f'{col}_Log'] = np.log1p(Data[col])

#Define features for Random Forest
cols_to_exclude = ['AchievementCount', 'PriceFinal']

# Calculate Game Age based on the release year relative to 2026
Data['Release_Year'] = Data['ReleaseDate'].dt.year
Data['Game_Age'] = 2026 - Data['Release_Year']

#calc features based on gameAge
Data['Metacritic_Per_Year'] = Data['Metacritic'] / (Data['Game_Age'] + 1)
Data['Achievement_Density'] = Data['AchievementCount'] / (Data['Game_Age'] + 1)
Data['Lang_Reach_Score'] = Data['Num_Supported_Languages'] * np.log1p(Data['Game_Age'] + 1)
Data['Global_Quality'] = Data['Metacritic'] * Data['Num_Supported_Languages']

# Categorical flags based on Game Age
Data['Is_Retro'] = (Data['Game_Age'] > 10).astype(int)
Data['Is_New_Trend'] = (Data['Game_Age'] < 2).astype(int)

# Sentiment and Quality interaction score
Data['Positive_Quality_Score'] = Data['Sent_avg'] * Data['Metacritic']

#Assessing quality relative to price
Data['Value_Score'] = Data['Metacritic'] / (Data['PriceFinal_Log'] + 1)

#Combining language support with achievements as a proxy for developer effort
Data['Engagement_Score'] = Data['Num_Supported_Languages'] * Data['AchievementHighlightedCount']
#=====================================================

y = Data['RecommendationCount']
X = Data.drop(columns=['RecommendationCount'])

mask = y.notna()
X = X[mask]
y = y[mask]
#=====================================================

# split Data train , validation , test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
leakage_cols = [
    'SteamSpyOwners',
    'SteamSpyOwners_Log',
    'SteamSpyOwnersVariance',
    'Owners_x_Metacritic',
    'Price_Success_Index',
    'Recs_Per_Year'
]
for df in [X_train, X_val, X_test]:
    df.drop(columns=leakage_cols, inplace=True, errors='ignore')
train_medians = X_train.select_dtypes(include=[np.number]).median()
X_train = X_train.fillna(train_medians)
X_val   = X_val.fillna(train_medians)
X_test  = X_test.fillna(train_medians)
print("Median fill done")

# Extract date features
date_mode = X_train['ReleaseDate'].mode()[0]
X_train['ReleaseDate'] = X_train['ReleaseDate'].fillna(date_mode)
X_val['ReleaseDate']   = X_val['ReleaseDate'].fillna(date_mode)
X_test['ReleaseDate']  = X_test['ReleaseDate'].fillna(date_mode)

for df in [X_train, X_val, X_test]:
    df['Release_Year']      = df['ReleaseDate'].dt.year
    df['Release_Month']     = df['ReleaseDate'].dt.month
    df['Release_Day']       = df['ReleaseDate'].dt.day
    df['Release_DayOfWeek'] = df['ReleaseDate'].dt.dayofweek

# drop original cols
X_train.drop(columns=['ReleaseDate'], inplace=True)
X_val.drop(columns=['ReleaseDate'], inplace=True)
X_test.drop(columns=['ReleaseDate'], inplace=True)
print("ReleaseDate preprocessing done")

# Handling requirements text features
req_cols = ['PCMinReqsText', 'PCRecReqsText', 'MacMinReqsText','LinuxMinReqsText']
for col in req_cols:
    for df in [X_train, X_val, X_test]:
        df[f'{col}_HasGPU']     = df[col].str.contains('GPU|graphics|GeForce|Radeon', case=False, na=False).astype(int)
        df[f'{col}_HasRAM']     = df[col].str.contains('RAM|memory|GB', case=False, na=False).astype(int)
        df[f'{col}_HasDirectX'] = df[col].str.contains('DirectX', case=False, na=False).astype(int)
        df[f'{col}_Length']     = df[col].fillna('').str.len()

for df in [X_train, X_val, X_test]:
    df.drop(columns=req_cols, inplace=True)
print(" Requirements columns done")

# Handling string features
text_cols = ['AboutText', 'ShortDescrip', 'DetailedDescrip']
for col in text_cols:
    for df in [X_train, X_val, X_test]:
        df[f'{col}_Length'] = df[col].fillna('').str.len()

for df in [X_train, X_val, X_test]:
    df['HasAboutText'] = df['AboutText'].fillna('').ne('').astype(int)

for df in [X_train, X_val, X_test]:
    df.drop(columns=text_cols, inplace=True)
print(" Text columns processed")

# make log to target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)


# make log to target
skewed_features = [
    'SteamSpyPlayersEstimate', 'SteamSpyOwners',
    'DLCCount', 'AchievementCount', 'PriceFinal', 'PriceInitial'
]

# for linear and poly
X_train_linear = X_train.copy()
X_val_linear = X_val.copy()
X_test_linear = X_test.copy()

for col in skewed_features:
    if col in X_train.columns:
        X_train_linear[f'{col}_Log'] = np.log1p(X_train[col])
        X_val_linear[f'{col}_Log'] = np.log1p(X_val[col])
        X_test_linear[f'{col}_Log'] = np.log1p(X_test[col])


#Preprocessing

bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

# apply on all spilts bool encode
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_val[bool_cols]   = X_val[bool_cols].astype(int)
X_test[bool_cols]  = X_test[bool_cols].astype(int)

#RobustScaler
features_to_scale = [
    'SteamSpyPlayersEstimate','SteamSpyPlayersVariance',
    'AchievementCount','AchievementHighlightedCount',
    'ScreenshotCount','AboutText_Length',
    'ShortDescrip_Length','DetailedDescrip_Length',
    'MacMinReqsText_Length','PCRecReqsText_Length',
    'PCMinReqsText_Length' , 'LinuxMinReqsText_Length'
]

scaler = RobustScaler()
X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_val[features_to_scale] = scaler.transform(X_val[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])




#StanderScaler
standard_features = ['Sent_x_Length',
                     'Reviews_Length',
                     'Metacritic',
                     'LinuxMinReqsText_Length'
                     ]
feature_scaler = StandardScaler()
X_train[standard_features] = feature_scaler.fit_transform(X_train[standard_features])
X_val[standard_features]   = feature_scaler.transform(X_val[standard_features])
X_test[standard_features]  = feature_scaler.transform(X_test[standard_features])
joblib.dump(feature_scaler, "feature_scaler.pkl")

#==========================================================
# Feature Selection
#==========================================================
#  Correlation Feature Selection (x_train data)
corr_matrix = X_train.select_dtypes(include=[np.number]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

print("\n--- Feature Selection: Correlation Result ---")
print(f"Number of columns recommended for removal due to redundancy:", len(to_drop))

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=False, cmap='YlGnBu')
plt.title('Correlation Heatmap After Preprocessing')
plt.show()

# dropping features
to_drop_corr = [
    'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance',
    'PurchaseAvail', 'Sent_max', 'PriceInitial', 'HasBackground',
    'PCRecReqsText_HasRAM', 'MacMinReqsText_HasRAM',
    'LinuxMinReqsText_HasRAM', 'DetailedDescrip_Length'
]
# Protect important featuers to be dropped
protected_features = ['Price_Success_Index', 'Metacritic_Per_Year']
#raise the threshold to 85
to_drop = [column for column in upper.columns if any(upper[column] > 0.85) and column not in protected_features]

X_train.drop(columns=to_drop, inplace=True, errors='ignore')
X_val.drop(columns=to_drop, inplace=True, errors='ignore')
X_test.drop(columns=to_drop, inplace=True, errors='ignore')

print(f"Data cleaned. Protected features kept. Remaining: {X_train.shape[1]}")

# ============================================
#  Correlation Feature Selection(linear path) =
# ============================================

corr_matrix_lin = X_train_linear.select_dtypes(include=[np.number]).corr().abs()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix_lin, annot=False, cmap='OrRd')
plt.title('Correlation Heatmap - Linear Path (Log Transformed)')
plt.show()

custom_drops = []
if 'PriceInitial' in corr_matrix_lin.columns and 'PriceFinal' in corr_matrix_lin.columns:
    if corr_matrix_lin.loc['PriceInitial', 'PriceFinal'] > 0.90:
        custom_drops.append('PriceInitial')
        print("Manual Check: PriceInitial will be dropped to keep PriceFinal.")

upper_lin = corr_matrix_lin.where(np.triu(np.ones(corr_matrix_lin.shape), k=1).astype(bool))
#important features for Liner model
protected_linear = ['Metacritic_Per_Year', 'Price_Success_Index']

auto_drops = [column for column in upper_lin.columns
              if any(upper_lin[column] > 0.90)
              and column not in protected_linear]

to_drop_corr_lin = list(set(custom_drops + auto_drops))
if 'PriceFinal' in to_drop_corr_lin and 'PriceInitial' in to_drop_corr_lin:
    to_drop_corr_lin.remove('PriceFinal')

X_train_linear.drop(columns=to_drop_corr_lin, inplace=True, errors='ignore')
X_val_linear.drop(columns=to_drop_corr_lin, inplace=True, errors='ignore')
X_test_linear.drop(columns=to_drop_corr_lin, inplace=True, errors='ignore')

print(f"Data cleaned successfully. Number of remaining columns: {X_train_linear.shape[1]}")

# ===============================
#  SelectKBest Feature Selection
# ===============================

X_train_lin_num = X_train_linear.select_dtypes(include=[np.number])
X_val_lin_num   = X_val_linear.select_dtypes(include=[np.number])
X_test_lin_num  = X_test_linear.select_dtypes(include=[np.number])


k_features = 40
selector_lin = SelectKBest(score_func=f_regression, k=k_features)

X_train_k_lin = selector_lin.fit_transform(X_train_lin_num, y_train_log)
X_val_k_lin  = selector_lin.transform(X_val_lin_num)
X_test_k_lin = selector_lin.transform(X_test_lin_num)



selected_linear_features = X_train_lin_num.columns[selector_lin.get_support()].tolist()

print(f"--- Linear Path Selection Complete ---")
print(f"Selected {len(selected_linear_features)} features for Linear Models.")


X_train_final_linear = pd.DataFrame(X_train_k_lin, columns=selected_linear_features)
X_val_final_linear   = pd.DataFrame(X_val_k_lin, columns=selected_linear_features)
X_test_final_linear  = pd.DataFrame(X_test_k_lin, columns=selected_linear_features)

# ====================================================================
#  Testing for the Optimal K (Hyperparameter Tuning for SelectKBest)
# ====================================================================

k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
rmse_results = []

for k in k_values:
    selector = SelectKBest(f_regression, k=min(k, X_train_lin_num.shape[1]))
    X_k = selector.fit_transform(X_train_lin_num, y_train_log)



    model = LinearRegression()

    scores = cross_val_score(
        model,
        X_k,
        y_train_log,
        scoring='neg_root_mean_squared_error',
        cv=5
    )

    rmse = -scores.mean()
    rmse_results.append(rmse)




plt.figure(figsize=(10,6))
plt.plot(k_values, rmse_results, marker='o')
plt.xlabel("K")
plt.ylabel("CV RMSE")
plt.title("K vs RMSE")
plt.grid()
plt.show()

best_k = k_values[np.argmin(rmse_results)]
print(f"\nConclusion: The best K for your model is {best_k}")

print('<'*60)
print("Trainning Models started...")
##################################  Hossam  ##############################
# ==========================
# RANDOM FOREST MODEL
# ==========================

# 1. Prepare numeric splits
X_train_rf = X_train.select_dtypes(include=[np.number]).copy()
X_val_rf   = X_val.select_dtypes(include=[np.number]).copy()
X_test_rf  = X_test.select_dtypes(include=[np.number]).copy()

# Drop original features replaced by Log transforms
cols_to_drop_final = ['SteamSpyOwners', 'PriceFinal', 'AchievementCount', 'SteamSpyPlayersEstimate']
for df in [X_train_rf, X_val_rf, X_test_rf]:
    df.drop(columns=cols_to_drop_final, inplace=True, errors='ignore')

common_rf_cols = X_train_rf.columns.tolist()
X_val_rf  = X_val_rf.reindex(columns=common_rf_cols, fill_value=0)
X_test_rf = X_test_rf.reindex(columns=common_rf_cols, fill_value=0)

# 2. Build model
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=20,   # زود
    max_features='sqrt',
    random_state=42
)

# 3. Cross-validation
cv_scores = cross_val_score(
    rf_model, X_train_rf, y_train_log,
    scoring='neg_root_mean_squared_error', cv=5
)
# حفظ أسماء الأعمدة النهائية لضمان الترتيب
final_columns = X_train_rf.columns.tolist()
# 4. Train

rf_model.fit(X_train_rf, y_train_log)

# 5. Feature Importance
feat_importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train_rf.columns
).sort_values(ascending=False)

# 6. Evaluation & Accuracy
y_test_pred_log = rf_model.predict(X_test_rf)
test_r2 = r2_score(y_test_log, y_test_pred_log)
rmse_rf = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))

y_train_pred_rf = rf_model.predict(X_train_rf)
train_r2_rf = r2_score(y_train_log, y_train_pred_rf)
print(" RANDOM FOREST Finished")
# ==========================
# RIDGE REGRESSION MODEL  =
# ==========================

# only 20 feature to be fast
selector_ridge = SelectKBest(f_regression, k=20)
X_train_ridge_k = selector_ridge.fit_transform(X_train_lin_num, y_train_log)
X_test_ridge_k  = selector_ridge.transform(X_test_lin_num)

# Scale
ridge_scaler = StandardScaler()
X_train_ridge_scaled = ridge_scaler.fit_transform(X_train_ridge_k)
X_test_ridge_scaled  = ridge_scaler.transform(X_test_ridge_k)
joblib.dump(ridge_scaler, "ridge_scaler.pkl")

# Polynomial
poly         = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_ridge_scaled)
X_test_poly  = poly.transform(X_test_ridge_scaled)
joblib.dump(poly, "poly_ridge.pkl")

# Tuning alpha
alpha_values     = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
ridge_cv_results = []

for alpha in alpha_values:
    ridge_cv = Ridge(alpha=alpha)
    scores   = cross_val_score(
        ridge_cv, X_train_poly, y_train_log,
        scoring='neg_root_mean_squared_error', cv=5
    )
    rmse_cv = -scores.mean()
    ridge_cv_results.append(rmse_cv)

best_alpha = alpha_values[np.argmin(ridge_cv_results)]

# Train
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_poly, y_train_log)

# Evaluation
y_test_pred_ridge_log = ridge_model.predict(X_test_poly)
y_train_pred_ridge_log = ridge_model.predict(X_train_poly)
test_r2_ridge         = r2_score(y_test_log, y_test_pred_ridge_log)
train_r2_ridge = r2_score(y_train_log, y_train_pred_ridge_log)
rmse_ridge            = np.sqrt(mean_squared_error(y_test_log, y_test_pred_ridge_log))
# diff_ridge     = train_r2_ridge - test_r2_ridge

print("RIDGE REGRESSION Finished")
########################### MOHAMED ALAA #############################################
# ===========================
# LINEAR REGRESSION MODEL
# ===========================

# Choosing best k =>> from tuning
selector_final = SelectKBest(f_regression, k=min(best_k, X_train_lin_num.shape[1]))
X_train_lin_final = selector_final.fit_transform(X_train_lin_num, y_train_log)
X_test_lin_final  = selector_final.transform(X_test_lin_num)

# Scaling for Linear Regression
lin_scaler = StandardScaler()
X_train_lin_scaled = lin_scaler.fit_transform(X_train_lin_final)
X_test_lin_scaled  = lin_scaler.transform(X_test_lin_final)
joblib.dump(lin_scaler, "lin_scaler.pkl")

# ── DIAGNOSTIC 3: what lin_scaler learned ─────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC 3 — lin_scaler (what the scaler was fitted on)")
print("="*60)
print(f"  Features in: {X_train_lin_final.shape[1]}")
print(f"  Scaler mean_ (first 10): {lin_scaler.mean_[:10].round(4).tolist()}")
print(f"  Scaler scale_ (first 10): {lin_scaler.scale_[:10].round(4).tolist()}")
print(f"  Post-scale train mean (should be ~0): {X_train_lin_scaled.mean():.6f}")
print(f"  Post-scale train std  (should be ~1): {X_train_lin_scaled.std():.6f}")
print("  >>> If test script post-scale mean/std is far from 0/1, scaler is misapplied.")
print("="*60 + "\n")


# train model
lr_model = LinearRegression()
lr_model.fit(X_train_lin_scaled, y_train_log)

# Evaluation
y_test_pred_lin_log = lr_model.predict(X_test_lin_scaled)
y_train_pred_lin_log = lr_model.predict(X_train_lin_scaled)
test_r2_lin = r2_score(y_test_log, y_test_pred_lin_log)
train_r2_lin = r2_score(y_train_log, y_train_pred_lin_log)
rmse_lin = np.sqrt(mean_squared_error(y_test_log, y_test_pred_lin_log))
# diff_lin     = train_r2_lin - test_r2_lin

print("LINER REGRESSION Finished")

# =======================
# ELASTIC NET MODEL
# =======================
already_scaled = features_to_scale + standard_features
unscaled_cols = [c for c in X_train_lin_num.columns if c not in already_scaled]

preprocessor = ColumnTransformer(
    transformers=[
        ("scale_unscaled", StandardScaler(), unscaled_cols),
    ],
    remainder='passthrough'
)

model = Pipeline([
    ("prep", preprocessor),

    # keep important features only
    ("select", SelectKBest(f_regression, k=min(best_k, 20))),

    # interactions BUT controlled
    ("interact", PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )),

    # scale AFTER interactions
    ("scale_final", StandardScaler()),

    ("elastic", ElasticNetCV(
        l1_ratio=[0.3, 0.5, 0.7, 0.9],      # balanced
        alphas=np.logspace(-3, 1, 25),      # medium size
        cv=3,                               # faster but still stable
        max_iter=70000,
        tol=1e-3,
        n_jobs=-1,
        random_state=42
    ))
])

# Train
model.fit(X_train_lin_num, y_train_log)

# Evaluate
y_pred = model.predict(X_test_lin_num)
y_train_pred = model.predict(X_train_lin_num)
y_val_pred = model.predict(X_val_lin_num)

test_r2_en = r2_score(y_test_log, y_pred)
train_r2_en = r2_score(y_train_log, y_train_pred)
val_r2_en = r2_score(y_val_log, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_test_log, y_pred))
# diff_en   = train_r2_en - test_r2_en

print("Elastic Finished")

############################### SHROUK ###################
# ==============================
# GRADIENT BOOSTING REGRESSOR  =
# ==============================
# data preparation as RF
X_train_gb = X_train_rf.copy()
X_val_gb   = X_val_rf.copy()
X_test_gb  = X_test_rf.copy()


gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    max_features=.7,
    subsample=0.8,
    min_samples_leaf=50,
    warm_start=True,
    random_state=42
)
# training
gb_model.fit(X_train_gb, y_train_log)

# Evaluation
y_test_pred_gb = gb_model.predict(X_test_gb)
test_r2_gb = r2_score(y_test_log, y_test_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test_log, y_test_pred_gb))

y_train_pred_gb = gb_model.predict(X_train_gb)
train_r2_gb = r2_score(y_train_log, y_train_pred_gb)
# diff_gb  = train_r2_gb - test_r2_gb

print("Gradient Boosting Finished")
#################################### AHMED HOSSAM ################################
# ========================
# LASSO REGRESSION MODEL =
# ========================

# train mldel to choose automatic alpha
lasso_model = LassoCV(
    cv=5,
    random_state=42,
    max_iter=50000,
    alphas=np.logspace(-4, 2, 50)
)
#  Linear data after  scaling
lasso_model.fit(X_train_lin_scaled, y_train_log)

# predection
y_test_pred_lasso_log = lasso_model.predict(X_test_lin_scaled)
y_train_pred_lasso_log = lasso_model.predict(X_train_lin_scaled)

# evaluation
test_r2_lasso = r2_score(y_test_log, y_test_pred_lasso_log)
train_r2_lasso = r2_score(y_train_log, y_train_pred_lasso_log)
rmse_lasso     = np.sqrt(mean_squared_error(y_test_log, y_test_pred_lasso_log))
# diff_lasso     = train_r2_lasso - test_r2_lasso

# Feature Selection
non_zero = np.sum(lasso_model.coef_ != 0)
total = len(lasso_model.coef_)

print("LASSO REGRESSION Finished")
# =============================
# POLYNOMIAL REGRESSION MODEL =
# =============================

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_lin_scaled)
X_test_poly = poly.transform(X_test_lin_scaled)
joblib.dump(poly, "poly_main.pkl")
#model trainning
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_log)

#Predection
y_test_pred_poly_log = poly_model.predict(X_test_poly)
y_train_pred_poly_log = poly_model.predict(X_train_poly)

# Evaluation
test_r2_poly = r2_score(y_test_log, y_test_pred_poly_log)
train_r2_poly = r2_score(y_train_log, y_train_pred_poly_log)
rmse_poly = np.sqrt(mean_squared_error(y_test_log, y_test_pred_poly_log))
# diff_poly     = train_r2_poly - test_r2_poly

print("POLYNOMIAL REGRESSION Finished")
######################################### MENNA MOHAMED #############################
# ========================================
# SUPPORT VECTOR REGRESSION (SVR) MODEL  =
# ========================================

#preparing data
X_train_svr_raw = X_train.select_dtypes(include=[np.number]).copy()
X_test_svr_raw  = X_test.select_dtypes(include=[np.number]).copy()

# scaling for SVR By Stander_Scaler
svr_x_scaler = StandardScaler()
X_train_svr = svr_x_scaler.fit_transform(X_train_svr_raw)
X_test_svr  = svr_x_scaler.transform(X_test_svr_raw)
joblib.dump(svr_x_scaler, "svr_x_scaler.pkl")
# Select top 15 features to reduce overfitting
selector_svr         = SelectKBest(f_regression, k=15)
X_train_svr_selected = selector_svr.fit_transform(X_train_svr, y_train_log)
X_test_svr_selected  = selector_svr.transform(X_test_svr)

# y_scaled
svr_y_scaler = StandardScaler()
y_train_svr = svr_y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).flatten()
joblib.dump(svr_y_scaler, "svr_y_scaler.pkl")

# building the model
svr_model = SVR(
    kernel='rbf',
    C=5.0,
    epsilon=0.5,
    gamma='scale'
)

# training
svr_model.fit(X_train_svr_selected, y_train_svr)

# prediction & validation
y_test_pred_svr_scaled = svr_model.predict(X_test_svr_selected)
y_test_pred_svr = svr_y_scaler.inverse_transform(y_test_pred_svr_scaled.reshape(-1, 1)).flatten()

y_train_pred_svr_scaled = svr_model.predict(X_train_svr_selected)
y_train_pred_svr = svr_y_scaler.inverse_transform(y_train_pred_svr_scaled.reshape(-1,1)).flatten()
# calc scales
test_r2_svr  = r2_score(y_test_log,  y_test_pred_svr)
train_r2_svr = r2_score(y_train_log, y_train_pred_svr)
rmse_svr     = np.sqrt(mean_squared_error(y_test_log, y_test_pred_svr))
# diff_svr     = train_r2_svr - test_r2_svr
print("SVR Finished.")

# =======================
# DECISION TREE MODEL   =
# =======================

#  Independent data & Variables
X_train_dt = X_train.select_dtypes(include=[np.number]).copy()
X_val_dt   = X_val.select_dtypes(include=[np.number]).copy()
X_test_dt  = X_test.select_dtypes(include=[np.number]).copy()

#building model depending on constrains =>> to no Overfitting
dt_model = DecisionTreeRegressor(
    max_depth=7,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# training
dt_model.fit(X_train_dt, y_train_log)

# evaluation on val & test
y_val_pred_dt = dt_model.predict(X_val_dt)
y_test_pred_dt = dt_model.predict(X_test_dt)
y_train_pred_dt = dt_model.predict(X_train_dt)

# cala scales
rmse_dt = np.sqrt(mean_squared_error(y_test_log, y_test_pred_dt))
r2_dt = r2_score(y_test_log, y_test_pred_dt)
train_r2_dt   = r2_score(y_train_log, y_train_pred_dt)
# diff_dt       = train_r2_dt - r2_dt

# top ten features
dt_importances = pd.Series(
    dt_model.feature_importances_,
    index=X_train_dt.columns
).sort_values(ascending=False)
print("Decision Tree  Finished")
#============================
#XGBOOST REGRESSOR
#===============================
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=2,
    reg_alpha=2.0,
    reg_lambda=10,
    min_child_weight=10,
    random_state=42,
    n_jobs=-1
)
# حفظ أسماء الأعمدة النهائية لضمان الترتيب
final_columns = X_train_rf.columns.tolist()
xgb_model.fit(X_train_rf, y_train_log)
y_pred_xgb = xgb_model.predict(X_test_rf)
xgb_r2 = r2_score(y_test_log, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_xgb))

y_train_pred_xgb = xgb_model.predict(X_train_rf)
train_r2_xgb = r2_score(y_train_log, y_train_pred_xgb)
# diff_xgb     = train_r2_xgb - xgb_r2

# # plt.figure(figsize=(10, 6))
# xgb.plot_importance(xgb_model, max_num_features=15, importance_type='weight')
# plt.title("Top 15 Features in XGBoost")
# plt.show()

print("XGBOOST REGRESSOR Finished")

#==================================
#feedforward neural network model =
#==================================

# Use same data as RF/XGB
X_train_nn = X_train_rf.copy()
X_val_nn   = X_val_rf.copy()
X_test_nn  = X_test_rf.copy()

# Scale
nn_standard_scaler = StandardScaler()
X_train_nn = nn_standard_scaler.fit_transform(X_train_nn)
X_val_nn   = nn_standard_scaler.transform(X_val_nn)
X_test_nn  = nn_standard_scaler.transform(X_test_nn)
joblib.dump(nn_standard_scaler, "nn_scaler.pkl")

# Build model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.4),

    Dense(32, activation='relu'),

    Dense(1)
])

nn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Early stopping (VERY IMPORTANT)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train
history = nn_model.fit(
    X_train_nn, y_train_log,
    validation_data=(X_val_nn, y_val_log),
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred_nn = nn_model.predict(X_test_nn).flatten()
y_train_pred_nn = nn_model.predict(X_train_nn).flatten()
nn_r2 = r2_score(y_test_log, y_pred_nn)
train_r2_nn = r2_score(y_train_log, y_train_pred_nn)
nn_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_nn))
# diff_nn     = train_r2_nn - nn_r2

print("Neural Network Finished")
#==================
#VotingRegression =
#==================
ensemble_final = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('gbm', gb_model),
    ('rf', rf_model)
], weights=[2, 2, 1])

ensemble_final.fit(X_train_rf, y_train_log)
final_pred = ensemble_final.predict(X_test_rf)
final_train_pred = ensemble_final.predict(X_train_rf)

final_accuracy = r2_score(y_test_log, final_pred)
train_r2_ens     = r2_score(y_train_log, final_train_pred)
rmse_ens         = np.sqrt(mean_squared_error(y_test_log, final_pred))
# diff_ens         = train_r2_ens - final_accuracy


residuals = y_test_log - final_pred
plt.figure(figsize=(10, 5))
sns.scatterplot(x=final_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot: Are we missing something?')
plt.xlabel('Predicted Popularity')
plt.ylabel('Error (Actual - Predicted)')
plt.show()

# Top 5 features and their relation to the target
# Make sure these column names match your DataFrame (Data)
target_col = 'RecommendationCount'
top_features = ['Metacritic_Per_Year', 'Achievement_Density', 'Metacritic', 'Lang_Reach_Score']

# Correlation
correlation_check = Data[top_features + [target_col]].corr()

print("VotingRegression Finished")




print("\n=== TRAIN X_lin_num ===")
print(f"Shape: {X_train_lin_num.shape}")
print(f"Mean: {X_train_lin_num.mean().mean():.4f}")
print(f"Max value: {X_train_lin_num.max().max():.4f}")
print("\nTop 10 columns by max value:")
print(X_train_lin_num.max().sort_values(ascending=False).head(10))
print("\nTop 10 columns by mean value:")
print(X_train_lin_num.mean().sort_values(ascending=False).head(10))

# ==============================================================================================
models = [
    {"name": "Random Forest",         "y_pred": y_test_pred_log,        "train_r2": train_r2_rf},
    {"name": "Ridge Regression",       "y_pred": y_test_pred_ridge_log,  "train_r2": train_r2_ridge},
    {"name": "Linear Regression",      "y_pred": y_test_pred_lin_log,    "train_r2": train_r2_lin},
    {"name": "Elastic Net",            "y_pred": y_pred,                 "train_r2": train_r2_en},
    {"name": "Gradient Boosting",      "y_pred": y_test_pred_gb,         "train_r2": train_r2_gb},
    {"name": "Lasso Regression",       "y_pred": y_test_pred_lasso_log,  "train_r2": train_r2_lasso},
    {"name": "Polynomial Regression",  "y_pred": y_test_pred_poly_log,   "train_r2": train_r2_poly},
    {"name": "SVR",                    "y_pred": y_test_pred_svr,        "train_r2": train_r2_svr},
    {"name": "Decision Tree",          "y_pred": y_test_pred_dt,         "train_r2": train_r2_dt },
    {"name": "XGBoost",                "y_pred": y_pred_xgb,             "train_r2": train_r2_xgb},
    {"name": "Neural Network",         "y_pred": y_pred_nn,              "train_r2": train_r2_nn },
    {"name": "Voting Ensemble",         "y_pred": final_pred,             "train_r2": train_r2_ens},
]

rows = []

for m in models:
    r2   = r2_score(y_test_log, m["y_pred"])
    rmse = np.sqrt(mean_squared_error(y_test_log, m["y_pred"]))
    acc  = r2 * 100

    if m["train_r2"] is not None:
        diff   = (m["train_r2"] - r2) * 100
        status = "Overfitting" if diff > 10 else "Balanced"
    else:
        status = "—"

    rows.append([
        m["name"],
        f"{rmse:.4f}",
        f"{r2:.4f}",
        f"{acc:.2f}%",
        status,
    ])

headers = ["Model", "RMSE", "R2", "Accuracy", "Status"]

print("\n" + "=" * 70)
print(tabulate(rows, headers=headers, tablefmt="grid"))
print("=" * 70)
print("  ____The End ____ 👌🤩")


all_predictions = {
    'Random Forest':         (y_test_pred_log,       y_train_pred_rf,          y_test_log, y_train_log),
    'Gradient Boosting':     (y_test_pred_gb,         y_train_pred_gb,          y_test_log, y_train_log),
    'XGBoost':               (y_pred_xgb,             y_train_pred_xgb,         y_test_log, y_train_log),
    'Decision Tree':         (y_test_pred_dt,         y_train_pred_dt,          y_test_log, y_train_log),
    'Neural Network':        (y_pred_nn,              y_train_pred_nn,          y_test_log, y_train_log),
    'Ridge Regression':      (y_test_pred_ridge_log,  y_train_pred_ridge_log,   y_test_log, y_train_log),
    'Lasso Regression':      (y_test_pred_lasso_log,  y_train_pred_lasso_log,   y_test_log, y_train_log),
    'Linear Regression':     (y_test_pred_lin_log,    y_train_pred_lin_log,     y_test_log, y_train_log),
    'Elastic Net':           (y_pred,                 y_train_pred,             y_test_log, y_train_log),
    'Polynomial Regression': (y_test_pred_poly_log,   y_train_pred_poly_log,    y_test_log, y_train_log),  # ✅
    'SVR':                   (y_test_pred_svr,        y_train_pred_svr_scaled,  y_test_log, y_train_log),  # ✅
    'Voting Ensemble':       (final_pred,             final_train_pred,         y_test_log, y_train_log),
}



print("\nSaving all artifacts …")

# ── Scalers ──────────────────────────────────────────────
joblib.dump(scaler,          "robust_scaler.pkl")

# SVR re-fits standard_scaler on its OWN data, so save a separate copy.
# Add these two lines RIGHT AFTER the SVR y-scaling block in main_final.py:
#
#   joblib.dump(standard_scaler, "svr_standard_scaler.pkl")   # after X_train_svr fit
#   joblib.dump(standard_scaler, "svr_y_scaler.pkl")          # after y_train_svr fit
#
# (See exact placement notes below)

# ── Feature selectors ────────────────────────────────────
joblib.dump(selector_lin,   "selector_lin.pkl")    # fitted in SelectKBest section
joblib.dump(selector_ridge, "selector_ridge.pkl")  # fitted in Ridge section
joblib.dump(selector_svr,   "selector_svr.pkl")    # fitted in SVR section
joblib.dump(selector_final, "selector_final.pkl")  # fitted for Linear Regression (best_k)


# ── Column lists / metadata ──────────────────────────────
joblib.dump(to_drop,            "corr_to_drop_rf.pkl")          # RF-path correlation drops
joblib.dump(to_drop_corr_lin,   "corr_to_drop_lin.pkl")         # linear-path correlation drops
joblib.dump(features_to_scale,  "features_to_scale.pkl")
joblib.dump(standard_features,  "standard_features.pkl")
joblib.dump(leakage_cols,       "leakage_cols.pkl")
joblib.dump(final_columns,      "final_columns_rf.pkl")         # RF/XGB/GBM column order
joblib.dump(selected_linear_features, "selected_lin_cols.pkl")
joblib.dump(train_medians,      "train_medians.pkl")
joblib.dump(date_mode,          "date_mode.pkl")
joblib.dump(cols_to_drop_final, "cols_to_drop_final.pkl")
joblib.dump(valid_langs,        "valid_langs.pkl")
joblib.dump(best_k,             "best_k.pkl")

joblib.dump(X_train_dt.columns.tolist(), "dt_columns.pkl")


# ── Models ───────────────────────────────────────────────
joblib.dump(rf_model,       "rf_model.pkl")
joblib.dump(gb_model,       "gb_model.pkl")
joblib.dump(xgb_model,      "xgb_model.pkl")
joblib.dump(dt_model,       "dt_model.pkl")
joblib.dump(ridge_model,    "ridge_model.pkl")
joblib.dump(lr_model,       "lr_model.pkl")
joblib.dump(lasso_model,    "lasso_model.pkl")
joblib.dump(poly_model,     "poly_model.pkl")
joblib.dump(svr_model,      "svr_model.pkl")
joblib.dump(model,          "elastic_model.pkl")
joblib.dump(ensemble_final, "ensemble_final.pkl")
joblib.dump(X_train_lin_num.columns.tolist(), "lin_num_columns.pkl")
# Save Neural Network in Keras native format


nn_model_ref = nn_model
nn_model.save("nn_model.keras")

print("All artifacts saved ✓")