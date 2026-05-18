import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from tabulate import tabulate
nltk.download('punkt')
nltk.download('punkt_tab')



#Load csv
Data = pd.read_csv("C:/Users/hp/Downloads/Milestone 2 Updated Datasets/Milestone 2 Updated Datasets/Datasets train splits/Online Games Popularity Predcition/train_data.csv")

# delete duplicate
Data = Data.drop_duplicates()
for col in Data.select_dtypes(include=['object']).columns:
    Data[col] = Data[col].str.strip() # delete any spaces

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

#drop "PriceCurrency" col => constant
Data.drop(columns=['PriceCurrency'], inplace=True, errors='ignore')
print("PriceCurrency column dropped.")

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

# => # of supported languages
Data['Num_Supported_Languages'] = Data['SupportedLanguages'].fillna('').apply(
    lambda x: len(x.split()))

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
numeric_cols = Data.select_dtypes(include=[np.number]).columns

object_cols = Data.select_dtypes(include=['object']).columns
Data[object_cols] = Data[object_cols].fillna("Unknown")

Data.loc[Data['PriceFinal'] == 0, 'IsFree'] = True
Data.loc[Data['IsFree'] == True, ['PriceInitial', 'PriceFinal']] = 0.0
mask = Data['PriceInitial'] < Data['PriceFinal']
Data.loc[mask, 'PriceInitial'] = Data.loc[mask, 'PriceFinal'] # logical features

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
Data['Price_Success_Index'] = Data['Metacritic_Per_Year'] * (Data['PriceFinal_Log'] + 1)
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
X = Data.drop(columns=['GamePopularity'])

fill_value = 'Medium'

Data['GamePopularity'] = Data['GamePopularity'].fillna(fill_value)

joblib.dump(fill_value, "target_fill_value.pkl")

target_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

y = Data['GamePopularity'].map(target_mapping)

joblib.dump(target_mapping, "target_mapping.pkl")
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
skewed_features = [
    'SteamSpyPlayersEstimate', 'SteamSpyOwners',
    'DLCCount', 'AchievementCount', 'PriceFinal', 'PriceInitial'
]
#=======================================================
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
standard_scaler = StandardScaler()
X_train[standard_features] = standard_scaler.fit_transform(X_train[standard_features])
X_val[standard_features]   = standard_scaler.transform(X_val[standard_features])
X_test[standard_features]  = standard_scaler.transform(X_test[standard_features])
#===========================
# Feature Selection
#===========================
#  Correlation Feature Selection (x_train data)
corr_matrix = X_train.select_dtypes(include=[np.number]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

print("\n--- Feature Selection: Correlation Result ---")
print(f"Number of columns GamePopularity for removal due to redundancy:", len(to_drop))

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


X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_val   = X_val.apply(pd.to_numeric, errors='coerce')
X_test  = X_test.apply(pd.to_numeric, errors='coerce')

X_train = X_train.fillna(0)
X_val   = X_val.fillna(0)
X_test  = X_test.fillna(0)

print(f"Data cleaned. Protected features kept. Remaining: {X_train.shape[1]}")


#==================================
#GRADIENTBOOSTING
#===================================
param_grid = {
    'n_estimators': [150, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best Model
best_model_GB = grid.best_estimator_

y_train_pred = best_model_GB.predict(X_train)

y_val_pred = best_model_GB.predict(X_val)

# Test Accuracy
y_test_pred = best_model_GB.predict(X_test)

print("      GRADIENTBOOSTING MODEL  FINISHED       ")

#==================================
# MLP -   NEURAL-NETWORK
#===================================
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train, y_train)


y_train_pred = mlp.predict(X_train)

y_val_pred = mlp.predict(X_val)

y_test_pred = mlp.predict(X_test)

print("      MLP MODEL  FINISHED       ")

# =================================================
# CATBOOST CLASSIFIER - HYPERPARAMETER TUNING
# ================================================

def run_experiment(iterations, lr, param_name, param_val):
    model = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=6, random_seed=42, verbose=0)

    # Calculate Training Time
    start_train = time.time()
    model.fit(X_train_cat, y_train)
    train_time = time.time() - start_train

    # Calculate Test Time
    start_test = time.time()
    test_predictions = model.predict(X_test_cat)
    test_time = time.time() - start_test

    train_acc = accuracy_score(y_train, model.predict(X_train_cat))
    test_acc = accuracy_score(y_test, model.predict(X_test_cat))

    return {
        'param': param_name, 'value': param_val,
        'train_acc %': f"{train_acc * 100:.2f}%",
        'test_acc %': f"{test_acc * 100:.2f}%",
        'gap %': f"{(train_acc - test_acc) * 100:.2f}%",
        'train_time': round(train_time, 2),
        'test_time (s)': round(test_time, 4)
    }

X_train_cat = X_train.select_dtypes(include=[np.number]).copy()
X_test_cat = X_test.select_dtypes(include=[np.number]).copy()
results_log = []

# Experiments
for it in [500, 1000, 1500]:
    results_log.append(run_experiment(it, 0.01, 'iterations', it))

for lr in [0.01, 0.05, 0.1]:
    results_log.append(run_experiment(500, lr, 'learning_rate', lr))

# print(tabulate(pd.DataFrame(results_log), headers='keys', tablefmt='pretty', showindex=False))

# Final Model Training & Saving
final_model = CatBoostClassifier(iterations=500, learning_rate=0.01, depth=6, random_seed=42, verbose=0)
final_model.fit(X_train_cat, y_train)



# =================================================
# XGB
# ================================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

best_acc = 0
best_model = None
best_params = None

n_trials = 15

for i in range(n_trials):

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "seed": 42,
        "max_depth": random.choice([4, 5,6]),
        "learning_rate": random.choice([0.01, 0.05, 0.1]),
        "subsample": random.choice([0.7, 0.8, 1.0]),
        "colsample_bytree": 0.8,
        "gamma": 0,
        "min_child_weight": 1
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    preds = np.argmax(model.predict(dval), axis=1)
    acc = accuracy_score(y_val, preds)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_params = params

# Test Evaluation
test_preds = np.argmax(best_model.predict(dtest), axis=1)

print("       XGBOOST CLASSIFIER MODEL  FINISHED       ")
# print("=" * 60)



#===============================================
#  LogisticRegression
#===============================================

model = LogisticRegression(
    C=1,
    solver='lbfgs',
    class_weight='balanced',
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_val)
acc = accuracy_score(y_val, pred)
print("      LogisticRegression  MODEL  FINISHED       ")

# print(f"Validation Accuracy = {acc * 100:.2f}%")

# =====================================================
#    KNN
# =====================================================

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=80, random_state=42)
# X_train_knn = pca.fit_transform(X_train)
# X_val_knn   = pca.transform(X_val)
#
# final_knn = KNeighborsClassifier(
#     n_neighbors=5,
#     metric='manhattan',
#     weights='distance'
# )
#
# final_knn.fit(X_train_knn, y_train)
# pred = final_knn.predict(X_val_knn)
# acc  = accuracy_score(y_val, pred)
# print("KNN MODEL FINISHED")
# =====================================================
#    KNN
# =====================================================

from sklearn.decomposition import PCA

pca = PCA(
  n_components=30,
  random_state=42
)
X_train_knn = pca.fit_transform(X_train)
X_val_knn   = pca.transform(X_val)

final_knn = KNeighborsClassifier(
  n_neighbors=5,
  metric='manhattan',
  weights='distance'
)
final_knn.fit(X_train_knn, y_train)
pred = final_knn.predict(X_val_knn)
acc  = accuracy_score(y_val, pred)
print("KNN MODEL FINISHED")


#==================================
#LIGHTGBM
#===================================
lgbm_model = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=20,
    max_depth=-1,
    min_child_samples=100,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    importance_type='gain',
    verbose=-1
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

y_train_pred = lgbm_model.predict(X_train)
y_val_pred   = lgbm_model.predict(X_val)
y_test_pred  = lgbm_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val, y_val_pred)
test_acc  = accuracy_score(y_test, y_test_pred)

if train_acc - test_acc > 0.05:
    print("\n⚠️ Warning: Potential Overfitting detected! (Large gap between Train and Test)")
else:
    print("\n✅ Success: Model seems to generalize well.")
print("      LIGHTGBM  MODEL  FINISHED       ")



#=================================================================================================

X_train_numeric = X_train.select_dtypes(include=[np.number])
X_val_numeric = X_val.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])

scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train_numeric)
X_val_scaled = scaler_svm.transform(X_val_numeric)

# ==========================================================
# SVM
# ==========================================================
svm_params = {
    'C':     [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.01]
}

svm_grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    svm_params, cv=3, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)

print("       SVM  MODEL  FINISHED       ")

# ==========================================================
# Decision Tree
# ==========================================================
dt_params = {
    'max_depth':        [6, 8, 12],
    'min_samples_leaf': [5, 10, 20]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=42),
    dt_params, cv=3, scoring='accuracy', n_jobs=-1
)
dt_grid.fit(X_train_numeric, y_train)
print("       DECISION TREE MODEL  FINISHED       ")


# ==========================================================
# Random Forest
# ==========================================================
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [10, 15, 20]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', min_samples_leaf=5, random_state=42),
    rf_params, cv=3, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train_numeric, y_train)



print("       RANDOM FOREST MODEL  FINISHED       ")


# =====================================================
# Best Params Tables
# =====================================================
best_params_data = [
    ["Logistic Regression", "C=1, solver=lbfgs"],
    ["KNN",                 "n_neighbors=5, metric=manhattan"],
    ["LightGBM",            "num_leaves=20, max_depth=6, lr=0.01"],
    ["XGBoost",             "max_depth=4, lr=0.1, subsample=0.7"],
    ["CatBoost",            "iterations=500, lr=0.01"],
    ["GradientBoosting",    "n_estimators=150, lr=0.05, max_depth=3"],
    ["MLP",                 "hidden=(128,64,32), relu, adam"],
    ["SVM",                 "C=10.0, gamma=0.01"],
    ["Decision Tree",       "max_depth=12, min_samples_leaf=5"],
    ["Random Forest",       "max_depth=20, n_estimators=300"],
]

print("\n========== Best Parameters Per Model ==========")
print(tabulate(best_params_data, headers=["Model", "Best Params"], tablefmt="fancy_grid"))

# =====================================================
# Final Comparison Table
# =====================================================
models = {
    "Logistic Regression": (model,                        X_train, X_val, y_train, y_val),
    "KNN":                 (final_knn,                    X_train_knn, X_val_knn, y_train, y_val),
    "LightGBM":            (lgbm_model,                   X_train, X_val, y_train, y_val),
    "CatBoost":            (final_model,                  X_train_cat, X_test_cat, y_train, y_test),
    "GradientBoosting":    (grid.best_estimator_,         X_train, X_val, y_train, y_val),
    "MLP":                 (mlp,                          X_train, X_val, y_train, y_val),
    "SVM":                 (svm_grid.best_estimator_,     X_train_scaled, X_val_scaled, y_train, y_val),
    "Decision Tree":       (dt_grid.best_estimator_,      X_train_numeric, X_val_numeric, y_train, y_val),
    "Random Forest":       (rf_grid.best_estimator_,      X_train_numeric, X_val_numeric, y_train, y_val),
}

comparison = []

for name, (m, X_tr, X_vl, y_tr, y_vl) in models.items():
    train_preds = m.predict(X_tr)
    val_preds   = m.predict(X_vl)

    train_acc = accuracy_score(y_tr, train_preds)
    val_acc   = accuracy_score(y_vl, val_preds)
    f1        = f1_score(y_vl, val_preds, average='weighted')

    gap = train_acc - val_acc
    if gap > 0.10:
        status = "Overfitting"
    elif val_acc < 0.70:
        status = "Underfitting"
    else:
        status = "Balanced"

    comparison.append([name, f"{train_acc*100:.2f}%", f"{val_acc*100:.2f}%", f"{f1:.4f}", status])

xgb_train_preds = np.argmax(best_model.predict(dtrain), axis=1)
xgb_val_preds   = np.argmax(best_model.predict(dval), axis=1)
xgb_train_acc   = accuracy_score(y_train, xgb_train_preds)
xgb_val_acc     = accuracy_score(y_val, xgb_val_preds)
xgb_f1          = f1_score(y_val, xgb_val_preds, average='weighted')
xgb_gap         = xgb_train_acc - xgb_val_acc
xgb_status      = "Overfitting" if xgb_gap > 0.10 else ("Underfitting" if xgb_val_acc < 0.70 else "Balanced")
comparison.append(["XGBoost", f"{xgb_train_acc*100:.2f}%", f"{xgb_val_acc*100:.2f}%", f"{xgb_f1:.4f}", xgb_status])

print("\n========== Final Model Comparison ==========")
print(tabulate(
    comparison,
    headers=["Model", "Train Acc", "Val Acc", "F1 (weighted)", "Status"],
    tablefmt="fancy_grid"
))



# ==========================================================
# Save
# ==========================================================
joblib.dump(lgbm_model, "lgbm_model.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
joblib.dump(standard_scaler, "standard_scaler.pkl")
joblib.dump(scaler_svm, "svm_scaler.pkl")
joblib.dump(train_medians, "train_medians.pkl")
joblib.dump(date_mode, "date_mode.pkl")
joblib.dump(X_train.columns.tolist(), "final_features_list.pkl")
joblib.dump(final_knn, "knn_model.pkl")
joblib.dump(pca, "knn_pca.pkl")
joblib.dump(rf_grid.best_estimator_, "random_forest_model.pkl")
joblib.dump(dt_grid.best_estimator_, "decision_tree_model.pkl")
joblib.dump(svm_grid.best_estimator_, "svm_model.pkl")
joblib.dump(model, "logistic_model.pkl")
joblib.dump(best_model, "xgb_random_search.pkl")
joblib.dump(final_model, "final_catboost_model.pkl")
joblib.dump(mlp, "mlp_model.pkl")
joblib.dump(best_model_GB, "gradient_boosting_model.pkl")
# joblib.dump(grid.best_estimator_, "gradient_boosting_model.pkl")


joblib.dump(X_test, "X_test_processed.pkl")
joblib.dump(y_test, "y_test.pkl")

print("All models saved!")