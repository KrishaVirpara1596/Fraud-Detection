
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE=Path(__file__).parent
DATA=BASE/"data"/"insurance_fraud_data.csv"
df=pd.read_csv(DATA).dropna(subset=["fraud reported"]).copy()
target="fraud reported"
drop_cols=["claim_number","claim_date","zip_code"]
X=df.drop(columns=[target]+drop_cols)
y=df[target].map({"N":0,"Y":1})
cat_cols=X.select_dtypes(include=["object"]).columns.tolist()
num_cols=[c for c in X.columns if c not in cat_cols]
prep=ColumnTransformer([
("num",SimpleImputer(strategy="median"),num_cols),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                 ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),cat_cols)
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=Pipeline([("preprocess",prep),("classifier",HistGradientBoostingClassifier(
    max_iter=250,learning_rate=.06,max_leaf_nodes=15,l2_regularization=1.0,random_state=42
))])
model.fit(X_train,y_train)
prob=model.predict_proba(X_test)[:,1]
pred=(prob>=.5).astype(int)
metrics={k:float(v) for k,v in {
"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred),
"recall":recall_score(y_test,pred),"f1":f1_score(y_test,pred),"roc_auc":roc_auc_score(y_test,prob)
}.items()}
joblib.dump(model,BASE/"model.joblib")
json.dump({"rows":len(df),"features":len(X.columns),"target":target,"dropped_columns":drop_cols,
"train_rows":len(X_train),"test_rows":len(X_test),"fraud_rate":float(y.mean()),
"threshold":.25,"metrics_at_default_0.5":metrics,
"categorical_features":cat_cols,"numeric_features":num_cols},open(BASE/"metadata.json","w"),indent=2)
print("Saved model.joblib and metadata.json")
print(metrics)
