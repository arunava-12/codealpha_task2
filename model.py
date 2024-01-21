import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv(r"C:\Users\HP\Documents\CodeAlpha\Credit Scoring\Dataset_Credit_Scoring.csv")

print(dataset.shape)
print(dataset.head(10))

# Drop 'ID' column
dataset = dataset.drop('ID', axis=1)

# Convert percentage and dollar columns to numeric and fill NaN values with the mean
percentage_columns = ['TLOpenPct', 'TLOpen24Pct', 'TLSatPct', 'TLBalHCPct']
dollar_columns = ['TLMaxSum', 'TLSum']

for col in percentage_columns + dollar_columns:
    dataset[col] = pd.to_numeric(dataset[col].str.replace('%', '').str.replace('$', ''), errors='coerce')

# Fill NaN values with the mean
dataset = dataset.fillna(dataset.mean())

y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

joblib.dump(sc, 'standard_scaler.joblib')
""" Later on, if you need to use the same scaling transformation on new data or in another script, you can load the StandardScaler object back from the file using joblib.load like :
loaded_scaler = joblib.load('standard_scaler.joblib')
new_data_scaled = loaded_scaler.transform(new_data) """

# Risk Model Building
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

predictions = classifier.predict_proba(X_test)
print(predictions)

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
print(df_prediction_prob)

df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
print(df_prediction_target)

df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])
print(df_test_dataset)

dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_csv('prediction_results.csv', index=False)

print(dfx.head(10))