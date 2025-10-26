import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

df = pd.read_csv("task_data.csv")  # path to file

# Pre-processing and data cleaning
df.columns = df.columns.str.strip()

new_cols = [i for i in df.columns if i not in ['ID', 'Cardiomegaly']]
object_cols = df[new_cols].select_dtypes(include=['object']).columns.tolist()

for col in object_cols:
    df[col] = df[col].str.replace(',', '.', regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop(['ID', 'Cardiomegaly'], axis=1)
y = df['Cardiomegaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# Decision Tree Classifier
model1 = DecisionTreeClassifier(max_depth=3, random_state=42)
model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)

print("Acuracy Decision Tree: ", accuracy_score(y_test, y_pred1))
print("\nConfusion matrix Decision Tree: ")
print(confusion_matrix(y_test, y_pred1))
print("\nClassification report Decision Tree: ")
print(classification_report(y_test, y_pred1))


# Logistic Regression Classifier
model2 = LogisticRegression(solver='liblinear', random_state=42)
model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)

print("Acuracy Logistic Regression: ", accuracy_score(y_test, y_pred2))
print("\nConfusion matrix Logistic Regression: ")
print(confusion_matrix(y_test, y_pred2))
print("\nClassification report Logistic Regression: ")
print(classification_report(y_test, y_pred2))


# Cross-validation
scores = cross_val_score(model2, X_train_scaled, y_train, cv=5)
print("Cross-validation results: \n", scores)
print("\nAverage accuracy:", scores.mean())

# AUC (ROC)
y_pred_proba1 = model1.predict_proba(X_test_scaled)[:, 1]
y_pred_proba2 = model2.predict_proba(X_test_scaled)[:, 1]

auc1 = roc_auc_score(y_test, y_pred_proba1)
auc2 = roc_auc_score(y_test, y_pred_proba2)

# Results table
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2)],
    'AUC (ROC)': [auc1, auc2]
})

print("\nResults tables:")
print(results)

# ROC curves
fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba1)
fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba2)

plt.figure(figsize=(6, 4))
plt.plot(fpr1, tpr1, label='Decission Tree')
plt.plot(fpr2, tpr2, label='Logistic Regression')
plt.plot([0,1], [0,1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()

# Precision–Recall curves
prec1, rec1, _ = precision_recall_curve(y_test, y_pred_proba1)
prec2, rec2, _ = precision_recall_curve(y_test, y_pred_proba2)

plt.figure(figsize=(6, 4))
plt.plot(rec1, prec1, label='Drzewo decyzyjne')
plt.plot(rec2, prec2, label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall curve')
plt.legend()
plt.show()





