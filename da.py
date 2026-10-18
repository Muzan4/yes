# =====================================================================
# 0. MASTER IMPORTS (Memorize this block for all practicals)
# =====================================================================
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Stats & Association Rules
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
import pyfpgrowth

# Scikit-Learn (Preprocessing, Models, Clustering, Metrics)
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    RocCurveDisplay, silhouette_score
)

# Quick tip to check sub-modules in exam if you forget names:
# print(sklearn.__all__)


# =====================================================================
# PRAC 1: BASIC DATA CLEANING, IMPUTATION & SCALING
# =====================================================================
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.upper()

# Inspect missing data
print("Missing values:\n", df.isna().sum())
print("Rows with any missing value:\n", df[df.isna().any(axis=1)])

# Handle missing values by column
df["NAME"]  = df["NAME"].fillna("unknown")
df["CITY"]  = df["CITY"].ffill().bfill()          # Fixed: .ffill().bfill()
df["DOB"]   = df["DOB"].fillna("1900-01-01")
df["MARKS"] = pd.to_numeric(df["MARKS"], errors="coerce")
df["MARKS"] = df["MARKS"].fillna(df["MARKS"].mean())

# Feature Scaling (Min-Max & Z-Score)
df["MARKSMINMAX"] = MinMaxScaler().fit_transform(df[["MARKS"]])
df["MARKSZSCORE"] = StandardScaler().fit_transform(df[["MARKS"]])

print(df["MARKS"].describe())

# Dropna variations & Save
df_any = df.dropna(how="any")
df_all = df.dropna(how="all")
df.to_csv("cleaned.csv", index=False)             # Fixed: index=False


# =====================================================================
# PRAC 2: OUTLIER REMOVAL (IQR METHOD) & VALIDATION
# =====================================================================
try:
    df = pd.read_csv('noisy_data.csv')
except:
    df = pd.DataFrame({
        'Date': ['01-01-2023', '02-01-2023', '03-01-2023', '04-01-2023', '05-01-2023'],
        'Product_ID': ['P1001', 'P1002', 'P1003', 'P1004', 'P1005'],
        'Price': [25.99, 19.95, 12.49, 9.99, 150],
        'Quantity': [5, 3, 8, 12, 1],
        'Sales_Region': ['North', 'South', 'East', 'West', 'North'],
        'Discount': ['0.1', '0.05', '0.15', '0.2', '0']
    })

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')

# Boxplot BEFORE cleaning
plt.boxplot(df['Price']); plt.title("Price Before Cleaning"); plt.show()

# Reusable IQR Outlier Function (Replaces outliers with NaN)
def remove_outliers(col):
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    return col.where((col >= q1 - 1.5 * iqr) & (col <= q3 + 1.5 * iqr))

df['Price'] = remove_outliers(df['Price'])
df['Quantity'] = remove_outliers(df['Quantity'])

# Filter valid categories & impute numeric medians
valid_regions = ['North', 'South', 'East', 'West']
df['Sales_Region'] = df['Sales_Region'].where(df['Sales_Region'].isin(valid_regions))
df = df.dropna(subset=['Sales_Region'])

cols = ['Price', 'Quantity', 'Discount']
df[cols] = df[cols].fillna(df[cols].median())

# Boxplot AFTER cleaning & Save
plt.boxplot(df['Price']); plt.title("Price After Cleaning"); plt.show()
df.to_csv("cleaned_sales_data.csv", index=False)
print("Cleaned Data:\n", df)


# =====================================================================
# PRAC 3: DYNAMIC PREPROCESSING PIPELINE (STUDENT DATA)
# =====================================================================
df = pd.read_csv("student-data.csv")
print("Missing before:\n", df.isna().sum())

# 1. Separate & fill categorical vs numerical columns automatically
cat_cols = df.select_dtypes(include=['object', 'string']).columns
num_cols = df.select_dtypes(include='number').columns

df[cat_cols] = df[cat_cols].ffill().bfill()
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 2. Scale Age
df["age_minmax"] = MinMaxScaler().fit_transform(df[["age"]])
df["age_zscore"] = StandardScaler().fit_transform(df[["age"]])

# 3. Boxplot Before -> Remove Outliers (IQR) -> Fill Median -> Boxplot After
plt.boxplot(df["age"]); plt.title("Age Before Outlier Removal"); plt.show()

Q1, Q3 = df["age"].quantile(0.25), df["age"].quantile(0.75)
IQR = Q3 - Q1
df["age"] = df["age"].where((df["age"] >= Q1 - 1.5 * IQR) & (df["age"] <= Q3 + 1.5 * IQR))
df["age"] = df["age"].fillna(df["age"].median())

plt.boxplot(df["age"]); plt.title("Age After Outlier Removal"); plt.show()

df_any = df.dropna(how='any')
df_all = df.dropna(how='all')
df.to_csv("student-data-cleaned.csv", index=False)


# =====================================================================
# PRAC 4: CORRELATION MATRIX, HEATMAP & CHI-SQUARE TEST
# =====================================================================
df = pd.read_csv("Suicide-rates-master.csv")

# Part A: Correlation Heatmap (Numeric Columns)
corr = df.select_dtypes(include=['number']).corr()
print("Correlation Matrix:\n", corr)

sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Part B: Chi-Square Test of Independence (Categorical Columns)
def run_chi2(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        return print(f"Error: '{col1}' or '{col2}' not found.")
    
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(table)
    
    print(f"--- Chi-Square: {col1} vs {col2} ---")
    print(f"Chi2: {chi2:.4f} | p-value: {p:.4f} | DoF: {dof}")
    print("Conclusion:", "REJECT H0 (Dependent)" if p < 0.05 else "FAIL TO REJECT H0 (Independent)")

run_chi2(df, "Gender", "Education Level")
run_chi2(df, "country", "age")
run_chi2(df, "sex", "generation")


# =====================================================================
# PRAC 5: DESCRIPTIVE STATS, SKEWNESS & VISUALIZATION SUITE
# =====================================================================
df = pd.read_csv("Suicide-rates-master.csv")

# 1. Summary Stats & Skewness Check
for col in df.select_dtypes(include=[np.number]).columns:
    mean, med, mode, skew = df[col].mean(), df[col].median(), df[col].mode()[0], df[col].skew()
    interp = "Symmetric" if np.isclose(mean, med, atol=0.1) else ("Right-skewed" if skew > 0 else "Left-skewed")
    print(f"{col} | Mean: {mean:.2f} | Med: {med:.2f} | Mode: {mode:.2f} | Skew: {skew:.2f} -> {interp}")

# 2. Basic Plots (Bar, Hist, Box, Violin, Scatter)
df['sex'].value_counts().plot(kind='bar', title='Bar - Gender'); plt.show()
plt.hist(df['suicides_no'], bins=20); plt.title("Histogram"); plt.show()
plt.boxplot(df['suicides_no']); plt.title("Box Plot"); plt.show()
plt.violinplot(df['suicides_no']); plt.title("Violin Plot"); plt.show()
plt.scatter(df['year'], df['suicides_no']); plt.title("Scatter Plot"); plt.show()

# 3. Advanced Plots (Stacked Bar & Donut Chart)
df.pivot_table(values='suicides_no', index='year', columns='sex', aggfunc='sum').plot(kind='bar', stacked=True)
plt.title("Stacked Bar"); plt.show()

plt.pie(df.groupby('sex')['suicides_no'].sum(), labels=['Female', 'Male'], autopct='%1.1f%%')
plt.gca().add_artist(plt.Circle((0, 0), 0.70, fc='white'))
plt.title("Donut Chart"); plt.show()


# =====================================================================
# PRAC 6: SINGLE CLASSIFIER (DECISION TREE) + METRICS & TREE PLOT
# =====================================================================
df = pd.read_csv("diabetes.csv")
X, y = df.drop(columns=['Outcome']), df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Std Error:", ((acc * (1 - acc)) / len(y_test)) ** 0.5)
print("Precision:", precision_score(y_test, y_pred))
print("Sensitivity (Recall):", recall_score(y_test, y_pred))
print("Specificity:", recall_score(y_test, y_pred, pos_label=0)) # Trick: pos_label=0 gives Specificity!
print("AUC:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Built-in ROC Plot & Tree Visualization
RocCurveDisplay.from_estimator(clf, X_test, y_test, plot_chance_level=True)
plt.show()

plot_tree(clf, feature_names=X.columns, class_names=['0', '1'], filled=True, max_depth=3)
plt.show()


# =====================================================================
# PRAC 7: MULTI-MODEL COMPARISON (DT, RF, NB, LR, SVM) & COMBINED ROC
# =====================================================================
df = pd.read_csv("diabetes.csv")
X, y = df.drop(columns=['Outcome']), df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "DT":  DecisionTreeClassifier(max_depth=3, random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "NB":  GaussianNB(),
    "LR":  LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    spec = recall_score(y_test, y_pred, pos_label=0)
    cm   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- {name} ---")
    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f} | AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison'); plt.legend(); plt.show()


# =====================================================================
# PRAC 9: CLUSTERING (K-MEANS, HIERARCHICAL, PCA & SILHOUETTE)
# =====================================================================
df = pd.read_csv("Mall_Customers.csv")
print(df.describe())

X = df.drop(columns=['CustomerID', 'Gender'])
X_scaled = StandardScaler().fit_transform(X)

# 1. Elbow Method to find optimal K
wss = [KMeans(k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 16)]
plt.plot(range(1, 16), wss, marker='o'); plt.title("Elbow Method"); plt.show()

# 2. K-Means + PCA 2D Plot
km = KMeans(3, n_init=10, random_state=42).fit_predict(X_scaled)
X_pca = PCA(2).fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km); plt.title("K-Means (PCA)"); plt.show()

# 3. Dendrogram + Hierarchical Clustering
dendrogram(linkage(X_scaled, method='ward')); plt.title("Dendrogram"); plt.show()

hc = AgglomerativeClustering(3).fit_predict(X_scaled)
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=hc)
plt.title("Hierarchical Clustering"); plt.show()

# 4. Evaluation Scores & Cluster Counts
print("K-Means Silhouette:", silhouette_score(X_scaled, km))
print("Hierarchical Silhouette:", silhouette_score(X_scaled, hc))
print("\nK-Means Counts:\n", pd.Series(km).value_counts())
print("\nHierarchical Counts:\n", pd.Series(hc).value_counts())


# =====================================================================
# PRAC 10: MARKET BASKET ANALYSIS (APRIORI & ASSOCIATION RULES)
# =====================================================================
data = pd.read_csv("groceries - groceries.csv")

# 1-Line One-Hot Encoding (Stack -> Get Dummies -> Groupby Level 0)
df = pd.get_dummies(data.stack()).groupby(level=0).any()

# Apriori & Top 15 Frequent Itemsets Plot
freq = apriori(df, min_support=0.01, use_colnames=True)
freq.nlargest(15, 'support').plot.bar(x='itemsets', y='support'); plt.show()

# Generate & Filter Rules
r = association_rules(freq, metric='lift', min_threshold=1)
print(r.sort_values('support', ascending=False))
print(r[r.antecedents.str.len() >= 2].query("confidence > 0.3"))
print(r[r.consequents.str.len() >= 2])

# Loop through different rule metrics
for m in ['lift', 'confidence', 'leverage', 'conviction']:
    print(m, "\n", association_rules(freq, metric=m, min_threshold=1))


# =====================================================================
# PRAC 11 & 12: FP-GROWTH (RAW TRANSACTIONS vs DISCRETIZED NUMERICALS)
# =====================================================================

# --- PRAC 11: Direct CSV Transaction Mining ---
data = [[x for x in r if x] for r in csv.reader(open('Book1.csv'))][1:]
print("Some records:", data[:10])

patterns = pyfpgrowth.find_frequent_patterns(data, 2) # Min support count = 2
print("Patterns:", patterns)


# --- PRAC 12: Binning Continuous Data into Categories for FP-Growth ---
trans = [[
    "High_Glucose" if float(r[1]) >= 126 else "Normal_Glucose",
    "High_BMI"     if float(r[5]) >= 25  else "Normal_BMI",
    "Age_40_plus"  if int(r[7]) >= 40    else "Age_below_40",
    "Diabetes"     if r[8] == "1"        else "No_Diabetes"
] for r in list(csv.reader(open('diabetes.csv')))[1:]]

print("Some Records:", *trans[:10], sep="\n")

patterns = pyfpgrowth.find_frequent_patterns(trans, 200) # Min support count = 200
print("\nFrequent Patterns:\n", pd.Series(patterns))
