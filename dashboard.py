import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data (replace this with your actual loading method)
# For example: df = pd.read_csv('your_data.csv')
# Here’s just a mock snippet:
df = pd.read_csv("dataset_cleaned.csv")  

# Your preprocessing pipeline
int_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
for col in int_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

for col in df.select_dtypes(include=['object', 'bool']).columns:
    df[col] = df[col].astype('category')

for col in df.select_dtypes(include='category').columns:
    df[col] = df[col].cat.codes
    df[col] = df[col].replace(-1, np.nan)

X = df.drop("subscribed", axis=1)
y = df["subscribed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('pca', PCA(n_components=3)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# App initialization
app = dash.Dash(__name__)
server = app.server

# UI layout
app.layout = html.Div([
    html.H2("Bank Marketing Subscription Prediction"),
    html.Div([
        html.Div([
            html.Label(col),
            dcc.Dropdown(
                id=col,
                options=[{'label': str(v), 'value': v} for v in sorted(X[col].dropna().unique())],
                value=X[col].dropna().unique()[0]
            )
        ]) if str(X[col].dtype).startswith('int') == False else html.Div([
            html.Label(col),
            dcc.RadioItems(
                id=col,
                options=[{'label': str(v), 'value': v} for v in sorted(X[col].dropna().unique())],
                value=X[col].dropna().unique()[0]
            )
        ])
        for col in X.columns
    ]),
    html.Br(),
    html.Button("Predict", id="predict-button", n_clicks=0),
    html.Div(id="prediction-output")
])

# Callback
@app.callback(
    Output("prediction-output", "children"),
    [Input(col, "value") for col in X.columns] + [Input("predict-button", "n_clicks")]
)
def predict_callback(*args):
    input_features = args[:-1]  # exclude n_clicks
    if None in input_features:
        return "Please fill in all fields."

    input_df = pd.DataFrame([input_features], columns=X.columns)
    pred = pipeline.predict(input_df)[0]
    return f"Prediction: {'Yes' if pred == 1 else 'No'}"

if __name__ == "__main__":
    app.run_server(debug=True)
