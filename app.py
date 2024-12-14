import base64
from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pandas as pd
import io
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import plotly.express as px
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import os

app = Dash(__name__)
app.css.append_css({"external_url": "/assets/styles.css"})
app.config.suppress_callback_exceptions=True

df = pd.DataFrame()
model = ""

app.layout = html.Div(children=[
    dcc.Upload(
        id="upload_file",
        children=html.Button(
            'Upload File'
        ),
        style={
            'width':'100%',
            'text-align':'center'
        },
        multiple=False
    ),
    html.Div(
        id="select_target",
    ),
    html.Div(
        id="select_category"
    ),
    html.Div(
        id="feature_checklist"
    ),
    html.Div(
        id="train_model"
    ),
    html.Div(
        id="prediction_result"
    )
])

@app.callback(Output('select_target', 'children'), Input('upload_file', 'contents'), State('upload_file', 'filename'))
def load_dataset(contents, filename):
    #TODO: CHECK IF FILE IS CSV OR NOT AND OTHER ERROR HANDLING THINGS
    global df
    if contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = io.BytesIO(base64.b64decode(content_string))
    df = pd.read_csv(decoded)
    return (
        html.Div([
            html.Label("Select target: "),
            dcc.Dropdown(
                id="column_dropdown",
                options=[{'label':col, 'value':col} for col in df.columns],
            )
        ])
    )

@app.callback(Output('select_category', 'children'), Input('column_dropdown', 'value'))
def category_select(value):
    if value is not None:
        return html.Div([
            # Graphs
            html.Div(
                className="graphs-container",
                children=[
                    # Categorical Stuff
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    dcc.RadioItems(
                                        id="category_options",
                                        options=[
                                            {'label': col, 'value': col}
                                            for col in df.select_dtypes(exclude=['number']).columns
                                        ],
                                        inputStyle={"margin-right": "5px"},
                                        labelStyle={"display": "inline-block", "margin-right": "15px"}
                                    )
                                ],
                                style={
                                    'text-align': 'center', 
                                    'margin-bottom': '10px'
                                }
                            ),
                            dcc.Graph(
                                id="categorical_graph",
                            )
                        ],
                        style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
                    ),
                    # Correlation stuff
                    html.Div(
                        dcc.Graph(
                            id="correlation_graph",
                        ),
                        style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}
                    )
                ],
                style={
                    'display': 'flex', 
                    'justify-content': 'space-between', 
                    'width': '100%'
                }
            )
        ])



@app.callback(Output("categorical_graph", "figure"), Input("category_options", 'value'), State("column_dropdown", 'value'))
def create_categorical_bar_graph(x_value, y_value):
    if not x_value or not y_value:
        return {
            'data': [],
            'layout': {'title': ""}
        }
    grouped_df = df.groupby([x_value])[y_value].mean().reset_index()
    x = grouped_df[x_value]
    y = grouped_df[y_value]
    return{
        'data':[{'x':x, 'y':y, 'type':'bar'}],
        'layout': {'title': f"Average {y_value} by {x_value}"}
    }

@app.callback(Output("correlation_graph", "figure"), Input("column_dropdown", "value"))
def create_correlation_bar_graph(target):
    corr = df.select_dtypes(include=['number']).corr()
    x = [col for col in df.select_dtypes(include=['number'])]
    y = [corr.loc[feature, target] for feature in x]
    return{
        'data':[{'x':x, 'y':y, 'type':'bar'}],
        'layout': {'title': f"Correlation strength of numerical variables with {target}"}
    }


@app.callback(Output("feature_checklist", "children"), Input("column_dropdown", "value"))
def create_feature_list(value):
    if value is not None:
        feature_columns = [col for col in df.columns if col != value]
        return([
            dcc.Checklist(
                id="feature_list",
                options=[
                    {'label': col, 'value': col}
                    for col in feature_columns
                ],
                className='checklist-container'
            ),
            html.Div([
                html.Button(
                    "Train",
                    id="train_button",
                    className='train-button'
                )
            ], className='train-button-container')
        ])

class CustomXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, **kwargs):
        # A custom wrapper for XGBRegressor to make it fully compatible with Scikit-learn.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )

    def fit(self, X, y, **fit_params):
        return self.model.fit(X, y, **fit_params)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            **self.kwargs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Update the underlying XGBRegressor model
        self.model.set_params(**params)
        return self

@app.callback(Output('train_model', 'children'), Input('train_button', 'n_clicks'), State('feature_list', 'value'), State('column_dropdown', 'value'))
def create_model_training(n_clicks, selected_features, target):
    global model
    global features
    features = selected_features
    if n_clicks is None:
        return ""
    if n_clicks > 0:
        #JUST DOING A SIMPLE MODEL FOR NOW
        #TODO: ADD MORE COMPLEX MODEL WITH PIPELINE, ONEHOT ENCODING, AND HANDLING MISSING VALUES
        X = df[selected_features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        numericals = X.select_dtypes(include=['number']).columns
        categoricals = X.select_dtypes(exclude=['number']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numericals), # num is just a label or a name for the transformer.
                ('cat', categorical_transformer, categoricals)
            ]
        )
        
        # Define the pipeline with preprocessing and model
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', CustomXGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        param_grid = {
            'model__n_estimators': [50, 100, 200],      # Number of trees
            'model__max_depth': [6, 10, 20],     # Maximum depth of each tree
            'model__learning_rate': [0.01, 0.1, 0.2]      # Minimum samples to split an internal node
        }

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,
            scoring = 'neg_mean_squared_error',
            verbose=1,
            error_score='raise'
            )
        
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
    return(
        html.Div([
            html.Div(
                html.P("R^2 score: "+str(r2)),
                style={'text-align':'center', 'margin-top':'20px'}
            ),
            html.Div([
                html.Label("Enter comma-separated feature values:"),
                dcc.Input(
                    id="feature_input",
                    type="text",
                    placeholder=', '.join(selected_features),
                    style={'margin-bottom': '10px'}
                ),
                html.Button("Predict", id="predict_button"),
            ],
            style={'text-align':'center', 'margin-top':'20px'})
        ])
    )

@app.callback(
    Output('prediction_result', 'children'),
    Input('predict_button', 'n_clicks'),
    State('feature_input', 'value'),
)
def predict(n_clicks, feature_input):
    global model
    if n_clicks is None:
        return ""
    
    try:
        feature_values = [float(x) if x.replace('.', '', 1).isdigit() else x for x in feature_input.split(',')]
        
        #Normalize
        
        prediction = model.predict(pd.DataFrame([feature_values], columns = features))
        
        return (
            html.Div(
                html.P(f"Predicted target value: {prediction[0]}"),
                style={'text-align': 'center', 'margin-top': '20px'}
            )
        )
    except Exception as e:
        return (
            html.Div(
                html.P(f"Error: {str(e)}")
            )
        )
server = app.server

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True)