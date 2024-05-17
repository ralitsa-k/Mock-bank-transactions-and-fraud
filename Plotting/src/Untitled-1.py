
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from base64 import b64encode
import numpy as np
import os

app = Dash(__name__)

curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('PlottingScripts')[0]

app.layout = html.Div([
    html.H1('Flow between banks '),
    html.P("Choose Fraud or Non-Fraud:"),
    dcc.RadioItems(
        id='fraud_option',
        options=[
            {'label': 'Fraud', 'value': 'fraud'},
            {'label': 'NonFraud', 'value': 'non-fraud'}
        ],
        value='fraud'
    ),
    html.Div(id='output'),
])

@app.callback(
    Output("output", "children"), 
    Input('fraud_option', 'value'))
def display_graph(fraud_option):

    full_df1 = pd.read_csv(curr_abs_path + '/OutputData/data_with_classified_scam_almost_final_Interest_fixed.csv')

    if fraud_option == 'fraud':
        full_df = full_df1.loc[full_df1.is_scam_transaction == 1,:]
        title = "Flow of money in FRAUD transactions"
    elif fraud_option == 'non-fraud':
        full_df = full_df1.loc[full_df1.is_scam_transaction == 0,:]
        title = "Flow of money in NON-FRAUD transactions"

        
    full_df_agg = full_df.groupby(['bank_from', 'bank_to']).size().reset_index(name='count')


    # Create the source, target, and value lists
    source = full_df_agg['bank_from'].tolist()
    target = full_df_agg['bank_to'].tolist()
    value = full_df_agg['count'].tolist()
          
    # Create a DataFrame containing all unique bank names (source and target)
    all_bank_namesRGB = pd.DataFrame(source + target)

    # Define a list of colors for the banks
    #colors = ['rgba(206, 4, 42, 0.8)', 'rgba(239, 124, 10, 0.8)','rgba(140, 30, 128, 0.8)','rgba(76, 32, 128, 0.8)', 'rgba(25, 94, 153, 0.8)', 'rgba(0, 138, 98, 0.8)']
    colors = ['rgba(206, 4, 42, 0.8)', 'rgba(239, 124, 10, 0.8)','rgba(140, 30, 128, 0.8)','rgba(76, 32, 128, 0.8)', 'rgba(25, 94, 153, 0.8)', 'rgba(0, 138, 98, 0.8)']


    # Create a mapping from bank names to colors
    color_mapping = dict(zip(all_bank_namesRGB[0].unique(), colors))
    # Map the colors to the bank names
    # Apply the hex to RGB conversion and add the opacity value

    all_bank_namesRGB['colors'] = all_bank_namesRGB[0].map(color_mapping)
    # Function to convert a hex color to an RGB tuple

    # Create a chord diagram figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color="black", width=0.5),
            color=all_bank_namesRGB['colors'].to_list(),
            label=source + target
        ),
        link=dict(
            source=[source.index(s) for s in source] + [len(source) + target.index(t) for t in target],
            target=[len(source) + target.index(t) for t in target] + [source.index(s) for s in source],
            value=value,
            color=all_bank_namesRGB['colors'].to_list()
        )
    ))

    fig.update_layout(margin=dict(l=100, r=100, t=60, b=30),
                  font=dict(size = 20, color = 'black'), width = 600, height = 350)
    
    return dcc.Graph(figure=fig) 
app.run_server(debug=False)


