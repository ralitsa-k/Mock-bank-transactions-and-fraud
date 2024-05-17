
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import os 


CONTENT_STYLE = {
    "margin-left": "25rem",
    'backgorund-color': '#1e1e1e',
    "margin-top":'4rem'
}
top_CONTENT_STYLE = {
    "margin-left": "30rem",
    'backgorund-color': '#1e1e1e'
}

bottom_cont_style = {
    "margin-left": "25rem",
    "margin-right":'5rem',
    "padding": "2rem 1rem",
    'backgorund-color': '#1e1e1e'
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "5rem",
    "left": "2rem",
    "bottom": "5rem",
    "width": "20rem",
    "height": "30rem",
    "padding": "2rem 1rem",
    "background-color": "#2b2b2b"
    
}

radio_style = {"padding-left": "10px",
               'display':'inline-block',
               'font-size': '30px',
               'text-align':'center'}

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],assets_folder ="static",
                 assets_url_path="static")
server = app.server
    
# DIVISIONS
header  = html.Div([ html.Div(
        className="app-header",
        children=[
            html.H2('Bank-Transactions Mock Data',
                     className="app-header--title")])
        ]
    )



# side = html.Div([
#     html.Center(
#         children=html.Div([
#             html.H3('Filter data:')
#         ], style = {'padding-bottom': '20px'})
#     ),
#     html.Center(dcc.RadioItems(
#         id='fraud_option',
#         options=[
#             {'label': html.Div(['Fraud'], className = 'button-44'), 'value': 'fraud'},
#             {'label': html.Div(['NonFraud'], className = 'button-44'), 'value': 'non-fraud'}
#         ],
#         value='fraud',
#         labelStyle={'display': 'block',
#                     'margin-bottom': '20px',
#                     'position':'center'
#                     })
                
#     )], style = SIDEBAR_STYLE)

side = html.Div([
    dbc.Row(html.Center(
        children=html.Div([
            html.H3('Filter data:')
        ], style = {'padding-bottom': '20px'})
    )),

    dbc.Row(html.Center(dcc.RadioItems(
        id='fraud_option',
        options=[
            {'label': html.Div(['Fraud'], className = 'button-44'), 'value': 'fraud'},
            {'label': html.Div(['NonFraud'], className = 'button-44'), 'value': 'non-fraud'}
        ],
        value='fraud',
        labelStyle={'display': 'block',
                    'margin-bottom': '20px',
                    'position':'center'
                    }))),
        dbc.Row(html.Center(
        children=html.Div([
            html.H3(id = 'output_KPI')
        ], style = {'padding-bottom': '20px'})
    )),
    
    ], style = SIDEBAR_STYLE)

# APP LAYOUT
bar = html.Div(id='output_ratio')            

top_row = html.Div([bar], style = CONTENT_STYLE, className='column1')

rows = html.Div([
    dbc.Row([dbc.Col(top_row)]),
     dbc.Row([dbc.Col(html.Div(id = 'output2')),
              dbc.Col(html.Div([html.Div(id='SankeyChart')]))], style = CONTENT_STYLE)
    ]
)

app.layout = html.Div([header, side, rows])

full_df1 = pd.read_csv(r'C:\Users\Ralitsa\Documents\All Files\DataSci\BanksDataGen\banks-data-gen\OutputData\final_bank_data.csv')

@app.callback(
    Output("SankeyChart", "children"), 
    Input('fraud_option', 'value'))

def display_graph(fraud_option):

    if fraud_option == 'fraud':
        full_df = full_df1.loc[full_df1.fraud_type != 'none',:]
        title = "Flow of money in FRAUD transactions"
    elif fraud_option == 'non-fraud':
        full_df = full_df1.loc[full_df1.fraud_type == 'none',:]
        title = "Flow of money in NON-FRAUD transactions"
  
    full_df_agg = full_df.groupby(['bank_from', 'bank_to']).size().reset_index(name='count')

    # Create the source, target, and value lists
    source = full_df_agg['bank_from'].tolist()
    target = full_df_agg['bank_to'].tolist()
    value = full_df_agg['count'].tolist()
          
    all_bank_namesRGB = pd.DataFrame(source + target)
    colors = ['#b25221 ','#ffba03', ' #63082b ',
                ' #656565 ',' #cc3851',' #ffffff ']
    color_mapping = dict(zip(all_bank_namesRGB[0].unique(), colors))
    all_bank_namesRGB['colors'] = all_bank_namesRGB[0].map(color_mapping)
    # Function to convert a hex color to an RGB tuple
    # Create a Sankey diagram figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color="black", width=0.5),
            color=all_bank_namesRGB['colors'].to_list(),
            label=source + target, 
            hovertemplate='Count %{value}<extra></extra>'
        ),
        link=dict(
            source=[source.index(s) for s in source] + [len(source) + target.index(t) for t in target],
            target=[len(source) + target.index(t) for t in target] + [source.index(s) for s in source],
            value=value,
            color=all_bank_namesRGB['colors'].to_list()
        )
    ))

    fig.update_layout(margin=dict(l=10, r=10, t=20, b=20),
                  font=dict(size = 16, color = 'white'), width = 600, height = 500,
                  plot_bgcolor='#222222',
                      paper_bgcolor = '#222222',
                      template = "plotly_dark")
    
    return dcc.Graph(figure=fig)


@app.callback(
    Output("output2", "children"), 
    Input('fraud_option', 'value'))
def display_graph2(fraud_option):


    if fraud_option == 'fraud':
        full_df = full_df1.loc[full_df1.fraud_type != 'none',:]
        title = "Flow of money in FRAUD transactions"
    elif fraud_option == 'non-fraud':
        full_df = full_df1.loc[full_df1.fraud_type == 'none',:]
        title = "Flow of money in NON-FRAUD transactions"
    
    colors = ['#f4ec5f','#ffba03','#ff7630','#f7443a','#ff4666',
                '#dc1636', '#c00033','#ab0132','#8d032f',
                '#63082b', '#571845','#a32d81','#989898']


    if fraud_option != 'fraud':
        plot_fraud = full_df.loc[(full_df1.fraud_type == 'none') &
                                (full_df.type != 'income') &
                                (full_df.Category != 'Housing') &
                                (full_df.Category != 'Investment'), :].groupby(['Category'])['Amount'].mean().reset_index()
    elif fraud_option == 'fraud':
        plot_fraud = full_df.loc[(full_df1.fraud_type != 'none') &
                                (full_df.Category != 'Investment'), :].groupby(['Category'])['Amount'].mean().reset_index()

    plot_fraud = plot_fraud.sort_values('Amount',ascending=False)
    plot_fraud['Amount'] = round(plot_fraud['Amount'],0)
   
    fig = px.bar(plot_fraud, y = 'Category', x = 'Amount', text="Amount",
                 labels=dict(Amount="Average Amount (£)"),color = 'Category',
                 color_discrete_sequence=colors)
    fig.update_layout(showlegend = False)
    fig.update_layout(margin=dict(l=10, r=10, t=0, b=0),
                      plot_bgcolor='#222222',
                      paper_bgcolor = '#222222',
                      template = "plotly_dark",font=dict(family="sans-serif",
                                                         size=14, color='white'), 
                      width = 700, height = 600,
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False))
    fig.update_traces(textfont_size=15,  textangle=0, textposition="outside", cliponaxis=False)

    return dbc.Col(dcc.Graph(figure=fig))

@app.callback(
    Output("output_KPI", "children"), 
    Input('fraud_option', 'value'))
def return_text(fraud_option):

    if fraud_option == 'fraud':
        full_df = full_df1.loc[full_df1.fraud_type != 'none',:]
    elif fraud_option == 'non-fraud':
        full_df = full_df1.loc[full_df1.fraud_type == 'none',:]
    value =  round(full_df.loc[full_df.type == 'spending', ['Amount']].sum().Amount)
    return f'Spent: £{value:,} '


@app.callback(
    Output("output_ratio", "children"),
    Input('fraud_option', 'value'))
def return_text(fraud_option):

    f_c = full_df1[['customer_id', 'customer_scammed']].drop_duplicates()
    counts = f_c.groupby('customer_scammed').count().reset_index()
    counts['customer_scammed'] = np.where(counts['customer_scammed']==1, 'scammed', 'not-scammed')
    counts = counts.sort_values('customer_scammed', ascending = False)

    
    counts1 = counts.loc[counts.customer_scammed == 'scammed', :]
    counts2 = counts.loc[counts.customer_scammed == 'not-scammed', :]

    if fraud_option == 'fraud':
        colors = ['#cc3851', '#656565']
        fig = px.bar(counts1.assign(bar = 1), x="customer_id", y = 'bar', text="customer_id", 
                    color = "customer_scammed",orientation='h',hover_name = 'customer_scammed',
                    hover_data={"customer_scammed":False, 'customer_id': False, 'bar':False},range_x=[0, 500],
        color_discrete_sequence=colors)
    
    
        bar2 = px.bar(counts2.assign(bar = 1), x="customer_id", y = 'bar', text="customer_id", 
                    color = "customer_scammed",orientation='h',hover_name = 'customer_scammed',
                    hover_data={"customer_scammed":False, 'customer_id': False, 'bar':False},range_x=[0, 500],
                    color_discrete_sequence=['#656565', '#cc3851'])
        fig.update_traces(textfont_size=20, textangle=0, cliponaxis=False, 
                    textposition='auto',marker_line=dict(width=2, color='white'))
        bar2.update_traces(textfont_size=20, textangle=0, cliponaxis=False, 
                    textposition='auto')
        fig.add_trace(bar2['data'][0])

    elif fraud_option == 'non-fraud':
        
        colors = ['#656565', '#cc3851']
        fig = px.bar(counts1.assign(bar = 1), x="customer_id", y = 'bar', text="customer_id", 
                    color = "customer_scammed",orientation='h',hover_name = 'customer_scammed',
                    hover_data={"customer_scammed":False, 'customer_id': False, 'bar':False},range_x=[0, 500],
        color_discrete_sequence=colors)
    
    
        bar2 = px.bar(counts2.assign(bar = 1), x="customer_id", y = 'bar', text="customer_id", 
                    color = "customer_scammed",orientation='h',hover_name = 'customer_scammed',
                    hover_data={"customer_scammed":False, 'customer_id': False, 'bar':False},range_x=[0, 500],
                    color_discrete_sequence=['#cc3851', '#656565'])
        bar2.update_traces(textfont_size=20, textangle=0, cliponaxis=False, 
                        textposition='auto',marker_line=dict(width=2, color='white'))
        fig.add_trace(bar2['data'][0])
        fig.update_traces(textfont_size=20, textangle=0, cliponaxis=False, 
                        textposition='auto')
    fig.update_layout(plot_bgcolor='#222222',
                      paper_bgcolor = '#222222',
                      template = "plotly_dark",
                      margin=dict(l=50, r=100, t=20, b=30),
                      xaxis_visible=False, yaxis_visible = False, 
                      autosize=False, width=900, height = 110, 
                      legend_title = None,
                      legend_font_size = 15)
        
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=False, host = '0.0.0.0', port=80)
    server = app.server
    

    
    
    
    
    