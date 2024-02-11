
# this script initially plots only the sankey showing flow between banks 
# I want to see some numbers as well
# For example, 

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import seaborn as sns
import os

curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('PlottingScripts')[0]

full_df = pd.read_csv(curr_abs_path + '/OutputData/data_with_classified_scam_almost_final_Interest_fixed.csv')
full_df.columns

# Create a DataFrame for the color mapping

# Group by 'bank_from' and 'bank_to', and calculate the count
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

# Create a chord diagram figure
fig = go.Figure(go.Sankey(
    node=dict(
        pad=5,
        thickness=20,
        line=dict(color="black", width=0.5),
        color=all_bank_namesRGB['colors'].to_list()
    ),
    link=dict(
        source=[source.index(s) for s in source] + [len(source) + target.index(t) for t in target],
        target=[len(source) + target.index(t) for t in target] + [source.index(s) for s in source],
        value=value,
        color=all_bank_namesRGB['colors'].to_list()
    )
))

all_bank_namesRGB.columns = ['bank', 'colour']
all_bank_colour = all_bank_namesRGB.drop_duplicates()

factro = (full_df_agg.groupby('bank_from')['count'].mean()/10000).reset_index()
pos1 = [0.06, 0.27, 0.55, 0.78, 0.95]
for x in range(len(np.unique(source))):
    fig.add_annotation(
        x=0.0,  # X-coordinate of the annotation (to the left)
        y=1-pos1[x], #((1 + 1) / ((x+8))) + factro['count'][x],  # Y-coordinate of the annotation
        text=np.unique(source)[x],  # Label text
        showarrow=False,  # Do not show an arrow
        font=dict(size=20, color =  all_bank_colour.loc[all_bank_colour['bank'] == np.unique(source)[x], 'colour'].values[0]),  # Adjust font size
        xshift=-80,  # Adjust the position to the left
        align='right'  # Align text to the right of the specified coordinates
    )

width = 700

pos2 = [-0.035, 0.95, 0.75, 0.52, 0.28, 0.08]
posx = [1.14] + 5*[1.18]
for x in range(len(np.unique(target))):
    fig.add_annotation(
        x=posx[x],  # X-coordinate of the annotation (to the left)
        y=pos2[x], #((1 + 1) / ((x+8))) + factro['count'][x],  # Y-coordinate of the annotation
        text=np.unique(target)[x],  # Label text
        showarrow=False,  # Do not show an arrow
        font=dict(size=20, color =  all_bank_colour.loc[all_bank_colour['bank'] == np.unique(target)[x], 'colour'].values[0]),  # Adjust font size
        align='left'  # Align text to the right of the specified coordinates
    )

fig.add_annotation(
    x = -0.1, y = 1.1,
    text = 'Bank from:',
    showarrow = False
)
fig.add_annotation(
    x = 1.1, y = 1.1,
    text = 'Bank to:',
    showarrow = False
)

# Customize the layout and appearance
fig.update_layout(margin=dict(l=100, r=100, t=60, b=30),
                  font=dict(size = 20, color = 'black'), width = width, height = 500)
fig.show()
