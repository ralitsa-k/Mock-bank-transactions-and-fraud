import matplotlib.pyplot as plt
from shiny import ui, render, App
import seaborn as sns
import os
import pandas as pd

curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('BankDataGen')[0]
full_df = pd.read_csv(curr_abs_path + 'BankDataGen/OutputData/data_with_classified_scam_almost_final_Interest_fixed.csv')
cc = full_df.groupby('customer_id').count()/10
cust_id_too_much = cc.loc[cc.Category > 70].index


app_ui = ui.page_fluid(
    ui.column(
        10,
        {"class": "col-md-10 col-lg-8 py-5 mx-auto text-lg-center text-left"},
        # Title
        ui.h1("Average Spent per Category"),
        # input slider
    ),
    ui.column(
        10,
        {"class": "col-md-15 col-lg-5 py-4 mx-auto"},
        # Title
         ui.input_selectize("fraud", "Data", 
                            choices = {'fraud':'Fraud','non_fraud':'Non-fraud'},
            width="100%",
        ),
    ),

    ui.panel_main(
        ui.row(
        ui.column(10,
            {"class": "card col-md-78 col-lg-5 py-4 mx-lg-center"},
            # output plot
            ui.output_plot("barplot", width = '100%')),
        ui.column(10,
            {"class":  "card col-md-78 col-lg-5 py-4 mx-lg-center"},
            # output plot
            ui.output_plot("boxplot", width = '100%'),
            )
        )
    )
)

def server(input, output, session):
    @output
    @render.plot
    def barplot():
        colors = ['#a7ba42','#95ccba','#ffdede','#f94f8a',
                '#fff0cb', '#f2cc84','#d1b2e0', '#660099','#079999']
        sns.set_palette(sns.color_palette(colors))

        if input.fraud() == 'non_fraud':
            plot_fraud = full_df.loc[~full_df.customer_id.isin(cust_id_too_much) &
                                    (full_df.fraud_type == 'none') &
                                    (full_df.type != 'income') &
                                    (full_df.Category != 'Housing') &
                                    (full_df.Category != 'Investment'), :].groupby(['Category'])['Amount'].mean().reset_index()
        elif input.fraud() == 'fraud':
            plot_fraud = full_df.loc[~full_df.customer_id.isin(cust_id_too_much) &
                                    (full_df.fraud_type != 'none') &
                                    (full_df.Category != 'Investment'), :].groupby(['Category'])['Amount'].mean().reset_index()
  
        plot_fraud = plot_fraud.sort_values('Amount',ascending=False)
        fig = plt.figure()
        ax = sns.barplot(plot_fraud,
                    y = 'Category',
                    x = 'Amount',
                    palette = colors)
        import textwrap
        def wrap_labels(ax, width, break_long_words=False):
            labels = []
            for label in ax.get_xticklabels():
                text = label.get_text()
                labels.append(textwrap.fill(text, width=width,
                              break_long_words=break_long_words))
            ax.set_xticklabels(labels, rotation=0)
            
        wrap_labels(ax, 5)
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        return fig
    @output
    @render.plot
    def boxplot():
        colors = ['#a7ba42','#95ccba','#ffdede','#f94f8a',
                '#fff0cb', '#f2cc84','#d1b2e0', '#660099','#079999']
        sns.set_palette(sns.color_palette(colors))

        if input.fraud() == 'non_fraud':
            plot_fraud = full_df.loc[~full_df.customer_id.isin(cust_id_too_much) &
                                    (full_df.fraud_type == 'none') &
                                    (full_df.type != 'income') &
                                    (full_df.Category != 'Housing') &
                                    (full_df.Category != 'Investment'), :].groupby(['Category','customer_id'])['Amount'].mean().reset_index()
        elif input.fraud() == 'fraud':
            plot_fraud = full_df.loc[~full_df.customer_id.isin(cust_id_too_much) &
                                    (full_df.fraud_type != 'none') &
                                    (full_df.Category != 'Investment'), :].groupby(['Category','customer_id'])['Amount'].mean().reset_index()
  
        plot_fraud = plot_fraud.sort_values('Amount',ascending=False)
        fig = plt.figure()
        ax = sns.boxplot(plot_fraud,
                    y = 'Category',
                    x = 'Amount',
                    palette = colors)
        import textwrap
        def wrap_labels(ax, width, break_long_words=False):
            labels = []
            for label in ax.get_xticklabels():
                text = label.get_text()
                labels.append(textwrap.fill(text, width=width,
                              break_long_words=break_long_words))
            ax.set_xticklabels(labels, rotation=0)
            
        wrap_labels(ax, 5)
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        return fig
    
app = App(app_ui, server)