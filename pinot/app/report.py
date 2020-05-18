# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pinot
import pandas as pd

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def dataframe(results_dict):
    # get all the results
    metrics = list(list(results_dict.values())[0].keys())
    ds_names = list(results_dict.keys())
    n_metrics = len(metrics)
    df = pd.DataFrame(
        [[value['final'].round(4) for metric, value in results.items()] for ds_name, results in results_dict.items()],
        columns=metrics,
        index=ds_names)
    return df
    
def markdown(results_dict):
    df = dataframe(results_dict)
    return df.to_markdown()

def visual(results_dict):
    # make plots less ugly
    from matplotlib import pyplot as plt

    plt.rc("font", size=14)
    plt.rc("lines", linewidth=6)

    # initialize the figure
    fig = plt.figure(figsize=(8, 3))

    # get all the results
    metrics = list(list(results_dict.values())[0].keys())
    n_metrics = len(metrics)

    # loop through metrics
    for idx_metric, metric in enumerate(metrics):
        ax = plt.subplot(1, n_metrics, idx_metric + 1)
        
        # loop through the results
        for ds_name, results in results_dict.items():

            # get all the recorded indices
            idxs = list(
                    [
                        key for key in results[metric].keys() if isinstance(key, int)
                    ])

            # sort it ascending
            idxs.sort()

            ax.plot(
                idxs,
                [
                    results[metric][idx]
                    for idx in idxs
                ],
                label=ds_name,
            )

        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.legend()

    return fig

def visual_base64(results_dict):
    fig = visual(results_dict)
    import io
    import base64
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=50)
    img.seek(0)
    img = base64.b64encode(img.read()).decode('utf-8')
    # img = "![img](data:image/png;base64%s)" % img
    return(img)

def html(results_dict):
    html_string = ""

    if isinstance(results_dict, dict):
        results_dict = [results_dict]
    
    for _results_dict in results_dict:

        html_string += """
        <p>
        <div style='height:15%%;width:100%%;'>
            <div style='float:left'>
                <img src='data:image/png;base64, %s'/>
            </div>
            <div style='float:left'>
                %s
            </div>
        </div>
        <br><br><br>
        <p/>
        """ % (visual_base64(_results_dict)[:-1], dataframe(_results_dict).to_html())

    return html_string

def html_multiple_train_and_test(results):
    html_string = ""
    for param, result in results:
        html_string += '<p><br><br><br>' + str(param) + '<p/>'
        html_string += html(result)
        html_string += '<br><br><br>'
        
    return html_string

def html_multiple_train_and_test_2d_grid(results):
    # make sure there are only two paramter types
    param_names = list(results[0][0].keys())
    assert len(param_names) == 2
    param_col_name, param_row_name = param_names

    param_col_values = list(set([result[0][param_col_name] for result in results ]))
    param_row_values = list(set([result[0][param_row_name] for result in results ]))

    param_col_values.sort()
    param_row_values.sort()
    

    # initialize giant table in nested lists
    table = [['NA' for _ in param_col_values] for _ in param_row_values]

    # populate this table
    for idx_col, param_col in enumerate(param_col_values):
        for idx_row, param_row in enumerate(param_row_values):
            param_dict = {
                        param_col_name: param_col, 
                        param_row_name: param_row
                        }

            # TODO:
            # make this less ugly

            for result in results:
                if result[0] == param_dict:
                    table[idx_row][idx_col] = html(result[1])
            
    html_string = ""
    html_string += "<table style='border: 1px solid black'>"
    

    # first row
    html_string += "<thread><tr style='border: 1px solid black'>"
    html_string += "<th style='border: 1px solid black'>" +\
            param_row_name + "/" + param_col_name +  "</th>"

    for param_col in param_col_values:
        html_string += "<th style='border: 1px solid black'>" + str(param_col) + "</th>"
    
    html_string += "</tr></thread>"

    # the rest of the rows
    for idx_row, param_row in enumerate(param_row_values):
        html_string += "<tr style='border: 1px solid black'>"
        
        # html_string += "<td></td>"

        html_string += "<th style='border: 1px solid black'>" + param_row  + " </th>"
        
        for idx_col, param_col in enumerate(param_col_values):
            html_string += "<td style='border: 1px solid black'>" + table[idx_row][idx_col] + "</td>"


        html_string += "</tr>"

    html_string += "</table>"
    return html_string
