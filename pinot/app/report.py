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
        [[value['final'].detach().numpy().round(4) for metric, value in results.items()] for ds_name, results in results_dict.items()],
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
            ax.plot(
                [
                    results[metric][idx].detach().numpy()
                    for idx in range(len(results[metric]) - 1)
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
    html_string = """
    <p>
    <div style='height:15%%;width:100%%;'>
        <div style='float:left'>
            <img src='data:image/png;base64, %s'/>
        </div>
        <div style='float:left'>
            %s
        </div>
    </div>
    <p/>
    """ % (visual_base64(results_dict)[:-1], dataframe(results_dict).to_html())

    return html_string

def html_multiple_train_and_test(results):
    html_string = ""
    for param, result in results:
        html_string += '<br>'
        html_string += '<hr>'
        html_string += '\n'
        html_string += '<br><p>' + str(param) + '<p/><br/>'
        html_string += html(result)
        html_string += '<br/>'
        
    return html_string
