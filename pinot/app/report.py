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
        [[value['final'] for metric, value in results.items()] for ds_name, results in results_dict.items()],
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
    fig = plt.figure()

    # get all the results
    metrics = list(list(results_dict.values())[0].keys())
    n_metrics = len(metrics)
    # loop through metrics
    for idx_metric, metric in enumerate(metrics):
        ax = plt.subplot(n_metrics, 1, idx_metric + 1)

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
