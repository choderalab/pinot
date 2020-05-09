# =============================================================================
# IMPORTS
# =============================================================================
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pinot

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def markdown(results_dict):
    # initialize markdown string
    md = ""

    # get all the results
    metrics = list(list(results_dict.values())[0].keys())
    ds_names = list(results_dict.keys())

    n_metrics = len(metrics)

    md += '{:<15}'.format('|')
    for metric in metrics:
        md += '{:<15}'.format('|%s' % metric)
    md += '|'
    md += '\n'

    for _ in range(n_metrics + 1):
        md += '{:<15}'.format('|' + '-' * 13)

    md += '|'
    md += '\n'

    for ds_name, results in results_dict.items():
        md += '{:<15}'.format('|' + ds_name)

        for metric, value in results.items():
            md += '{:<15}'.format('|%.4f' % value['final'])

        md += '|'
        md += '\n'

    return md


def visual(results_dict):
    # make plots less ugly
    from matplotlib import pyplot as plt
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=6)

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
                [results[metric][idx].detach().numpy() for idx in range(
                    len(results[metric]) - 1)],
                label=ds_name)

        ax.set_xlabel('epochs')
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.legend()

    return fig
