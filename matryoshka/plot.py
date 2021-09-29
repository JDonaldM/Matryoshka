'''
File containing some simple plotting functions.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def sample_space(samples_list, set_labels, param_labels, save=False, figsize=(15,15), filename=None):
    '''
    Function for producing a corner plot of the sample space.
    Adapted from corner by Daniel Foreman-Mackey.

    Args:
        samples_list (list) : List of arrays with shape (n, d).
        set_labels (list) : Label for each element in `samples_list`.
        param_labels (list) : Labels for each dimension of the sample space. Should habe length d.
        save (bool) : If True, the plot will be saved. If True, `filename` must not be None.
         Default is False.
        figsize (tuple) : Tuple that defines the size of the plot. Default is (15, 15).
        filename (str) : Filename
    '''
    # How many sample sets to plot?
    N = len(samples_list)

    # How many dimensions?
    d = samples_list[0].shape[1]

    # TODO: This is super ugly. Implement proper colour cycle.
    colours = list(mcolors.TABLEAU_COLORS.keys())

    # How many samples in each set. Used for zorder
    sizes = []
    for i in range(N):
        sizes.append(samples_list[i].shape[0])

    fig, ax = plt.subplots(d,d,figsize=figsize)
    for i in range(d):
        for j in range(d):
            if i == j:
                for l in range(N):    
                    ax[i,j].hist(samples_list[l][:,i],density=True,histtype='step',
                                 linewidth=1,zorder=sizes[::-1][l], 
                                 color=colours[l])
                ax[i,j].set_yticklabels([])
                ax[i,j].set_title(param_labels[i])
            elif j < i:
                for l in range(N):
                    ax[i,j].scatter(samples_list[l][:,j], samples_list[l][:,i],s=2., 
                                    zorder=sizes[::-1][l], color=colours[l])
            if j > i:
                ax[i,j].set_frame_on(False)
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
            if i < 7 - 1:
                ax[i,j].set_xticklabels([])
            else:
                ax[i,j].set_xlabel(param_labels[j])
                [l.set_rotation(45) for l in ax[i,j].get_xticklabels()]
            if j != 0 :
                    ax[i,j].set_yticklabels([])
            elif i != j:
                ax[i,j].set_ylabel(param_labels[i])

    legend_patches = []
    for l in range(N):
        legend_patches.append(mpatches.Patch(label=set_labels[l], color=colours[l]))
    ax[1,3].legend(handles=legend_patches, fontsize="x-large", frameon=False)
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)

    if save is True:
        plt.savefig(filename)
    else:
        plt.show()

def per_err(truths, predictions, xvalues, xlabel=None, ylabel=None, save=False, filename=None):
    '''
    Function for plotting the scale dependent percentage error from
    emulator predictions.

    Args:
        truths (array) : Array containing the truth. Should have shape (n, k).
        predictions (array) : Array containing the emulator predictions.
         Shouls have shape (n, k).
        xvalues (array) : Array containing the x-values. Should have shape (k,).
        xlabel (str) : X-axis label.
        ylabel (str) : Y-axis label.
        save (bool) : If True, the plot will be saved. If True, `filename`
         must not be None. Default is False.
        filename (str) : Filename
    '''

    # Calculate the scale dependant percentage error.
    per_err = (predictions/truths - 1)*100

    plt.plot(xvalues, per_err.T, color='lightgrey',zorder=0,alpha=0.3)

    plt.fill_between(xvalues,
                       np.percentile(per_err,2.5,axis=0),np.percentile(per_err,100-2.5,axis=0), 
                       label=r'$95\%$',alpha=0.3,color='tab:blue')
    plt.fill_between(xvalues,
                       np.percentile(per_err,16,axis=0),np.percentile(per_err,100-16,axis=0),
                       label=r'$68\%$',alpha=0.3,color='tab:green')

    plt.set_xlabel(xlabel)
    plt.set_ylabel(ylabel)
    plt.grid(True,color='k',linestyle=':')
    plt.tight_layout()

    if save is True:
        plt.savefig(filename)
    else:
        plt.show()