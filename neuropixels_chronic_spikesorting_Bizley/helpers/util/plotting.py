#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Plotting helper functions and color definitions
"""

import numpy as np
import matplotlib.pyplot as plt
import math


NICE_COLORS = {'white': 3 * [255],
               'black': 3 * [0],
               'blue': [0, 120, 255],
               'orange': [255, 110, 0],
               'green': [35, 140, 45],
               'red': [200, 30, 15],
               'violet': [220, 70, 220],
               'turquoise': [60, 134, 134],
               'gray': [130, 130, 130],
               'lightgray': 3 * [150],
               'darkgray': 3 * [100],
               'yellow': [255, 215, 0],
               'cyan': [0, 255, 255],
               'dark orange': [244, 111, 22],
               'deep sky blue': [0, 173, 239],
               'deep sky blue dark': [2, 141, 212],
               'tomato': [237, 28, 36],
               'forest green': [38, 171, 73],
               'orange 2': [243, 152, 16],
               'crimson': [238, 34, 53],
               'jaguar': [35, 31, 32],
               'japanese': [59, 126, 52],
               'christi': [135, 208, 67],
               'curious blue': [2, 139, 210],
               'aluminium': [131, 135, 139],
               'buttercup': [224, 146, 47],
               'chateau green': [43, 139, 75],
               'orchid': [125, 43, 139],
               'fiord': [80, 96, 108],
               'punch': [157, 41, 51],
               'lemon': [217, 182, 17],
               'new mpl blue': [31, 119, 180],
               'new mpl red': [214, 39, 40]
               }

for k in NICE_COLORS:
    NICE_COLORS[k] = np.asarray(NICE_COLORS[k])/255.


def set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8, family='Arial'):

    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname(family)

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname(family)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname(family)

    if size_text is not None:
        for at in ax.texts:
            at.set_fontsize(size_text + add_size)
            at.set_fontname(family)


def adjust_axes(ax,
                tick_length=True,
                tick_direction=True,
                spine_width=0.5,
                pad=2):

    if tick_length:
        ax.tick_params(axis='both', which='major', length=2)

    if tick_direction:
        ax.tick_params(axis='both', which='both', direction='out')

    if pad is not None:
        ax.tick_params(axis='both', which='both', pad=pad)

    for s in ax.spines:
        spine = ax.spines[s]
        if spine.get_visible():
            spine.set_linewidth(spine_width)


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def simple_twinx_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')


def create_scalar_colormap(values=None, vmin=0, vmax=1,
                           cmap=plt.get_cmap('jet')):
    """create color values using scalar_map.to_rgba(value)"""

    import matplotlib.colors as colors
    import matplotlib.cm as cm

    cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    if values is not None:
        if isinstance(values, (list, np.ndarray)):
            return np.asarray([scalar_map.to_rgba(x) for x in values])
        else:
            return scalar_map.to_rgba(values)
    else:
        return scalar_map


def plot_scatter(ax, x, y, xerr, yerr, xlabel='', ylabel='',
                 calc_wilcoxon=True, color=3*[.5], ecolor=3*[.75],
                 fmt='o', show_N=True, lsline=False, r_value=False,
                 equal_scaling=True, **kwargs):
    """create scatter plot with equally-spaced axes, diagonal line etc."""

    from scipy import stats

    ax.axis('on')
    if equal_scaling:
        ax.axis('scaled')

    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt,
                    color=color, ecolor=ecolor, **kwargs)
    else:
        ax.scatter(x, y, color=color, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xerr is not None and yerr is not None:
        max_err = max([xerr.max(), yerr.max()])
    else:
        max_err = 0

    if equal_scaling:
        xmin = min([x.min(), y.min()]) - max_err
        xmax = max([x.max(), y.max()]) + max_err
        zz = np.linspace(xmin, xmax, 100)
        ax.plot(zz, zz, 'k--')

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)

    else:
        zz = np.linspace(x.min(), x.max(), 100)

    if lsline:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        ax.plot(zz, slope * zz + intercept, '-', linewidth=1, color=3*[0])

    txt = ''
    if show_N:
        txt += 'N = %d' % x.shape[0]

    if lsline and r_value:
        txt += '\nr = %g' % r_value

    if calc_wilcoxon:
        # NOTE: stats.wilcoxon might not be the best choice so better
        # use R's wilcoxon if you want to make sure things are correct
        p_value = stats.wilcoxon(x, y)[1]
        txt += '\n' + 'p = %.3e' % p_value

    if len(txt) > 0:
        ax.text(.05, .8, txt, transform=ax.transAxes)
        
        
def add_labels_bars(ax, labels, spacing=5):
    """Add labels to the end of each bar in a bar chart.
    
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
        of the plot to annotate.
        
        labels (list or array): List of labels, needs to be same length as 
        the number of bars in the plot. If None, plot the values of the y axis
        
        spacing (int): The distance between the labels and the bars.
    """
    
    # For each bar: Place a label
    for i, rect in enumerate(ax.patches):
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if labels is None:
            label = "{:.1f}".format(y_value)
        else:
            label=labels[i]
        
        if not math.isnan(label):
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                        # positive and negative values.