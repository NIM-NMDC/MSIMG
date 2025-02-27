import logging
import seaborn as sns
import matplotlib.pyplot as plt


logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def plot_data_map(dataframe, plot_dir, hue_metric, title, model_name, show_hist=False, max_instances_to_plot=55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

    logger.info(f'Plotting figure for {title} using the {model_name} model ...')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else dataframe.shape[0])

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f'{x:.1f}' for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10),)
        grid_spec = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(grid_spec[:, 0])

    # Make the scatter plot.
    # Choose a palette.
    palette = sns.diverging_palette(260, 15, n=num_hues, sep=10, center='dark')
    plot = sns.scatterplot(
        data=dataframe,
        x=main_metric,
        y=other_metric,
        hue=hue,
        style=style,
        palette=palette,
        ax=ax0,
        s=30
    )

    # Annotate regions.
    # ec = edge color, lw = line width, fc = face color.
    box_border = lambda c: dict(boxstyle='round,pad=0.3', ec=c, lw=2, fc='white')
    annotate_func = lambda text, xyc, box_border_color : ax0.annotate(
        text, xy=xyc, xycoords='axes fraction', fontsize=15, color='black', va='center', ha='center', rotation=350, bbox=box_border(box_border_color)
    )
    an1 = annotate_func('ambiguous', xyc=(0.9, 0.5), box_border_color='black')
    an2 = annotate_func('easy-to-learn', xyc=(0.27, 0.85), box_border_color='red')
    an3 = annotate_func('hard-to-learn', xyc=(0.35, 0.25), box_border_color='blue')

    if not show_hist:
        plot.legend(nloc=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')
    plot.set_title(f'{title}-{model_name} Data Map', fontsize=17)

    if show_hist:
        # Make the histograms.
        ax1 = fig.add_subplot(grid_spec[0, 1])
        ax2 = fig.add_subplot(grid_spec[1, 1])
        ax3 = fig.add_subplot(grid_spec[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True)  # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{plot_dir}/{title}_{model_name}.pdf' if show_hist else f'figures/compact_{title}_{model_name}.pdf'
    fig.savefig(filename, dpi=300)
    logger.info(f'Plot saved to {filename} ...')



