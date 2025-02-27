import os
import logging
import argparse

from utils.selection_utils import read_training_dynamics
from training_dynamics_filtering import compute_training_dynamics_metrics
from visualization.visualization_utils import plot_data_map


logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='../', required=True, help='Root directory of the project.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where model training dynamics stats reside.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where data for task resides.')
    parser.add_argument('--filtered_output_dir', type=str, help='Output directory where filtered datasets are to be written.')
    parser.add_argument('plot_dir', type=str, help='Directory where plots are to be saved.')
    parser.add_argument('--metric', choices=('threshold_closeness', 'confidence', 'variability', 'correctness', 'forgetfulness'), help='Metric to filter data by.')
    parser.add_argument('--filter', action='store_true', help='Whether to filter data subsets based on specified `metric`.')
    parser.add_argument('--plot', action='store_true', help='Whether to plot data maps and save as `pdf`.')
    parser.add_argument('--confidence_interval', action='store_true', help='Compute the confidence interval for variability.')
    parser.add_argument('--worst', action='store_true', help='Select from the opposite end of the spectrum acc to metric, for baseline.')
    parser.add_argument('--both_ends', action='store_true', help='Select from both ends of the spectrum acc to metric.')
    parser.add_argument('--burn_out', type=int, default=128, help='Epochs for which to compute training dynamics.')
    parser.add_argument('--model_name', type=str, help='Model name for which data map is being plotted.')

    args = parser.parse_args()

    training_dynamics = read_training_dynamics(args.model_dir, id_field='uuid')
    num_total_epochs = len(list(training_dynamics.values())[0]["logits"])
    args.burn_out = min(args.burn_out, num_total_epochs)
    logger.info(f'Using {args.burn_out} epochs for training dynamics (out of {num_total_epochs} available).')
    training_dynamics_metrics = compute_training_dynamics_metrics(args, training_dynamics)

    burn_out_str = f"epochs_{args.burn_out}" if args.burn_out > num_total_epochs else ""
    training_dynamics_filename = os.path.join(args.model_dir, f"training_dynamics_metrics_{burn_out_str}.jsonl")
    training_dynamics_metrics.to_json(training_dynamics_filename, orient='records', lines=True)
    logger.info(f"Metrics based on Training Dynamics written to {training_dynamics_filename}")

    # if args.filter:
    #     assert args.filtering_output_dir
    #     if not os.path.exists(args.filtering_output_dir):
    #         os.makedirs(args.filtering_output_dir)
    #     assert args.metric
    #     write_filtered_data(args, training_dynamics_metrics)

    if args.plot:
        assert args.plots_dir
        if not os.path.exists(args.plots_dir):
            os.makedirs(args.plots_dir)
        plot_data_map(training_dynamics_metrics, args.plots_dir, title=args.task_name, show_hist=True, model=args.model)