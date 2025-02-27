import os
import json
import numpy as np
import pandas as pd
import logging

from tqdm import tqdm


logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def log_training_dynamics(output_dir, epoch, train_ids, train_logits, train_labels):
    """
    Save training dynamics (logits) from given epoch as record of a `.jsonl` file.
    """
    training_dynamics_df = pd.DataFrame({
        'uuid': train_ids,
        f'logits_epoch_{epoch}': train_logits,
        'label': train_labels
    })

    logging_dir = os.path.join(output_dir, f'training_dynamics')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    epoch_file_name = os.path.join(logging_dir, f'training_dynamics_epoch_{epoch}.jsonl')
    training_dynamics_df.to_json(epoch_file_name, lines=True, orient='record')
    logger.info(f'Training Dynamics logged to {epoch_file_name}')


def read_training_dynamics(model_dir, id_field='uuid'):
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
    - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    training_dynamics = {}

    training_dynamics_dir = os.path.join(model_dir, 'training_dynamics')
    num_epochs = len([file for file in os.listdir(training_dynamics_dir) if os.path.isfile(os.path.join(training_dynamics_dir, file))])

    logger.info(f'Reading {num_epochs} files from  {training_dynamics} ...')

    for epoch_num in tqdm(range(num_epochs)):
        epoch_file = os.path.join(training_dynamics_dir, f'training_dynamics_epoch_{epoch_num}.jsonl')
        assert os.path.exists(epoch_file)

        with open(epoch_file, 'r') as file:
            for line in file:
                record = json.loads(line.strip())
                id = record[id_field]
                if id not in training_dynamics:
                    assert epoch_num == 0
                    training_dynamics[id] = {'label': record['label'], 'logits': []}
                training_dynamics[id]['logits'].append(record[f'logits_epoch_{epoch_num}'])

    logger.info(f'Read training dynamics for {len(training_dynamics)} train instances.')

    return training_dynamics
