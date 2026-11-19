"""Parse the published fine-tuning command with its actual official defaults."""
import official_bootstrap as bootstrap
import argparse
import json
from pathlib import Path

from magat_plus.training_args import add_training_args
from magat_plus.run_expert import add_expert_dataset_args
from magat_plus.convert_to_imitation_dataset import add_imitation_dataset_args
from magat_plus.generate_additional_data import add_additional_data_args


def official_parser():
    p = argparse.ArgumentParser()
    for add in (add_expert_dataset_args, add_imitation_dataset_args, add_additional_data_args, add_training_args):
        p = add(p)
    return p


def official_arguments(checkpoints=None):
    return ['--dataset_dir', str(bootstrap.ART/'dataset'), '--num_samples', '1000',
        '--expert_algorithm', 'LaCAM-30', '--obs_radius', '5', '--map_type', 'mixed',
        '--map_types', 'pregen=1.0', '--num_agents', '32+48+64+80',
        '--max_episode_steps', '512', '--min_dist', '10',
        '--map_dir', str(bootstrap.ART/'maps/dense_warehouse'), '--num_maps', '1',
        '--maps_name', 'dense_warehouse', '--save_termination_state',
        '--add_data_cost_to_go', '--normalize_cost_to_go', '--use_lists',
        '--checkpoints_dir', str(checkpoints or bootstrap.ART/'checkpoints'),
        '--run_name', 'magat_plus_dense_warehouse_finetuned', '--device', '-1',
        '--run_online_expert', '--num_epochs', '52',
        '--attention_mode', 'MAGAT_multiplicative', '--model_residuals', 'all',
        '--use_edge_attr', '--use_edge_attr_for_messages', 'positions+manhattan',
        '--edge_attr_processor', 'MLP', '--load_positions_separately',
        '--train_on_terminated_agents', '--recursive_oe', '--cnn_mode', 'ResNetLarge_withMLP',
        '--collision_shielding', 'naive', '--action_sampling', 'probabilistic',
        '--pretrain_weights_path', str(bootstrap.CHECKPOINT)]


def configuration(checkpoints=None):
    return official_parser().parse_args(official_arguments(checkpoints))


if __name__ == '__main__':
    bootstrap.ART.mkdir(parents=True, exist_ok=True)
    args = configuration()
    (bootstrap.ART/'official-configuration.json').write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({k:getattr(args,k) for k in ['num_epochs','batch_size','num_agents',
        'num_samples','num_run_oe','validation_every_epochs','lr_start','lr_end','weight_decay',
        'model_seed','dataset_seed','collision_shielding','action_sampling','action_sampling_temperature']}, indent=2))
