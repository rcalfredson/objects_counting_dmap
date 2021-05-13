from net_retrain_manager import NetRetrainManager
from batch_train_opts import options
import os
import subprocess

# -d egg-fullsize -n FCRN_A -m P:\Robert\objects_counting_dmap\model_backup\egg_FCRN_A_expanded_dataset_v2.pth -lr 0.0045 -e 1 -hf 0.5 -vf 0.5 --plot --batch_size 4
# 2021-03-05
# python trainByBatchLinux.py 5 '-d egg-unshuffled -n FCRN_A -lr 0.004 -e 3000 -hf 0.5 -vf 0.5 -rot --plot --batch_size 4 --config /media/Synology3/Robert/objects_counting_dmap/configs/shuffle_data_at_start_2021-02-23.json'


opts = options()


def run_one_training(existing_model=None):
    subprocess.call(
        "python train.py --export_at_end %s%s"
        % (
            opts.trainParams,
            ""
            if existing_model is None
            else f' -m "{os.path.join(opts.existing_nets, existing_model)}"',
        ),
        shell=True,
        preexec_fn=os.setsid,
    )


if opts.existing_nets:
    net_retrainer = NetRetrainManager(opts.existing_nets)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain()
        run_one_training(net_to_retrain)
    exit()
for n in range(opts.n_repeats):
    run_one_training()
