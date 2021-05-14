from batch_train_opts import options
from net_retrain_manager import NetRetrainManager
import os
import platform
import subprocess


# 2021-01-04
# python trainByBatchWin.py 5 "-d egg-combined -n FCRN_A  -lr 0.0045 -e 150 -hf 0.5 -vf 0.5 --plot --batch_size 4"
# robert@NZXT-U:/media/Synology3/Robert/objects_counting_dmap$ python3.7 trainByBatchLinux.py 5 "-d egg-combined -n FCRN_A  -lr 0.0045 -e 150 -hf 0.5 -vf 0.5 --plot --batch_size 4"
# 2021-01-05
# python trainByBatchWin.py 5 "-d egg -n FCRN_A -lr 0.003 -e 150 -rot -hf 0.5 -vf 0.5 --plot --batch_size 4"
# 2021-01-19
# (pytorchEnv) P:\Robert\objects_counting_dmap>python trainByBatchWin.py 5 "-d egg -n FCRN_A -lr 0.0015 -e 150 -hf 0.5 -vf 0.5 -rot --plot --batch_size 4"
# 2021-03-05
# python train.py -d egg-unshuffled -n FCRN_A -lr 0.004 -e 3000 -hf 0.5 -vf 0.5 -rot --plot --batch_size 4 --left_col_plots scatter --config P:\Robert\objects_counting_dmap\configs\shuffle_data_at_start_2021-02-23.json


if platform.node() == "Yang-Lab-Dell2":
    condaEnv = "detectronEnv"
    driveLetter = "R"
else:
    condaEnv = "pytorchEnv"
    driveLetter = "P"

opts = options()


def run_one_training(existing_model=None):
    command_being_called = (
        'start /w %%windir%%\\System32\\cmd.exe "/K" C:\\Users\\Tracking\\anaconda3\\Scripts\\activate.bat C:\\Users\\Tracking\\anaconda3 ^& conda activate %s ^& %s: ^& cd Robert\\objects_counting_dmap ^& cd ^& python train.py --export_at_end %s%s ^& exit'
        % (
            condaEnv,
            driveLetter,
            opts.trainParams,
            ""
            if existing_model is None
            else f' -m {os.path.join(opts.existing_nets, existing_model)}',
            # else f' -m "{os.path.join(opts.existing_nets, existing_model)}"',
        )
    )

    subprocess.call(
        command_being_called,
        shell=True,
    )


if opts.existing_nets:
    # check for the net "lock file" and create if not already present
    net_retrainer = NetRetrainManager(opts.existing_nets)
    while net_retrainer.nets_remaining_to_retrain():
        net_to_retrain = net_retrainer.get_random_net_to_retrain(debug=True)
        run_one_training(net_to_retrain)
    exit()
for n in range(opts.n_repeats):
    run_one_training()
    # subprocess.call('python train.py --export_at_end %s'%opts.trainParams)
    # subprocess.call('start %windir%\\System32\\cmd.exe \"/C\" C:\\Users\\Tracking' +\
    #   '\\anaconda3\\Scripts\\activate.bat C:\\Users\\Tracking\\anaconda3 ' +\
    #   '^& conda activate pytorchEnv ^& P: ^& cd Robert\\objects_counting_dmap' +\
    #   ' ^& python train.py --export_at_end %s'%opts.trainParams, shell=True)
