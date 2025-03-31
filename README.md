# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning (ML) applications via task offloading. To this aim,  radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slice approaches rely on static computing models, thus limiting the ability of the resulting frameworks to take advantage of important dynamic semantic data representation opportunities granted by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework that optimizes bandwidth and computing resource allocation by leveraging a new generation of  dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding, which provides an ample set of communication payload and computing options to the network controller. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and forward error correction (FEC) in determining the performance of computer vision tasks. In this context, we design a new quantization approach, which we refer to as ``box'' quantization, that improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate a slicing problem where the mobile devices (MD)  maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split neural networks, and the edge server (ES) allocates resources to maximize the worst inference accuracy among all MDs. To solve this problem, we propose a multi-agent distributed optimization framework, where the MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

## Structure
```bash
OMNIS/
│── baselines/             # Implementations of baselines
│   ├── cto_main.py            # Script to run CTO methods
│   ├── dts_main.py            # Script to run OMNIS-TS methods
│   ├── gdo_main.py            # Script to run GDO methods
│   └── rss_main.py            # Script to run RSS methods
│
│── experiemnts/           # Channel generation scripts
│   ├── results/        # Script to generate communication channel conditions
│   ├── eval_action.py       # Evaluate the action pick probability under different parameters
│   ├── eval_convergence.py      # Evaluate the instant performance in each time slot
│   ├── eval_snr.py        # Evalaute the performance under different SNR values
│   ├── eval_user_num.py     # Evalaute the performance under different user numbers
│   ├── convergence_figure_gen.m        # Script to generate Fig. 5
│   ├── diff_snr_figure_gen.m        # Script to generate Fig. 6
│   ├── diff_user_num_figure_gen.m        # Script to generate Fig. 7
│   ├── action_stat.m        # Script to generate Fig. 8
│   └── gain_calculation.m        # Script to calculate the performance gain
│
│── observations/           # Scripts for observation experiment visualization
│   ├── acc_payload_fig_1_gen.m            # Script to generate Fig. 2
│   └── acc_payload_fig_2_gen.m     # Script to generate Fig. 4 with box quantization
│
│── omnis/               # Core OMNIS framework
│   ├── omnis_main.py    # Script to run OMNIS algorithm
│   ├── action_space.py  # Define the action-context space of the MAB agent
│   ├── cbo.py           # Define UCB and TS method
│   └── util.py          # OMNIS utility functions
│
│── sys_data/           
│   ├── acc_data/          # accuracy data of the proposed DNN model
│       ├── acc_fit.py     # To fit the discrete SNR-accuracy pair
│   ├── mimo_channel_gen/
│       ├── mimo_channel_gen.py  # The SNRs of each user are randomly distributed
│                                # within a range, and will output one .npy file 
│       └── mimo_channel_gen_fix_snr.py   # The SNR value of each user is fixed to a specific value
|                                         #  to compare the impact of different SNRs on performance, 
|                                         #  and each SNR setting will output one .npy file
│   └── config.py     # Configuration settings file
```



## How to Try OMNIS


1. **Generate the communication channels (no specific execution order):**
   ```bash
   python3 sys_data/mimo_channel_gen/mimo_channel_gen.py  
   python3 sys_data/mimo_channel_gen/mimo_channel_gen_fix_snr.py  
   ```
2. **Run the OMNIS and other baseline methods with fixed parameters (no specific execution order):**
   ```bash
   python3 omnis/omnis_main.py
   python3 baselines/cto_main.py
   python3 baselines/dts_main.py 
   python3 baselines/gdo_main.py
   python3 baselines/rss_main.py
   ```
3. **Execute large-scale comparison experiments with varying parameters (no specific execution order):**
   ```bash
   python3 experiments/eval_convergence.py 
   python3 experiments/eval_user_num.py   
   python3 experiments/eval_snr.py  
   python3 experiments/eval_action.py  
   ```


3. **Generate observation / evaluation figures by running the corresponding MATLAB scripts in `observations` / ``experiments`` directory.**


## Important Notes
1. In current version, Pycharm or other IDE is recommended, running it directly via VSCode will result in an error because VScode treats the importted folder as a library.

2. Ensure that the working directory is set to the root of the repository when running any Python files.

3. Make sure that the number of time slots and users in ``mimo_channel_gen.py`` / ``mimo_channel_gen_fix_snr.py`` is greater than or equal to the corresponding values in `config.py` to support simulation

4. The system configurations and hyperparameters are managed via the ``config.py`` file. Modify it to adjust simulation settings before running experiments. When running `eval_user_num.py`, make sure the maximum evaluation user number in the variable `user_num_list` of `eval_user_num.py` less or equal to the variable `self.user_num` of `config.py`.

5. This repository does not include the training and evaluation code of the multi-branch dynamic split neural network. It only provides an interface between the neural network evaluation data and the optimization framework. If you have any questions regarding neural network details, please contact Ian Andrew Harshbarger (iharshba@uci.edu). 



## Contributing
We welcome contributions to improve OMNIS. To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Open a pull request detailing the changes.

---

## Acknowledgments


The multi-agent MAB optimization scheme is developed based on https://github.com/jaayala/contextual_bayesian_optimization
