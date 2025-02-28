import os

import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from fractions import Fraction

def initialize_system_parameters():
    # Models with backbone, quantization method, and quantization channel
    models = [
        {'name': 'model_1', 'backbone': 'MaskRCNN', 'quant_method': 'Box', 'quant_channel': 3},
        {'name': 'model_2', 'backbone': 'MaskRCNN', 'quant_method': 'Box', 'quant_channel': 6},
        {'name': 'model_3', 'backbone': 'MaskRCNN', 'quant_method': 'Box', 'quant_channel': 12},
        {'name': 'model_4', 'backbone': 'FRCNN', 'quant_method': 'Box', 'quant_channel': 3},
        {'name': 'model_5', 'backbone': 'FRCNN', 'quant_method': 'Box', 'quant_channel': 6},
        {'name': 'model_6', 'backbone': 'FRCNN', 'quant_method': 'Box', 'quant_channel': 12},
        {'name': 'model_7', 'backbone': 'MaskRCNN', 'quant_method': 'Entropic', 'quant_channel': 3},
        {'name': 'model_8', 'backbone': 'MaskRCNN', 'quant_method': 'Entropic', 'quant_channel': 6},
        {'name': 'model_9', 'backbone': 'MaskRCNN', 'quant_method': 'Entropic', 'quant_channel': 12},
        {'name': 'model_10', 'backbone': 'FRCNN', 'quant_method': 'Entropic', 'quant_channel': 3},
        {'name': 'model_11', 'backbone': 'FRCNN', 'quant_method': 'Entropic', 'quant_channel': 6},
        {'name': 'model_12', 'backbone': 'FRCNN', 'quant_method': 'Entropic', 'quant_channel': 12},
        {'name': 'model_13', 'backbone': 'MaskRCNN', 'quant_method': 'standard', 'quant_channel': 3},
        {'name': 'model_14', 'backbone': 'MaskRCNN', 'quant_method': 'standard', 'quant_channel': 6},
        {'name': 'model_15', 'backbone': 'MaskRCNN', 'quant_method': 'standard', 'quant_channel': 12},
        {'name': 'model_16', 'backbone': 'FRCNN', 'quant_method': 'standard', 'quant_channel': 3},
        {'name': 'model_17', 'backbone': 'FRCNN', 'quant_method': 'standard', 'quant_channel': 6},
        {'name': 'model_18', 'backbone': 'FRCNN', 'quant_method': 'standard', 'quant_channel': 12},
    ]

    # FLOPs for head and tail models
    head_flops = {model['name']: np.random.randint(1e8, 1e9) for model in models}
    tail_flops = {model['name']: np.random.randint(1e8, 1e9) for model in models}

    # Data size in bits for head model output
    data_size = {model['name']: np.random.randint(150e3, 300e3) for model in models}  # in bits

    # Device parameters

    # NVIDIA Jetson Nano (Mobile Device)
    md_freq = 921.6e6  # GPU frequency in Hz
    md_cores = 128  # Number of CUDA cores
    md_flops_per_cycle = 2  # FLOPs per cycle per core
    md_power_coeff = 0.2  # Power consumption coefficient

    # NVIDIA Jetson Orin NX (Edge Server)
    es_freq = 1.5e9  # GPU frequency in Hz
    es_cores = 2048  # Number of CUDA cores
    es_flops_per_cycle = 2  # FLOPs per cycle per core
    es_power_coeff = 0.7  # Power consumption coefficient

    # Users
    users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
    user_snr = {'user_1': 0, 'user_2': 1, 'user_3': 2, 'user_4': 3, 'user_5': 4}  # SNR in dB
    user_delay_constraint = {user: np.random.uniform(0.005, 0.01) for user in users}  # Delay constraint in seconds

    # Bandwidth resource blocks
    total_bandwidth = 20  # Total bandwidth in resource blocks (in MHz)

    # Modulation and channel estimation error
    modulation = 'bpsk'  # Modulation scheme (can be dynamically changed)
    esterr = 0.5  # Channel estimation error (can be dynamically changed)

    return {
        'models': models,
        'head_flops': head_flops,
        'tail_flops': tail_flops,
        'data_size': data_size,
        'md_freq': md_freq,
        'md_cores': md_cores,
        'md_flops_per_cycle': md_flops_per_cycle,
        'md_power_coeff': md_power_coeff,
        'es_freq': es_freq,
        'es_cores': es_cores,
        'es_flops_per_cycle': es_flops_per_cycle,
        'es_power_coeff': es_power_coeff,
        'users': users,
        'user_snr': user_snr,
        'user_delay_constraint': user_delay_constraint,
        'total_bandwidth': total_bandwidth,
        'modulation': modulation,  # Added modulation
        'esterr': esterr  # Added esterr
    }


def load_ldpc_ber_data(snr, rate, modulation='64qam', esterr=0.5):
    """
    Load LDPC BER data from a .mat file based on SNR, rate, modulation, and esterr.
    Perform curve fitting to get BER for the given SNR.

    Parameters:
        snr (float): SNR value in dB.
        rate (fraction): Code rate (e.g., 1/2, 2/3, 3/4, 5/6).
        modulation (str): Modulation scheme (e.g., 'bpsk', '64qam').
        esterr (float): Channel estimation error (e.g., 0.5).

    Returns:
        float: BER value for the given SNR.
    """

    # Define the directory where the .mat file should be stored
    save_dir = "sys_data/trans_sys_sim/ldpc_mimo_data/"

    # Generate the filename based on SNR, rate, modulation, and estimation error
    filename = f"snr_0_0.5_10_{modulation}_esterr_{esterr}_rate_{rate.numerator}_{rate.denominator}.mat"

    # Concatenate the directory and filename to get the full file path
    full_path = os.path.join(save_dir, filename)

    try:
        # Load the .mat file
        data = scipy.io.loadmat(full_path)

        # Extract SNR and BER values
        snr_values = data['ebno_db_vec'].flatten()  # SNR values from the file
        ber_values = data['ber'].flatten()  # BER values from the file

        # Perform curve fitting using interpolation
        ber_interp = interp1d(snr_values, ber_values, kind='linear', fill_value="extrapolate")

        # Get BER for the given SNR
        ber = ber_interp(snr)
        return ber
    except FileNotFoundError:
        print(f"File {filename} not found. Please check the parameters.")
        return None
    except KeyError:
        print(f"File {filename} does not contain the expected variables ('ebno_db_vec' and 'bler').")
        return None

# Calculate processing delay
def calculate_processing_delay(params):
    models = params['models']
    head_flops = params['head_flops']
    tail_flops = params['tail_flops']
    md_freq = params['md_freq']
    md_cores = params['md_cores']
    md_flops_per_cycle = params['md_flops_per_cycle']
    es_freq = params['es_freq']
    es_cores = params['es_cores']
    es_flops_per_cycle = params['es_flops_per_cycle']

    head_delay = {}
    tail_delay = {}

    for model in models:
        model_name = model['name']
        head_delay[model_name] = head_flops[model_name] / (md_freq * md_cores * md_flops_per_cycle)
        tail_delay[model_name] = tail_flops[model_name] / (es_freq * es_cores * es_flops_per_cycle)

    return head_delay, tail_delay




def fair_bandwidth_allocation(params):
    """
    Allocate bandwidth continuously to ensure fair transmission rates among users.

    Parameters:
        params (dict):
            - 'users' (list): List of user identifiers.
            - 'total_bandwidth' (float): Total available bandwidth.
            - 'user_snr' (dict): Signal-to-noise ratio (SNR) for each user.

    Returns:
        dict: A dictionary mapping each user to an allocated bandwidth.
    """
    users = params['users']
    total_bandwidth = params['total_bandwidth']
    user_snr = params['user_snr']
    num_users = len(users)

    def get_rate(bandwidth, snr):
        """Compute the transmission rate using Shannon capacity formula."""
        return bandwidth * np.log2(1 + 10 ** (snr / 10))

    # Step 1: Initial bandwidth allocation based on SNR weights
    weights = np.array([1 / np.log2(1 + 10 ** (user_snr[user] / 10)) for user in users])
    bandwidth_allocation = {user: total_bandwidth * (weights[i] / sum(weights)) for i, user in enumerate(users)}

    # Step 2: Iteratively adjust bandwidth to minimize rate variance
    learning_rate = 0.01  # Step size for adjustment
    max_iterations = 500
    prev_variance = float('inf')

    for _ in range(max_iterations):
        rates = {user: get_rate(bandwidth_allocation[user], user_snr[user]) for user in users}
        rate_values = np.array(list(rates.values()))
        rate_variance = np.var(rate_values)

        if rate_variance >= prev_variance:
            break  # Stop if variance does not decrease

        prev_variance = rate_variance

        avg_rate = np.mean(rate_values)
        for user in users:
            if rates[user] > avg_rate:
                bandwidth_allocation[user] -= learning_rate  # Decrease bandwidth for higher-rate users
            else:
                bandwidth_allocation[user] += learning_rate  # Increase bandwidth for lower-rate users

        # Normalize bandwidth allocation to maintain total bandwidth constraint
        total_allocated = sum(bandwidth_allocation.values())
        bandwidth_allocation = {user: (bw / total_allocated) * total_bandwidth for user, bw in bandwidth_allocation.items()}

    return bandwidth_allocation



# Main simulation loop
def model_delay_simulation():
    params = initialize_system_parameters()
    head_delay, tail_delay = calculate_processing_delay(params)

    # Available code rates (only 1/2, 2/3, 3/4, 5/6 are allowed)
    allowed_rates = [Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(5, 6)]

    # Simulation loop
    for model in params['models']:
        model_name = model['name']
        print(f"Model: {model_name} (Backbone: {model['backbone']}, Quant Method: {model['quant_method']}, Quant Channel: {model['quant_channel']})")

        # Bandwidth allocation (fair allocation)
        bandwidth_allocation = fair_bandwidth_allocation(params)

        for user in params['users']:
            snr = params['user_snr'][user]
            delay_constraint = params['user_delay_constraint'][user]

            # Calculate allowed transmission delay
            total_delay = head_delay[model_name] + tail_delay[model_name]
            allowed_transmission_delay = delay_constraint - total_delay

            if allowed_transmission_delay < 0:
                print(f"User {user} cannot meet delay constraint with model {model_name}")
                continue

            bandwidth =  bandwidth_allocation[user] # (in Mhz)

            # Calculate transmission rate (assuming Shannon capacity)
            snr_linear = 10 ** (snr / 10)
            transmission_rate = bandwidth * 10e6 * np.log2(1 + snr_linear)

            # Calculate allowed data size
            allowed_data_size = transmission_rate * allowed_transmission_delay

            # Select LDPC code rate based on allowed data size (only 1/2, 2/3, 3/4, 5/6 are allowed)
            data_size = params['data_size'][model_name]
            valid_rates = []
            for rates in allowed_rates:
                if data_size / rates <= allowed_data_size:
                    valid_rates.append(rates)
            # If valid_rates is not empty, choose the minimum valid rate, otherwise choose the maximum from allowed_rates
            selected_rate = min(valid_rates) if valid_rates else max(allowed_rates)

            # Load LDPC BER data and get BER for the selected SNR, rate, modulation, and esterr
            modulation = params['modulation']  # Get modulation from params
            esterr = params['esterr']  # Get esterr from params
            ber = load_ldpc_ber_data(snr, selected_rate, modulation=modulation, esterr=esterr)

            if ber is not None:
                print(f"{user} with SNR {snr} dB: Delay Constraint {delay_constraint},  Head Delay {head_delay[model_name]}, Tail Delay {tail_delay[model_name]}, Allocated Bandwidth {bandwidth} Trans Delay {allowed_transmission_delay}, "
                      f"Transmit Data Size {params['data_size'][model_name]}, Allowed Data Size {allowed_data_size}, Selected Code Rate {selected_rate}, BER {ber}")
            else:
                print(f"{user} with SNR {snr} dB: Failed to load BER data.")




if __name__ == "__main__":
    model_delay_simulation()