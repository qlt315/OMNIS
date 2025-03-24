import numpy as np

# Simulation Parameters

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)


radius = 250  # MD movement area radius in meters
user_num = 10  # Number of mobile devices
time_slot_num = 300  # Number of time slots
Nm, Ne = 4, 64  # Number of antennas at MD and ES
bandwidth = 1e6  # System bandwidth in Hz (1 MHz)
speed_range = (10, 20)  # MD speed range in m/s
freq_c = 2.4e9  # Carrier frequency in Hz
c = 3e8  # Speed of light in m/s
snr_range = (0, 10)  # Fixed SNR range per user




def generate_positions(user_num):
    """Generate initial random positions within a circular area."""
    angles = np.random.uniform(0, 2 * np.pi, user_num)
    distances = np.sqrt(np.random.uniform(0, radius ** 2, user_num))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return np.stack((x, y), axis=-1)


def update_positions(positions, speeds, directions, dt=0.1, inertia=0.9):
    """Update MD positions with smooth direction changes."""
    direction_noise = np.random.uniform(-np.pi / 8, np.pi / 8, len(directions))
    new_directions = inertia * directions + (1 - inertia) * (directions + direction_noise)

    dx = speeds * np.cos(new_directions) * dt
    dy = speeds * np.sin(new_directions) * dt
    new_positions = positions + np.stack((dx, dy), axis=-1)

    return np.clip(new_positions, -radius, radius), new_directions


def path_loss(d):
    """Compute path loss using the log-distance model."""
    return 128.1 + 37.6 * np.log10(d / 1000)  # d in km, output in dB


def awgn_noise():
    """Generate AWGN noise power in linear scale."""
    noise_power_dBm = -174 + 10 * np.log10(bandwidth)  # Noise power in dBm
    noise_power_linear = 10 ** (noise_power_dBm / 10 - 3)  # Convert to Watts
    return noise_power_linear


# Initialize positions and movement parameters
positions = generate_positions(user_num)
speeds = np.random.uniform(speed_range[0], speed_range[1], user_num)  # Random speeds
directions = np.random.uniform(0, 2 * np.pi, user_num)  # Random initial directions
noise_power = awgn_noise()  # Compute noise power

# Step 1: Assign initial SNR to each user
user_snr_dB = np.random.uniform(*snr_range, user_num)  # Fixed per user
user_snr_linear = 10 ** (user_snr_dB / 10)  # Convert dB to linear scale

# Step 2: Generate Initial Channels
fading_memory = 0.9  # Memory factor for correlated small-scale fading
small_scale_fading_prev = np.zeros((user_num, Ne, Nm), dtype=np.complex128)  # Store past fading

def generate_initial_channels(positions):
    """Generate initial channel matrices based on assigned SNRs."""
    H_init = np.zeros((user_num, Ne, Nm), dtype=np.complex128)

    for i, pos in enumerate(positions):
        d = np.linalg.norm(pos) + 1e-3  # Avoid division by zero
        path_loss_dB = path_loss(d)
        path_loss_linear = 10 ** (-path_loss_dB / 20)

        # Small-scale fading (Rayleigh)
        small_scale_fading = (np.random.randn(Ne, Nm) + 1j * np.random.randn(Ne, Nm)) / np.sqrt(2)
        small_scale_fading_prev[i] = small_scale_fading  # Store for future updates

        H_raw = path_loss_linear * small_scale_fading

        # Compute required power scaling to match SNR
        P_signal = np.linalg.norm(H_raw, 'fro') ** 2
        P_target = user_snr_linear[i] * noise_power
        scaling_factor = np.sqrt(P_target / P_signal)

        H_init[i] = H_raw * scaling_factor

    return H_init


def update_channel(H_prev):
    """Update the channel with time-correlated fading and small SNR variations."""
    H_new = np.zeros((user_num, Ne, Nm), dtype=np.complex128)

    for i in range(user_num):
        # Time-correlated small-scale fading
        new_fading = (np.random.randn(Ne, Nm) + 1j * np.random.randn(Ne, Nm)) / np.sqrt(2)
        small_scale_fading = fading_memory * small_scale_fading_prev[i] + (1 - fading_memory) * new_fading
        small_scale_fading_prev[i] = small_scale_fading  # Update stored fading

        # Adjust small SNR variation (±0.5 dB change)
        snr_variation_dB = np.random.uniform(-0.5, 0.5)  # Small random SNR change
        new_snr_dB = np.clip(user_snr_dB[i] + snr_variation_dB, *snr_range)
        new_snr_linear = 10 ** (new_snr_dB / 10)

        # Scale previous channel matrix to match new SNR
        P_signal = np.linalg.norm(H_prev[i], 'fro') ** 2
        P_target = new_snr_linear * noise_power
        scaling_factor = np.sqrt(P_target / P_signal)

        H_new[i] = H_prev[i] * scaling_factor  # Adjusted channel

    return H_new


# Generate Initial Channel Data
H_t = generate_initial_channels(positions)

# Store channel data
channel_data = np.zeros((time_slot_num, user_num, Ne, Nm), dtype=np.complex128)
snr_values = []  # Store all SNR values for verification

for t in range(time_slot_num):
    # Store current channel state
    channel_data[t] = H_t

    # Store SNR values
    computed_snr_linear = np.linalg.norm(H_t, axis=(1, 2)) ** 2 / noise_power
    computed_snr_dB = 10 * np.log10(computed_snr_linear)
    snr_values.extend(computed_snr_dB)

    # Update positions smoothly
    positions, directions = update_positions(positions, speeds, directions)

    # Update channel with small variation
    H_t = update_channel(H_t)

# Save channel data
np.save("sys_data/trans_sys_sim/mimo_channel_gen/mimo_channel_data.npy", channel_data)
print("MIMO channel data saved as 'mimo_channel_data.npy'")

# Verify SNR distribution
snr_values = np.array(snr_values)
print(f"SNR range: {snr_values.min():.2f} dB to {snr_values.max():.2f} dB")
print(f"Mean SNR: {snr_values.mean():.2f} dB")
print(f"Standard deviation of SNR: {snr_values.std():.2f} dB")
