import numpy as np

# Simulation Parameters
radius = 250  # MD movement area radius in meters
num_mds = 10  # Number of mobile devices
num_slots = 3000  # Number of time slots
Nm, Ne = 4, 64  # Number of antennas at MD and ES
bandwidth = 1e6  # System bandwidth in Hz (1 MHz)
speed_range = (10, 20)  # MD speed range in m/s
freq_c = 2.4e9  # Carrier frequency in Hz
c = 3e8  # Speed of light in m/s
snr_values = [0, 2, 4, 6, 8, 10]  # List of SNR values (dB) to loop over

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)


def generate_positions(num_mds):
    """Generate initial random positions within a circular area."""
    angles = np.random.uniform(0, 2 * np.pi, num_mds)
    distances = np.sqrt(np.random.uniform(0, radius ** 2, num_mds))
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
positions = generate_positions(num_mds)
speeds = np.random.uniform(speed_range[0], speed_range[1], num_mds)  # Random speeds
directions = np.random.uniform(0, 2 * np.pi, num_mds)  # Random initial directions
noise_power = awgn_noise()  # Compute noise power

# Step 2: Generate Initial Channels
fading_memory = 0.9  # Memory factor for correlated small-scale fading
small_scale_fading_prev = np.zeros((num_mds, Ne, Nm), dtype=np.complex128)  # Store past fading


def generate_initial_channels(positions, user_snr_dB):
    """Generate initial channel matrices based on assigned SNRs."""
    user_snr_linear = 10 ** (user_snr_dB / 10)  # Convert dB to linear scale
    H_init = np.zeros((num_mds, Ne, Nm), dtype=np.complex128)

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


def update_channel(H_prev, user_snr_dB):
    """Update the channel with time-correlated fading and small SNR variations."""
    H_new = np.zeros((num_mds, Ne, Nm), dtype=np.complex128)
    user_snr_linear = 10 ** (user_snr_dB / 10)  # Convert dB to linear scale

    for i in range(num_mds):
        # Time-correlated small-scale fading
        new_fading = (np.random.randn(Ne, Nm) + 1j * np.random.randn(Ne, Nm)) / np.sqrt(2)
        small_scale_fading = fading_memory * small_scale_fading_prev[i] + (1 - fading_memory) * new_fading
        small_scale_fading_prev[i] = small_scale_fading  # Update stored fading

        # Scale previous channel matrix to match SNR
        P_signal = np.linalg.norm(H_prev[i], 'fro') ** 2
        P_target = user_snr_linear[i] * noise_power
        scaling_factor = np.sqrt(P_target / P_signal)

        H_new[i] = H_prev[i] * scaling_factor  # Adjusted channel

    return H_new


# Loop over different SNR values
for snr_value in snr_values:
    print(f"Simulating for SNR = {snr_value} dB")

    # Store channel data for each SNR value
    channel_data = np.zeros((num_slots, num_mds, Ne, Nm), dtype=np.complex128)
    snr_values_record = []  # Store all SNR values for verification

    # Assign fixed SNR for this loop
    user_snr_dB = np.full(num_mds, snr_value)  # Fixed SNR value for each user

    # Generate Initial Channel Data
    H_t = generate_initial_channels(positions, user_snr_dB)

    for t in range(num_slots):
        # Store current channel state
        channel_data[t] = H_t

        # Store SNR values
        computed_snr_linear = np.linalg.norm(H_t, axis=(1, 2)) ** 2 / noise_power
        computed_snr_dB = 10 * np.log10(computed_snr_linear)
        snr_values_record.extend(computed_snr_dB)

        # Update positions smoothly
        positions, directions = update_positions(positions, speeds, directions)

        # Update channel with small variation
        H_t = update_channel(H_t, user_snr_dB)

    # Save channel data for this specific SNR value as an NPZ file
    np.save(f"mimo_channel_data_snr_{snr_value}.npy", channel_data)
    print(f"MIMO channel data for SNR = {snr_value} saved as 'mimo_channel_data_snr_{snr_value}.npy'")

    # Verify SNR distribution
    snr_values_record = np.array(snr_values_record)
    print(f"SNR range: {snr_values_record.min():.2f} dB to {snr_values_record.max():.2f} dB")
    print(f"Mean SNR: {snr_values_record.mean():.2f} dB")
    print(f"Standard deviation of SNR: {snr_values_record.std():.2f} dB")
