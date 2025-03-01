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
target_snr_db = (0, 10)  # Target SNR range in dB

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


def update_positions(positions, speeds, directions, dt=1.0):
    """Update MD positions based on random mobility model."""
    dx = speeds * np.cos(directions) * dt
    dy = speeds * np.sin(directions) * dt
    new_positions = positions + np.stack((dx, dy), axis=-1)
    return np.clip(new_positions, -radius, radius)  # Keep within bounds


def path_loss(d):
    """Compute path loss using the log-distance model."""
    return 128.1 + 37.6 * np.log10(d / 1000)  # d in km, output in dB


def awgn_noise():
    """Generate AWGN noise power in linear scale."""
    noise_power_dBm = -174 + 10 * np.log10(bandwidth)  # Noise power in dBm
    noise_power_linear = 10 ** (noise_power_dBm / 10 - 3)  # Convert to Watts
    return noise_power_linear


def generate_channel(positions, noise_power):
    """Generate MIMO channel matrix with controlled SNR."""
    H = np.zeros((num_mds, Ne, Nm), dtype=np.complex128)
    snr_list = []  # Store computed SNR values

    for i, pos in enumerate(positions):
        d = np.linalg.norm(pos) + 1e-3  # Avoid division by zero
        path_loss_dB = path_loss(d)
        path_loss_linear = 10 ** (-path_loss_dB / 20)  # Convert dB to linear scale

        # Small-scale fading (Rayleigh)
        small_scale_fading = (np.random.randn(Ne, Nm) + 1j * np.random.randn(Ne, Nm)) / np.sqrt(2)
        H_raw = path_loss_linear * small_scale_fading

        # Compute raw signal power
        P_signal = np.linalg.norm(H_raw, 'fro') ** 2

        # Randomly select a target SNR in dB within the range [0, 10]
        target_snr_dB = np.random.uniform(*target_snr_db)
        target_snr_linear = 10 ** (target_snr_dB / 10)

        # Compute target signal power
        P_target = target_snr_linear * noise_power

        # Scale H to match the desired SNR
        scaling_factor = np.sqrt(P_target / P_signal)
        H[i] = H_raw * scaling_factor

        # Verify SNR after scaling
        computed_snr_linear = np.linalg.norm(H[i], 'fro') ** 2 / noise_power
        computed_snr_dB = 10 * np.log10(computed_snr_linear)
        snr_list.append(computed_snr_dB)

    return H, np.array(snr_list)


# Initialize positions and movement parameters
positions = generate_positions(num_mds)
speeds = np.random.uniform(speed_range[0], speed_range[1], num_mds)  # Random speeds
directions = np.random.uniform(0, 2 * np.pi, num_mds)  # Random initial directions
noise_power = awgn_noise()  # Compute noise power

# Generate and store channel data
channel_data = np.zeros((num_slots, num_mds, Ne, Nm), dtype=np.complex128)
snr_values = []  # Store all SNR values for verification

for t in range(num_slots):
    H_t, snr_t = generate_channel(positions, noise_power)
    channel_data[t] = H_t
    snr_values.extend(snr_t)  # Collect SNR values

    positions = update_positions(positions, speeds, directions)  # Update MD positions
    directions = np.random.uniform(0, 2 * np.pi, num_mds)  # Randomly change direction

# Save channel data
np.save("mimo_channel_data.npy", channel_data)
print("MIMO channel data saved as 'mimo_channel_data.npy'")

# Verify SNR distribution
snr_values = np.array(snr_values)
print(f"SNR range: {snr_values.min():.2f} dB to {snr_values.max():.2f} dB")
print(f"Mean SNR: {snr_values.mean():.2f} dB")
print(f"Standard deviation of SNR: {snr_values.std():.2f} dB")
