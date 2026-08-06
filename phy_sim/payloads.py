"""Branch payload sizes, mirroring sys_data/config.py (single source: Config.data_size).

Payload = quantized intermediate-feature tensor of one inference task, in bytes.
One task = one transport block in the link-level simulation.
"""

# Keep in sync with sys_data/config.py Config.data_size (bytes)
PAYLOAD_BYTES = {
    'Box3': 6000,
    'Box6': 13260,
    'Box12': 33580,
    'Standard3': 11230,
    'Standard6': 22460,
    'Standard12': 44930,
}

BRANCHES = list(PAYLOAD_BYTES.keys())

PAYLOAD_BITS = {b: 8 * n for b, n in PAYLOAD_BYTES.items()}


def payload_bytes(branch):
    return PAYLOAD_BYTES[branch]


def payload_bits(branch):
    return PAYLOAD_BITS[branch]
