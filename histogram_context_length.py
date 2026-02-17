import json
import matplotlib.pyplot as plt
import numpy as np

sequence_lengths = []
for file in [
    "../out/sequences/val_ids.json",
    "../out/sequences/ids.json",
    "../out/sequences/test_ids.json"
]:
    with open(file) as f:
        sequences = json.load(f)
        sequence_lengths.extend(
            [len(sequence[0]) + len(sequence[1]) for sequence in sequences])

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.hist(sequence_lengths, bins=100)
ax1.set_yscale("log")
ax1.set_title("Sequence length histogram")
ax1.set_xlabel("Sequence length of training data")
ax1.set_ylabel("Frequency")

log_bins = np.logspace(np.log10(min(sequence_lengths)), np.log10(max(sequence_lengths)), num=100)
ax2.hist(sequence_lengths, bins=log_bins)
ax2.set_xscale("log")
ax2.set_title("Sequence length histogram")
ax2.set_xlabel("Sequence length of training data")
ax2.set_ylabel("Frequency")

plt.show()
