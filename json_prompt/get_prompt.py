import numpy as np
import json

# Load the .npz file
data = np.load("./example3.npz")
pianoroll = data["combined_piano_roll"]  # Shape: [pitch, time]
# Keep only the first 80 time steps
pianoroll = pianoroll[:, :80]

notes = []

# Iterate over every MIDI pitch (0–127)
for pitch in range(pianoroll.shape[0]):
    active = pianoroll[pitch]
    start = None
    for t in range(len(active)):
        if active[t] == 1 and start is None:
            # Note-on detected
            start = t
        elif (active[t] == 0 or t == len(active) - 1) and start is not None:
            # Note-off detected
            end = t if active[t] == 0 else t + 1  # Include final frame if still active
            duration = end - start
            notes.append({
                "start": start,
                "pitch": pitch,
                "duration": duration
            })
            start = None

# Pack data into JSON format
json_data = {"prompt": notes}

# Save to JSON file
with open("prompt3.json", "w") as f:
    json.dump(json_data, f, indent=2)

print("Conversion complete, saved as prompt.json")
