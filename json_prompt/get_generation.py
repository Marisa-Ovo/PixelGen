import numpy as np
import json

def get_json(pianoroll):
    notes = []

    # Iterate through all pitches (MIDI 0–127)
    for pitch in range(pianoroll.shape[0]):
        active = pianoroll[pitch]
        start = None
        for t in range(len(active)):
            if active[t] == 1 and start is None:
                # Note-on detected
                start = t
            elif (active[t] == 0 or t == len(active) - 1) and start is not None:
                # Note-off detected
                end = t if active[t] == 0 else t + 1
                duration = end - start
                notes.append({
                    "start": start + 80,
                    "pitch": pitch,
                    "duration": duration
                })
                start = None

    # Pack into JSON format
    json_data = {"generation": notes}

    return json_data

if __name__ == '__main__':
    data = np.load("./gen1.npz")
    pianoroll = data["combined_piano_roll"]
    pianoroll = pianoroll[:, 80:272] 

    json_data = get_json(pianoroll)

    # Save to JSON file
    with open("gen1.json", "w") as f:
        json.dump(json_data, f, indent=2)