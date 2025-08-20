import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from midi_utils.utils import image_to_patch_tokens

def json_to_pianoroll_88(json_file, length=80):
    """
    Convert a MuseCoco-style JSON prompt into an 88-key (pitch 20–108) binary pianoroll.

    Parameters
    ----------
    json_file : str
        Path to the JSON file whose top-level key is "prompt".
    length : int, optional
        Number of time steps to keep (default = 80).

    Returns
    -------
    np.ndarray
        Binary pianoroll of shape (88, length).
    """
    # Load JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    notes = data["prompt"]

    # Initialize 88 × length matrix (pitch range 20–108)
    pianoroll = np.zeros((88, length), dtype=np.int8)

    for note in notes:
        start = note["start"]
        duration = note["duration"]
        pitch = note["pitch"]

        if 20 <= pitch <= 108:  # Keep only notes within 88-key range
            row = pitch - 20    # pitch 20 → row 0, pitch 108 → row 88-1
            end = min(start + duration, length)  # Clamp to avoid overflow
            pianoroll[row, start:end] = 1

    return pianoroll


def load_json(json_file):
    """
    Convenience wrapper: JSON → pianoroll → patch tokens.
    """
    pianoroll_88 = json_to_pianoroll_88(json_file)
    patch_tokens = image_to_patch_tokens(torch.tensor(pianoroll_88, dtype=torch.uint8))
    return patch_tokens


if __name__ == '__main__':
    # Example usage
    pianoroll_88 = json_to_pianoroll_88("prompt.json")
    print("Conversion complete, pianoroll shape:", pianoroll_88.shape)

    # Save as npz (for testing)
    # np.savez("pianoroll_88.npz", combined_piano_roll=pianoroll_88)  # test only

    patch_tokens = image_to_patch_tokens(torch.tensor(pianoroll_88, dtype=torch.uint8))
    np.savez("pianoroll_44.npz", patch_roll=patch_tokens)
