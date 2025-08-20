import numpy as np
from PIL import Image
import pypianoroll
import torch


def ch1tensor256_save(ch1tensor, img_save_path="temp_ch1.png"):
    # Convert torch.Tensor to numpy array if needed
    if isinstance(ch1tensor, torch.Tensor):
        ch1tensor = ch1tensor.cpu().numpy()

    # Ensure values are in range [0,255] and dtype is uint8
    ch1tensor = np.clip(ch1tensor * 2, 0, 255).astype(np.uint8)

    # Save as grayscale image
    img = Image.fromarray(ch1tensor).convert('L')  # 'L' = grayscale
    img.show()
    img.save(img_save_path)


# torch
def pianoroll_to_multitrack(pianoroll, velocity=64, tempo=60):
        """
        Convert a binary pianoroll tensor [128, T] into a pypianoroll.Multitrack object.

        Args:
            pianoroll (torch.Tensor): Binary tensor where axis 0 = MIDI pitch (0-127),
                                      axis 1 = time steps.
            velocity (int): MIDI velocity (1-127) to assign to active notes.
            tempo (int): Constant BPM value for every time step in the output.

        Returns:
            pypianoroll.Multitrack: One-track multitrack (“Piano”) with fixed tempo.
        """
        # Move data to CPU and convert to NumPy
        pianoroll = pianoroll.cpu().numpy()

        tracks = []

        # Binarize and scale by velocity
        pianoroll = (pianoroll > 0).astype(int) * velocity
        # Transpose: pypianoroll expects shape [T, 128]
        pianoroll = pianoroll.T

        piano_track = pypianoroll.StandardTrack(
            pianoroll=pianoroll,
            program=0,      # 0 = Acoustic Grand Piano
            is_drum=False,
            name="Piano"
        )
        tracks.append(piano_track)

        steps = pianoroll.shape[0]
        # Create a constant-tempo array, one value per time step
        tempo = np.full((steps,), tempo)

        # Assemble the Multitrack object
        multitrack = pypianoroll.Multitrack(
            tracks=tracks,
            tempo=tempo     # constant BPM for the whole sequence
        )
        return multitrack


def image_to_patch_tokens(image, H=2, W=4):
    """
    Convert a binary image to patch-level tokens.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Binary image where 1 = note-on, 0 = silence.
        Shape = [img_h, img_w].
    H : int, optional
        Patch height (default = 2).
    W : int, optional
        Patch width  (default = 4).

    Returns
    -------
    np.ndarray
        Token matrix of shape (num_patch_cols, num_patch_rows).
        Each token is an integer in [0, 2**(H*W) - 1].
    """

    # If input is a torch.Tensor, move to CPU and convert to NumPy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    patch_h, patch_w = H, W
    img_h, img_w = image.shape[0], image.shape[1]

    # Pad width so that img_w is divisible by patch_w
    padding_w = (patch_w - img_w % patch_w) % patch_w
    if padding_w > 0:
        # Pad on the right with zeros (silence)
        image = np.pad(image, ((0, 0), (0, padding_w)),
                       mode='constant', constant_values=0)

    img_w = image.shape[1]  # Update width after padding

    num_patch_rows = img_h // patch_h  # e.g., 44 when img_h=88 and H=2
    num_patch_cols = img_w // patch_w  # Number of patches along width

    # Initialize token matrix: shape (columns, rows)
    tokens = np.zeros((num_patch_cols, num_patch_rows), dtype=np.int64)

    # Iterate columns (left → right), then rows (top → bottom)
    for j in range(num_patch_cols):
        for i in range(num_patch_rows):
            # Extract patch of size H×W
            patch = image[i * patch_h:(i + 1) * patch_h,
                          j * patch_w:(j + 1) * patch_w]
            # Flatten to 1-D array of length H*W (row-major order)
            patch_flat = patch.flatten()

            # Encode binary patch to integer token
            token = 0
            for bit in patch_flat:
                token = (token << 1) | int(bit)
            tokens[j, i] = token  # Column major storage

    return tokens

def patch_tokens_to_image(tokens):
    """
    Reconstruct the original binary image from token indices,
    following the same layout as produced by image_to_patch_tokens.

    Args:
        tokens (Tensor or np.ndarray): Token index matrix of shape [x, 44],
            where each value is in the range 0–255.

    Returns:
        torch.Tensor: Reconstructed binary image of shape [88, 4*x]
            (values 0/1).
    """
    # Convert to NumPy if input is a torch.Tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()

    num_patch_cols, num_patch_rows = tokens.shape
    patch_h, patch_w = 2, 4  # Patch size = 2×4
    img_h, img_w = patch_h * num_patch_rows, patch_w * num_patch_cols  # Output image size

    # Initialize the output image
    image = np.zeros((img_h, img_w), dtype=np.uint8)

    # Loop over columns (left → right) and rows (top → bottom)
    for j in range(num_patch_cols):
        for i in range(num_patch_rows):
            token = tokens[j, i]
            if token > 255:
                token = 1  # Safety clamp

            # Convert token to 8-bit binary array
            patch_flat = np.array([int(x) for x in format(token, '08b')], dtype=np.uint8)

            # Reshape to 2×4 patch
            patch = patch_flat.reshape((patch_h, patch_w))

            # Place patch back into the correct position
            image[i * patch_h:(i + 1) * patch_h,
                  j * patch_w:(j + 1) * patch_w] = patch

    # Convert to torch.Tensor and return
    image_tensor = torch.tensor(image)
    return image_tensor


def split_tensor_on_condition(gen: torch.Tensor, target_value: int = 256):
    """
    Split a [N, 44] tensor into two parts at the first all-`target_value` row.

    Parameters
    ----------
    gen : Tensor
        Input tensor of shape [N, 44].
    target_value : int, default 256
        Row delimiter value.

    Returns
    -------
    (Tensor | None, Tensor | None)
        short_gen : rows before the delimiter
        full_gen  : rows starting from the delimiter
        If no delimiter row is found, both returns are None.
    """
    delimiter_idx = None
    for idx in range(gen.shape[0]):
        if torch.all(gen[idx] == target_value):
            delimiter_idx = idx
            break

    if delimiter_idx is not None:
        return gen[:delimiter_idx], gen[delimiter_idx:]
    else:
        print("No delimiter row (all 256) found.")
        return None, None

def remove_all_256_columns(full_gen: torch.Tensor, target_value: int = 256):
    """
    Remove every row that consists entirely of `target_value`.

    Parameters
    ----------
    full_gen : Tensor  [N, 44]
    target_value : int, default 256

    Returns
    -------
    Tensor
        Filtered tensor with padding rows removed.
    """
    mask = torch.all(full_gen == target_value, dim=1)
    return full_gen[~mask, :]


def map_div_tokens(ground_indices):
    pianoroll = patch_tokens_to_image(ground_indices)
    newroll = pianoroll[:,::8]
    div_tokens = image_to_patch_tokens(newroll)
    return div_tokens

def split_tensor_on_condition(gen: torch.Tensor, target_value: int = 256):
    """
    Split a [N, 44] tensor into two parts at the first all-`target_value` row.

    Parameters
    ----------
    gen : Tensor
        Input tensor of shape [N, 44].
    target_value : int, default 256
        Row delimiter value.

    Returns
    -------
    (Tensor | None, Tensor | None)
        short_gen : rows before the delimiter
        full_gen  : rows starting from the delimiter
        If no delimiter row is found, both returns are None.
    """
    delimiter_idx = None
    for idx in range(gen.shape[0]):
        if torch.all(gen[idx] == target_value):
            delimiter_idx = idx
            break

    if delimiter_idx is not None:
        return gen[:delimiter_idx], gen[delimiter_idx:]
    else:
        print("No delimiter row (all 256) found.")
        return None, None

def remove_all_256_columns(full_gen: torch.Tensor, target_value: int = 256):
    """
    Remove every row that consists entirely of `target_value`.

    Parameters
    ----------
    full_gen : Tensor  [N, 44]
    target_value : int, default 256

    Returns
    -------
    Tensor
        Filtered tensor with padding rows removed.
    """
    mask = torch.all(full_gen == target_value, dim=1)
    return full_gen[~mask, :]



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