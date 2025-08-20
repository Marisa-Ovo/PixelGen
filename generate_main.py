# Sampling entry script
# --------------------------------------------------------------
# 1. Load pretrained NotaGenLMHeadModel
# 2. Convert MIREX prompt JSON → internal tensors
# 3. Autoregressively generate patch tokens with `generate_special`
# 4. Convert tokens → pianoroll, save MIDI
# --------------------------------------------------------------

import torch
from transformers import GPT2Config
import safetensors.torch
from midi_utils.utils import map_div_tokens, patch_tokens_to_image,pianoroll_to_multitrack
from json_prompt.get_generation import get_json
from model import PixelGenLMHeadModel
from json_prompt.load_json import load_json
import os, sys, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------------------------------------------------------------------
# model hyper-parameters
# ------------------------------------------------------------------
PATCH_SIZE        = 44
PATCH_LENGTH      = 2048
CHAR_NUM_LAYERS   = 5
PATCH_NUM_LAYERS  = 12
HIDDEN_SIZE       = 1024

time_signature_dict = {"2/2": 0, "2/4": 1, "6/8": 2, "3/4": 3, "4/4": 4}

patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=1,
)
char_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE + 1,
    max_position_embeddings=PATCH_SIZE + 1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=258,
)

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def split_tensor_on_condition(gen: torch.Tensor, target_value: int = 256):
    """
    Split a [N,44] tensor at the first row that is entirely `target_value`.

    Returns
    -------
    (Tensor | None, Tensor | None)
        short_gen : rows before delimiter
        full_gen  : rows from delimiter onward
    """
    delimiter = None
    for idx in range(gen.shape[0]):
        if torch.all(gen[idx] == target_value):
            delimiter = idx
            break
    if delimiter is not None:
        return gen[:delimiter], gen[delimiter:]
    else:
        print("Delimiter row (all 256) not found.")
        return None, None


def remove_all_256_columns(full_gen: torch.Tensor, target_value: int = 256):
    """
    Remove every row composed solely of `target_value`.
    """
    mask = torch.all(full_gen == target_value, dim=1)
    return full_gen[~mask, :]


def convert_prompt_json_to_internal(input_json_path: str):
    """
    Convert MIREX prompt JSON to internal tensor dict.

    Returns a dict with:
      * 'ground_tokens' : Tensor[1, N, 44]
      * 'div_tokens'    : Tensor[1, 2, 44]   (first two rows, div-8 version)
    """
    ground_indices = load_json(input_json_path)           # [N,44] tensor
    div_indices    = map_div_tokens(ground_indices)       # [N,44] numpy
    div_indices    = div_indices[:2, :]                   # first 2 rows

    div_tensors    = torch.from_numpy(div_indices).unsqueeze(0).long()
    ground_indices = torch.from_numpy(ground_indices).unsqueeze(0).long()

    return {"ground_tokens": ground_indices, "div_tokens": div_tensors}


def tokens_to_generation_notes(restored_tensor: torch.Tensor):
    """
    Convert a [128,T] pianoroll tensor into MIREX 'generation' list of dicts.
    """
    json_data = get_json(restored_tensor)
    return json_data


@torch.no_grad()
def load_model(device):
    model = PixelGenLMHeadModel(patch_config, char_config).to(device)
    weights = safetensors.torch.load_file(
        "./pretrained_model/model.safetensors"
    )
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


@torch.no_grad()
def generate_once(model, device, tensors, idx, out_dir,input_json):
    temperature = 1.1
    seq_len_ctrl = 16
    top_k = 10

    seq_len       = torch.tensor([[seq_len_ctrl]]).to(device)
    time_signature = torch.tensor([[4]]).to(device)

    gen = model.generate_special(
        gnd_patches=tensors["ground_tokens"].to(device),
        div_patches=tensors["div_tokens"].to(device),
        time_signature=time_signature,
        max_length=90,
        temperature=temperature,
        seq_len=seq_len,
        top_k=top_k,
    ).squeeze(0)                                          # [N,44]

    _, full_idx = split_tensor_on_condition(gen)
    full_idx    = remove_all_256_columns(full_idx)
    print(full_idx.shape)

    # Convert tokens → pianoroll
    music_roll = patch_tokens_to_image(full_idx)
    pianoroll  = music_roll.cpu()



    restored = torch.zeros((128, pianoroll.shape[1]))
    restored[20:108, :] = pianoroll
    if restored.shape[1] > 271:
        restored = restored[:, :272]


    restored_copy = restored[:, 80:]
    json_out = get_json(restored_copy)

    file_name = os.path.basename(input_json)
    # Save to JSON file
    with open(os.path.join(out_dir,f"{file_name}_gen{idx}.json"), "w") as f:
        json.dump(json_out, f, indent=2)



    midi = pianoroll_to_multitrack(restored, tempo=18)
    midi_path = os.path.join(out_dir, f"{file_name}_new{idx}_t{temperature}_l{seq_len_ctrl}_k{top_k}.mid")
    midi.write(midi_path)


def prepare_prompt_tensors_from_json(input_json_path: str):
    return convert_prompt_json_to_internal(input_json_path)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) >= 4:
        input_json = sys.argv[1]
        output_dir = sys.argv[2]
        n_sample   = int(sys.argv[3])
    else:
        print("No CLI arguments provided; using defaults (n_sample=10, output=./outputs).")
        input_json = "./json_prompt/prompt.json"
        output_dir = "./outputs"
        n_sample   = 10

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = load_model(device)
        tensors = prepare_prompt_tensors_from_json(input_json)

        for idx in range(n_sample):
            generate_once(model, device, tensors, idx, output_dir,input_json)


if __name__ == "__main__":
    main()
