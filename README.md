# PixelGen

A transformer-based model for piano music continuation that generates coherent 12-measure continuations from 4-measure prompts.

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
The model follows the MIREX 2025 submission format:

```bash
./generation.sh "/path/to/input.json" "/path/to/output_folder" 8
```

**Parameters:**
- `input.json`: Prompt file containing 5 measures (including pickup)
- `output_folder`: Directory for generated samples
- `8`: Number of samples to generate (sample_01.json to sample_08.json)

### Input Format
```json
{
  "prompt": [
    {
      "start": 16,
      "pitch": 72,
      "duration": 6
    }
  ]
}
```

### Output Format
```json
{
  "generation": [
    {
      "start": 80,
      "pitch": 65,
      "duration": 4
    }
  ]
}
```



  year={2025}
}
```
