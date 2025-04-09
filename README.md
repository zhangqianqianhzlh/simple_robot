# simple_robot

A robot navigation system in simulated environments using vision-language models (VLMs) to make navigation decisions based on visual input.

## Description

This project leverages AI2Thor for simulating indoor environments and uses Qwen2.5-VL vision-language model to enable a robot to navigate based on natural language instructions. The robot processes first-person view images and chooses appropriate actions to complete specified navigation tasks.

## Requirements

- Python 3.6+
- OpenAI API access or Silicon Cloud API access
- AI2Thor
- OpenCV
- PyYAML
- PIL/Pillow
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/simple_robot.git
cd simple_robot

# start a conda env
conda create -p /data23/xu_ruochen/conda_envs/simple_robot python=3.10 -y
conda activate /data23/xu_ruochen/conda_envs/simple_robot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `llm.yaml` file with your API credentials:

```yaml
SiliconCloud:
  api_key: "your_api_key_here"
  base_url: "your_base_url_here"
```

## Quick Start

1. Ensure you have created the `llm.yaml` configuration file with valid API credentials.

2. Run the main script:
```bash
python main.py
```

This will start a navigation session where the robot attempts to find a white fridge according to the instructions in the default scenario.

## Usage Examples

### Running a Custom Navigation Task

Modify the `target` variable in `main.py` to change the navigation goal:

```python
target = "find the kitchen table and stop when you see it"
```

### Testing the VLM Model

You can test the vision-language model on individual images:

```bash
python test_model.py
```

### Using the VLM in Your Own Code

```python
from models import VLM

# Initialize the VLM with your config
vlm = VLM("llm.yaml")

# Process a single image
response = vlm.run("path_to_image.jpg", "Qwen/Qwen2.5-VL-32B-Instruct", "Describe what you see")

# Process multiple images
responses = vlm.run(["image1.jpg", "image2.jpg"], "Qwen/Qwen2.5-VL-32B-Instruct", 
                    "Compare these two images")
```

## Project Structure

- `env.py`: Defines the AI2Thor environment wrapper
- `models.py`: Implements the VLM class for vision-language processing
- `main.py`: Main script for robot navigation
- `test_model.py`: Script for testing the VLM on single images

## License

[Your license information here]
