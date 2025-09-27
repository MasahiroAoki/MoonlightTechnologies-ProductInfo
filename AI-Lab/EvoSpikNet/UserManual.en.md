# Copyright 2025 Moonlight Technologies Inc.. All Rights Reserved.
# Auth Masahiro Aoki

# EvoSpikeNet Dashboard User Manual

## 1. Introduction

This document explains how to use each feature of the `EvoSpikeNet Dashboard` and the technical implementation behind them. This dashboard is a tool for interactively running and visualizing the various functions of the EvoSpikeNet framework from a browser.

## 2. Setup and Installation

This project requires a Python 3.8 or higher environment.

### 2.1. Installing Dependencies

Run the following command in the project's root directory to install all necessary libraries.

```bash
pip install -e .
```

This command automatically installs the following key dependencies listed in the `pyproject.toml` file:
- `torch`
- `pandas`
- `plotly`
- `dash`
- `networkx`
- `tqdm`
- `wikipedia-api`
- `beautifulsoup4`
- `psutil`
- `snntorch`
- `transformers`
- `openai`

The `-e` flag means installing in "editable mode," which is recommended for development as changes to the source code are immediately reflected.

### 2.2. (Optional) Setting the OpenAI API Key

To use the `examples/generate_distilled_dataset.py` script (LLM data distillation feature), you need an OpenAI API key. Please set the following environment variable.

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 3. Main Interface

The dashboard consists of three main tabs.

1.  **Simulations & Models Tab:**
    -   This tab consolidates features for running basic SNN simulations and inference with trained models.
    -   Select a feature from the `Select Model / Feature` dropdown, configure the displayed parameters, and click the "Run Execution" button to run.
2.  **LM Training (Advanced) Tab:**
    -   A dedicated interface for flexibly training the language model (`EvoSpikeNetLM`) using external data sources like Wikipedia and Aozora Bunko. See section `4.5` for details.
3.  **System & Data Tab:**
    -   A "Command Panel" for executing project management tasks such as running tests, generating data, and simple model training.

---

## 4. Detailed Feature Explanations

### 4.1. Standard SNN Simulation

-   **Purpose:**
    To simulate the behavior of a core SNN model composed of basic LIF (Leaky Integrate-and-Fire) neurons and sparse synaptic connections.
-   **Parameters:**
    -   `Input Neurons`: Number of neurons in the input layer.
    -   `Output Neurons`: Number of neurons in the output layer.
    -   `Time Steps`: The number of time steps to run the simulation.
    -   `Connectivity Ratio`: The ratio (0.0 to 1.0) at which neurons from the input and output layers are randomly connected.
-   **Output:**
    A raster plot showing the firing activity of the output layer neurons is displayed. Black dots represent neuron firings (spikes) at each time step.
-   **Logical Implementation:**
    Pressing the `Run Execution` button calls the `run_standard_snn_simulation` function in `frontend/app.py`. This function uses the `LIFNeuronLayer`, `SynapseMatrixCSR`, and `SNNModel` classes from `evospikenet.core` to build an SNN with the specified parameters. Spikes are recorded with `DataMonitorHook`, and a raster plot is generated with `InsightEngine`.

### 4.2. Entangled Synchrony Layer

-   **Purpose:**
    To verify the behavior of a special layer inspired by quantum entanglement. It stochastically generates synchronous firing of neuron groups based on input spikes and context information.
-   **Parameters:**
    -   `Number of Neurons`: The number of neurons in the layer.
    -   `Initial Strength (0-255)`: The initial value of the synchronization strength. A higher value makes it more likely for neuron groups to fire in sync.
-   **Output:**
    The generated synchronous spike pattern is displayed as a heatmap.
-   **Logical Implementation:**
    The `run_synchrony_layer_simulation` function is called. This function instantiates the `EntangledSynchronyLayer` from `evospikenet.core` and runs a forward pass with random spikes and context data. The output is visualized as a single-timestep spike pattern.

### 4.3. Hardware Fitness Evaluator

-   **Purpose:**
    To demonstrate an evaluation function that calculates a "fitness" score, considering not only simulation performance but also hardware metrics like power consumption and temperature.
-   **Parameters:**
    -   `Power (e.g., 100-150)`: Simulated power consumption.
    -   `Temperature (e.g., 30-80)`: Simulated operating temperature.
    -   `Error Rate (0.0-1.0)`: The model's computational error rate.
-   **Output:**
    A single fitness score calculated based on the input hardware metrics is displayed.
-   **Logical Implementation:**
    The `run_fitness_evaluation` function is called. It uses the `HardwareFitnessEvaluator` from `evospikenet.fitness` to calculate a fitness score from the input parameters.

### 4.4. Text Classification

-   **Purpose:**
    To classify the sentiment (positive/negative) of an input sentence in real-time using a pre-trained `SpikingTransformerClassifier` model.
-   **Parameters:**
    -   `Enter a sentence to classify:`: A text area to input the English sentence you want to classify.
-   **Output:**
    The predicted sentiment label (`Positive` or `Negative`) and the model's confidence in that prediction are displayed as a percentage.
-   **Logical Implementation:**
    The `run_text_classification` function is called. This function uses the pre-trained model and vocabulary (`vocab.json`) that are loaded from the `saved_models/` directory upon app startup and held in memory. The input sentence is converted into a tensor of word IDs by the `tokenize_and_pad` function and fed into the model. A softmax function is applied to the model's output (logits) to calculate the final prediction and confidence.
    **Note:** To use this feature, you must first train the model using the "Train Text Classifier" button in the "Command Panel".

### 4.5. LM Training (Advanced)

-   **Purpose:**
    To train the `EvoSpikeNetLM` language model on more practical and large-scale data sources, such as Wikipedia, Aozora Bunko, and local files, in addition to the default corpus. This tab provides a comprehensive UI for a flexible training process.
-   **Parameters:**
    -   `Select Data Source`: Choose the data source for training.
        -   `Wikipedia`: Uses a Wikipedia article with the specified title as the corpus. Enter the article title (e.g., `Artificial intelligence`) in the input field below.
        -   `Aozora Bunko`: Uses a work from Aozora Bunko as the corpus. Enter the URL of the work's HTML file in the input field below.
        -   `Local File`: Uses a text file (`.txt`) in the workspace as the corpus. Enter the path to the file (e.g., `my_corpus.txt`) in the input field below.
        -   `Default Corpus`: Uses the small, hardcoded default corpus within the script.
    -   `Epochs`: The number of times to repeat the training.
    -   `Learning Rate`: Determines the step size when updating the model's parameters.
    -   `Block Size`: The sequence length (number of tokens) of text used in a single training step.
    -   `Batch Size`: The number of sequences used in a single parameter update.
-   **Operation:**
    1.  Set all parameters.
    2.  Click the "Start Advanced Training" button to begin the training process in the background.
    3.  Progress, such as the current epoch and loss, will be displayed in real-time in the log area below the button.
    4.  When training is complete, a `--- Training finished. ---` message will be displayed.
-   **Logical Implementation:**
    When the "Start Advanced Training" button is clicked, the `start_advanced_training` callback in `frontend/app.py` is triggered. This callback assembles the command-line arguments for running `python examples/train_evospikenet_lm.py` based on the parameters set in the UI. The training script is executed asynchronously via `subprocess.Popen`, and its standard output is redirected to a temporary log file. The UI uses a `dcc.Interval` component to periodically poll this log file and stream its contents to the log display area, achieving real-time progress reporting.

---

## 5. Command Panel

The `System & Data` tab contains buttons for executing project management tasks.

-   **Generate Data:**
    -   **Execution Script:** `./scripts/run_data_generation.sh`
    -   **Purpose:** Calls `scripts/generate_spike_data.py` to generate dummy data for basic SNN simulations. (Currently mainly for demo purposes)
-   **Run CPU Tests:**
    -   **Execution Script:** `./scripts/run_tests_cpu.sh`
    -   **Purpose:** Runs the entire project's test suite in CPU mode using `pytest`. Used to verify code integrity.
-   **Train Text Classifier:**
    -   **Execution Command:** `python examples/run_text_classification_experiment.py`
    -   **Purpose:** Runs the training for the text classification model. Upon successful completion, a trained model (`text_classifier.pth`) and vocabulary file (`vocab.json`) are saved to the `saved_models/` directory, making the "Text Classification" tab functional.
-   **Train EvoSpikeNetLM:**
    -   **Execution Command:** `python examples/train_evospikenet_lm.py`
    -   **Purpose:** Runs a simple training of the `EvoSpikeNetLM` language model with default settings. For more detailed data source selection or parameter tuning, use the "LM Training (Advanced)" tab.
-   **Train Spiking LM (New):**
    -   **Execution Command:** `python examples/train_spiking_evospikenet_lm.py`
    -   **Purpose:** Runs both training and evaluation for the newly implemented `SpikingEvoSpikeNetLM` model at once. After training, the model and tokenizer are saved to `saved_models/`, and the evaluation phase runs subsequently.
    -   **Output:** In addition to the training log, the final evaluation results, including Average Loss and **Perplexity**, are displayed. Perplexity is a performance metric for language models, where a lower value indicates that the model can generate text more plausibly.
-   **Generate Distilled Data (LLM):**
    -   **Execution Command:** `python examples/generate_distilled_dataset.py`
    -   **Purpose:** Calls a Large Language Model (LLM) to automatically generate higher-quality and more diverse test data. Requires the `OPENAI_API_KEY` environment variable to be set. The generated data is saved as `distilled_dataset.json`.

---

## 6. Advanced Workflows

### 6.1. Hyperparameter Tuning

To maximize model performance, it is important to tune hyperparameters such as the learning rate and model architecture. The project includes a helper script to automate this process.

-   **Execution Script:** `./scripts/run_hp_tuning.sh`
-   **Purpose:** Automatically runs hyperparameter tuning for the `SpikingEvoSpikeNetLM` model.
-   **Mechanism:**
    1.  Open the script, and you will find arrays like `LEARNING_RATES` and `NUM_BLOCKS`. Set these arrays with the parameter values you want to try.
    2.  When the script is executed, it sequentially calls `examples/train_spiking_evospikenet_lm.py` for all combinations of these values.
    3.  The training results for each run (model file and tokenizer) are saved in a uniquely named subdirectory within `saved_models/`, reflecting the parameter names (e.g., `lr_0.001-blocks_2`). This prevents the results of each experiment from being overwritten.
-   **How to Use:**
    1.  Open `scripts/run_hp_tuning.sh` in a text editor.
    2.  Edit the `LEARNING_RATES` and `NUM_BLOCKS` arrays to a list of values you want to test.
    3.  Run `./scripts/run_hp_tuning.sh` in the terminal.
    4.  The log for each run will be displayed in the console, and upon completion, the results will be organized under `saved_models/`.

### 6.2. Generating Text with a Trained Model

Once the `SpikingEvoSpikeNetLM` has been trained, you can use the model to generate (infer) new text. A dedicated script is provided for this purpose.

-   **Execution Command:** `python examples/run_spiking_lm_generation.py`
-   **Purpose:** Loads a trained `SpikingEvoSpikeNetLM` and generates text that follows a given prompt.
-   **Mechanism:**
    1.  The script loads the trained model (`spiking_lm.pth`), tokenizer, and model configuration (`model_config.json`) from the directory specified by the `--run-name` argument.
    2.  It reconstructs the model with the exact same architecture as during training, based on the model configuration.
    3.  It tokenizes the text provided with `--prompt` and passes it to the model's `generate` method to perform text generation.
    4.  The generated token sequence is decoded and output as human-readable text.
-   **How to Use:**
    ```bash
    # Example of generating text with a model saved as 'my_test_run'
    python examples/run_spiking_lm_generation.py \
        --run-name "my_test_run" \
        --prompt "EvoSpikeNet is a framework that" \
        --max-new-tokens 50 \
        --temperature 0.8
    ```
-   **Key Arguments:**
    -   `--run-name`: (Required) The run name (directory name) of the desired model saved in `saved_models/`.
    -   `--prompt`: (Required) The starting prompt string for text generation.
    -   `--max-new-tokens`: The maximum number of new tokens to generate.
    -   `--temperature`: Controls the randomness of the generation. Lower values are more deterministic, while higher values produce more diverse text.
    -   `--top-k`: Restricts sampling to the top K most likely tokens.

---

## 7. Command-Line Usage: Running Scripts

The `scripts/` directory contains convenient wrapper scripts for running the project's various features using Docker containers. These scripts internally call `docker compose` commands to launch containers in the appropriate environment (CPU or GPU).

**Note:** To run these scripts, `docker` and `docker-compose` must be installed on your local environment.

### 7.1. Running Tests

-   **Scripts:**
    -   `./scripts/run_tests_cpu.sh` (CPU mode)
    -   `./scripts/run_tests_gpu.sh` (GPU mode)
-   **Purpose:** To run the project's entire test suite and verify code integrity.
-   **Example Command:**
    ```bash
    # Run tests on CPU
    ./scripts/run_tests_cpu.sh
    ```

### 7.2. Running the Sample Application

-   **Scripts:**
    -   `./scripts/run_prod_cpu.sh` (CPU mode)
    -   `./scripts/run_prod_gpu.sh` (GPU mode)
-   **Purpose:** Runs `example.py`. This is a simple sample that instantiates `SNNModel` and runs a basic forward pass with random spike data.
-   **Example Command:**
    ```bash
    ./scripts/run_prod_cpu.sh
    ```

### 7.3. Launching the Frontend

-   **Scripts:**
    -   `./scripts/run_frontend_cpu.sh` (CPU mode)
    -   `./scripts/run_frontend_gpu.sh` (GPU mode)
-   **Purpose:** Launches the Dash-based web frontend.
-   **Example Command:**
    ```bash
    # Launch frontend in CPU mode
    ./scripts/run_frontend_cpu.sh
    ```
-   **Access:** Once the server is running, open `http://localhost:8050` in your browser.

### 7.4. Launching the Development Environment

-   **Scripts:**
    -   `./scripts/run_dev_cpu.sh` (CPU mode)
    -   `./scripts/run_dev_gpu.sh` (GPU mode)
-   **Purpose:** Launches an interactive `bash` shell with the project's source code mounted. Use it for development tasks like code changes, debugging, and adding packages.
-   **Example Command:**
    ```bash
    ./scripts/run_dev_cpu.sh
    ```

### 7.5. Data Generation and Hyperparameter Tuning

-   **Scripts:**
    -   `./scripts/run_data_generation.sh`
    -   `./scripts/run_hp_tuning.sh`
-   **Purpose:**
    -   `run_data_generation.sh`: Calls `scripts/generate_spike_data.py` to generate dummy data.
    -   `run_hp_tuning.sh`: Automatically runs hyperparameter tuning for the `SpikingEvoSpikeNetLM` model. See the comments in the script for details.

---

## 8. Core Concepts and Architecture

This section explains the technical concepts of the main components that form the core of the EvoSpikeNet framework.

### 8.1. Basic SNN Components (`evospikenet.core`)

-   **`LIFNeuronLayer`**:
    -   **Overview:** Implements a single layer of Leaky Integrate-and-Fire (LIF) neurons. This is one of the most fundamental spiking neuron models that mimics the behavior of biological neurons.
    -   **Features:**
        -   **Integer-based Arithmetic:** For performance and hardware simulation purposes, all calculations such as membrane potential and threshold are done with integers.
        -   **Stateful:** Each neuron maintains its own membrane potential (`potential`) as an internal state. This state is reset by the `SNNModel` class at the beginning of a sequence.
        -   **Energy-Driven:** Can optionally take an `EnergyManager` to impose an energy cost on neuron firing.

-   **`SynapseMatrixCSR`**:
    -   **Overview:** Manages connections between neurons (synapses) as a memory-efficient **CSR (Compressed Sparse Row)** format sparse matrix.
    -   **Features:**
        -   **Sparse Connections:** A sparse matrix is used to mimic the structure of the brain, where many neurons are not connected to each other.
        -   **CPU Workaround:** Current PyTorch (version 2.x) does not support integer sparse matrix multiplication on the CPU. Therefore, the forward pass of this class temporarily converts the matrix and input to floating-point numbers for calculation and then converts the result back to integers as a workaround. This is a performance consideration.

### 8.2. Spiking Transformer (`evospikenet.attention`)

-   **`SpikingTransformerBlock`**:
    -   **Overview:** A reconstruction of a standard Transformer block with spiking neuron components. It is the core of the `SpikingEvoSpikeNetLM` model.
    -   **Components:**
        -   **`ChronoSpikeAttention`**: A hybrid attention mechanism. At each time step, it treats input spikes as floating-point numbers and performs standard self-attention calculation (Scaled Dot-Product Attention). The result is then passed through an LIF neuron to generate spikes as output.
        -   **`SpikingFFN`**: A simple, fully connected feed-forward network composed of two LIF neuron layers.
    -   **Features:**
        -   **Hybrid Approach:** It avoids the complexity of calculating attention in a pure SNN by combining proven standard attention calculations with temporal spike processing by LIF neurons.
        -   **Simplified Residual Connection:** Since Layer Normalization used in standard Transformers is not compatible with the binary nature of spikes, a simple residual connection using addition and clipping (`torch.clamp(x + ...)` ) is implemented.

### 8.3. Model Architectures (`evospikenet.models`)

-   **`SpikingEvoSpikeNetLM`**:
    -   **Overview:** The core text-generative spiking language model of this project. It is constructed by combining `TASEncoding` (converting text to spike trains) and `SpikingTransformerBlock`.
    -   **Features:** Designed to work in conjunction with more advanced self-organization and learning mechanisms like `AEG` and `MetaSTDP`.

-   **`MultiModalEvoSpikeNetLM`**:
    -   **Overview:** A multi-modal spiking language model that can take both text and images as input.
    -   **Architecture:**
        1.  **Encoding:** Text is converted to spike trains by `TASEncoding`, and images are converted by `SpikingVisionEncoder` (a convolutional SNN).
        2.  **Feature Fusion:** Spike features obtained from the entire image are broadcast (expanded) to the spike trains corresponding to each token of the text and concatenated along the feature dimension.
        3.  **Fusion Layer:** The combined features are processed by a linear layer and an LIF neuron layer (`fusion_lif`) to generate a single fused spike train.
        4.  **Processing:** The fused spike train is passed through multiple `SpikingTransformerBlock`s to process contextual information.
        5.  **Decoding:** The final output spikes are integrated over time and passed through a linear layer to be converted into vocabulary logits (probability distribution) to predict the next token.

### 8.4. Current Status and Usage of the Multi-Modal Model

The `MultiModalEvoSpikeNetLM` is designed as a more advanced model capable of processing both text and image inputs.

-   **Architecture:**
    -   For a detailed description of the model's architecture, please refer to the previous section `8.3. Model Architectures`.

-   **Current Implementation Status and Demonstration:**
    -   The model's architecture is fully implemented and tested.
    -   A demonstration script, `examples/train_multi_modal_lm.py`, is provided to verify the model's operation.
    -   **Important:** This script is intended to show that the model's training process (forward/backward pass) can run without errors using randomly generated **dummy image and text data**. The functionality to **load real image/text files for training or to save a trained model has not yet been implemented.**

-   **How to Run (Demo):**
    ```bash
    # Run the demo script
    python examples/train_multi_modal_lm.py --epochs 3
    ```
    This command trains the model for 3 epochs on dummy data and displays the average loss for each epoch.

-   **Future Outlook:**
    -   Implementation of a data loader for real-world datasets (e.g., image captioning data).
    -   Addition of functionality to save and load trained models.
    -   Development of an inference script to generate text conditionally based on an image and a prompt.

    These features are subjects for future development.
