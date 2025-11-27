# Functional Specification: Distributed Brain Simulation

Last Updated: 2025-11-26

## 1. Overview

The Distributed Brain Simulation is a large-scale Spiking Neural Network (SNN) architecture inspired by the functional specialization and coordinated operation of the human brain. This simulation achieves complex multi-modal task processing by orchestrating multiple independent processes (nodes).

A central **PFC (Prefrontal Cortex) Decision Engine** acts as a command center, dynamically routing tasks to specialized functional modules—such as vision, audition, language, and motor control—based on the input information. This system provides a flexible and scalable platform for researchers and developers to explore the brain's information processing mechanisms and build more advanced AI systems.

## 2. Architecture

The simulation employs a master/slave distributed architecture based on the `torch.distributed` library.

- **Master Node (Rank 0):**
  - Assigned to the **PFC (Prefrontal Cortex) Decision Engine**.
  - Responsible for accepting, interpreting, and dispatching all tasks to slave nodes.
  - Calculates the system's overall cognitive load and adjusts its own operations accordingly.

- **Slave Nodes (Rank > 0):**
  - Assigned to modules with specific functions, such as vision, language, or motor control.
  - They wait for instructions from the PFC, execute their assigned tasks, and return the results.

- **Inter-Process Communication:**
  - Data communication between nodes is handled using `torch.distributed`'s `send` and `recv` primitives.
  - The system implements a robust **manifest-based communication protocol** to send multiple data types at once (e.g., text and an image). The PFC first sends a manifest tensor indicating the contents of the subsequent data (e.g., `[text_present, image_present, audio_absent]`), followed by the actual data tensors in order. This enables specialized nodes to handle multimodal models like `MultiModalEvoSpikeNetLM`.

- **Hierarchical Module Structure:**
  - Certain functions (especially language, vision, and audition) can form a hierarchical structure with child nodes that perform more detailed sub-tasks. For example, the "Language" module can manage child processes for "Word Embedding," "TAS Encoding," and so on, building a more complex language processing pipeline.

![Architecture Diagram](https://dummyimage.com/800x400/cccccc/000000.png&text=Architecture+Diagram)
*(Figure: A conceptual diagram of the master/slave architecture centered on the PFC, showing inter-node communication.)*

## 3. Key Components

### 3.1. PFCDecisionEngine

Implemented in `evospikenet/pfc.py`, the PFC Decision Engine is the core of the entire simulation.

- **Roles:**
  - **Task Routing:** Interprets the content of input data (text, image, audio) and assigns the task to the most appropriate functional module.
  - **Working Memory:** Uses a `LIFNeuronLayer` to maintain short-term information, preserving context for sequential task processing.
  - **Cognitive Load Calculation:** Quantifies the system's "cognitive load" by calculating the entropy of its decision-making probability distribution.

- **Unique Features:**
  - **ChronoSpikeAttention:** A custom attention mechanism designed to process temporal spike information. It determines which parts of an incoming spike train to focus on, enhancing decision-making accuracy.
  - **Quantum-Inspired Feedback Loop:**
    - The `QuantumFeedbackSimulator` simulates a 2-qubit circuit, using the cognitive load (entropy) as a parameter.
    - The expectation value derived from this simulation serves as a feedback signal to dynamically modulate the membrane potential of the PFC's working memory (the LIF layer).
    - This allows the PFC to self-regulate its internal state based on the complexity of its own decisions, realizing an advanced meta-cognitive function.

### 3.2. Functional Modules

These are the specialized slave modules that operate under the direction of the PFC.

- **VisualModule:** Processes image data. When hierarchical, it has sub-modules for tasks like edge detection, shape extraction, and object classification.
- **AuditoryModule:** Processes audio data. Can be divided into sub-modules for MFCC feature extraction, phoneme classification, etc.
- **LanguageModule:** Processes text data. It can have child processes forming a complex pipeline for word embedding, encoding, and RAG (Retrieval-Augmented Generation).
- **MotorModule:** Generates motor control signals.
- **SpeechGenerationModule:** Synthesizes speech from text.

## 4. Execution and Operation

The simulation is managed through a web UI implemented in `frontend/pages/distributed_brain.py`.

### 4.1. UI Overview

The UI consists of three main tabs: "Node Configuration," "Brain Simulation," and "Learning."

- **Simulation Control (Always Visible):**
  - **Select Simulation Type:** Choose a predefined simulation configuration (described later).
  - **Start/Stop Nodes:** Controls the startup and shutdown of the entire simulation.

### 4.2. Operational Flow

1.  **Select Simulation Type:**
    - Choose a predefined configuration, such as "Language Focus," from the dropdown in the "Simulation Control" panel.

2.  **Configure Nodes (Node Configuration Tab):**
    - A list of required nodes (PFC, Visual, etc.) for the selected type is displayed.
    - Assign each slave node (Rank > 0) to a host using the dropdown. Selecting `localhost` runs all processes locally. You can also add and configure remote hosts.

3.  **Start the Simulation:**
    - Click the "Start Nodes" button. The processes for each node will be launched according to the configuration. Local nodes are run via `subprocess`, while remote nodes are executed in the background via SSH.

4.  **Execute a Query (Brain Simulation Tab):**
    - Once the simulation is running, you can interact with it from this tab.
    - You can input a text prompt and upload image or audio files.
    - Clicking the "Execute Query" button sends the input data to the PFC (Rank 0) via the API.

### 4.3. Visualization and Monitoring

The "Brain Simulation" tab provides real-time visualization of the simulation's internal state.

- **Live Simulation Graph:**
  - A network graph powered by Cytoscape.
  - It displays the status of each node (represented by color) and active communication between nodes (highlighted edges).
- **PFC Energy and Entropy:**
  - A time-series plot showing the PFC's energy consumption and cognitive load (entropy).
- **PFC Spike Train / Membrane Potential:**
  - Visualizations of the firing patterns and membrane potential distribution of the neurons comprising the PFC's working memory, allowing for detailed analysis of its internal state.
- **Node Logs:**
  - A dropdown menu allows you to select a node and view its standard output log (`/tmp/sim_rank_{rank}.log`). This is essential for debugging the distributed system.

## 5. Data Flow

The typical data flow when a user executes a query is as follows:

1.  **UI → API:** The user submits a prompt (text/image/audio) from the UI. The data is Base64-encoded and POSTed to the `/api/distributed_brain/prompt` endpoint on the API server.
2.  **API → PFC:** The API server interprets the data and saves it as a file (`/tmp/evospikenet_prompt_{uuid}.json`) that the simulation process (specifically the PFC) polls for.
3.  **PFC (Task Interpretation and Routing):**
    - The PFC (Rank 0) detects and reads the prompt file.
    - It determines the content of the data (e.g., does it contain an image? audio?) and decides on the optimal slave node to handle the processing (e.g., `VisualModule` at Rank 1).
4.  **PFC → Slave (Data Transmission):**
    - Following the designed communication protocol, the PFC first sends a manifest tensor to the target rank.
    - It then sequentially sends the relevant data included in the prompt (e.g., the text tensor and the image tensor).
5.  **Slave (Task Execution):**
    - The slave node first receives the manifest to understand the type and number of subsequent data tensors.
    - It receives the data tensors the required number of times and passes them to its model (e.g., `MultiModalEvoSpikeNetLM`) to execute the task and generate a result tensor.
6.  **Slave → PFC (Result Return):**
    - The slave node sends the result tensor back to the PFC (Rank 0) using `dist.send`.
7.  **PFC → API:**
    - The PFC receives the final result, converts it into a human-readable format (e.g., text), and POSTs it to the `/api/distributed_brain/result` endpoint.
8.  **API → UI:**
    - The UI periodically polls the API server for the latest result and displays it in the "Query Response" area.

## 6. Selectable Simulation Configurations

The `simulation-type-dropdown` allows you to choose from the following predefined configurations:

- **Language Focus (9-Proc):**
  - A configuration specialized for language processing. Includes the PFC, main modules, and dedicated nodes for a language processing pipeline (Embed, TAS).
- **Image Focus (11-Proc):**
  - A configuration specialized for visual processing. The `VisualModule` acts as a parent node, forming a hierarchy with child nodes for edge detection, shape extraction, and object classification.
- **Audio Focus (12-Proc):**
  - A configuration specialized for auditory information processing. The `AuditoryModule` and `SpeechGenerationModule` each have child nodes, forming pipelines for speech recognition and synthesis.
- **Motor Focus (11-Proc):**
    - A configuration specialized for motor control. The `MotorModule` acts as a parent node with children for trajectory generation (Traj), PID control (Cereb), and PWM signal generation (PWM).
- **Full Brain (21-Proc):**
  - The largest-scale simulation configuration, integrating all of the hierarchical modules described above.
