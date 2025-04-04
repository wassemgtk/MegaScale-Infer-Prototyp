# MegaScale-Infer Prototype

## Overview

This repository contains a Python-based prototype implementation of the "MegaScale-Infer" system, inspired by the paper *"MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism"* by Ruidong Zhu et al. (ByteDance Seed & Peking University, 2025). The system is designed for efficient and cost-effective serving of large-scale Mixture-of-Experts (MoE) models by disaggregating attention and feed-forward network (FFN) modules, employing ping-pong pipeline parallelism, and optimizing communication. 

This prototype adapts these concepts for training an MoE model in a Google Colab environment, simulating disaggregation and communication due to Colab's single-device limitation. It includes a simplified MoE architecture, a synthetic dataset, and a training loop, making it suitable for experimentation and educational purposes.

## Features

- **Disaggregated Expert Parallelism**: Simulates separation of attention and FFN (expert) modules.
- **Ping-Pong Pipeline Parallelism**: Processes micro-batches to mimic overlapping computation and communication.
- **MoE Layer**: Implements a gating mechanism with top-k expert selection.
- **Training Support**: Includes a full training loop with a dummy dataset, loss function, and optimizer.
- **Simulated M2N Communication**: Uses delays to emulate network latency between modules.

## Prerequisites

- **Google Colab Account**: Free tier with GPU runtime recommended (e.g., T4 GPU).
- **Python**: Version 3.7+ (pre-installed in Colab).
- **PyTorch**: Version 2.0+ (installable via `pip` if not pre-installed).

## Setup

1. **Open Google Colab**:
   - Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

2. **Set Runtime to GPU** (optional, for faster training):
   - Click `Runtime` > `Change runtime type` > Select `GPU` > Save.

3. **Install Dependencies**:
   - Run the following command in a Colab cell to ensure PyTorch is installed:
     ```bash
     !pip install torch
     ```
   - Colab typically has PyTorch pre-installed (e.g., 2.0.x as of April 2025). Verify with:
     ```python
     import torch
     print(torch.__version__)
     ```

4. **Copy the Script**:
   - Paste the full training script (provided separately) into a Colab code cell.

## Usage

1. **Run the Script**:
   - Execute the cell containing the script by clicking the play button or pressing `Shift + Enter`.
   - The script will:
     - Initialize a `MegaScaleInfer` model with an MoE architecture.
     - Create a synthetic dataset (`DummyDataset`) with random inputs and targets.
     - Train the model for 5 epochs, printing loss and timing metrics.

2. **Sample Output**:
   ```
   Running on: cuda
   Starting training...
   Micro-batch 1/4: Attention time: 0.0123s, Expert time: 0.0456s
   Micro-batch 2/4: Attention time: 0.0118s, Expert time: 0.0432s
   ...
   Epoch [1/5], Batch [0/32], Loss: 1.2345
   Epoch [1/5] completed in 12.34s, Avg Loss: 1.1234, Total Attention Time: 0.4567s, Total Expert Time: 1.6789s
   ...
   Training completed!
   ```

3. **Customize Hyperparameters**:
   - Modify the `main()` function to adjust:
     - `hidden_size`: Dimensionality of input/output (default: 256).
     - `num_experts`: Number of FFN experts (default: 8).
     - `top_k`: Number of experts selected per token (default: 2).
     - `batch_size`: Training batch size (default: 32).
     - `num_micro_batches`: Number of micro-batches for ping-pong pipeline (default: 4).
     - `num_epochs`: Training epochs (default: 5).

## Code Structure

- **`AttentionModule`**: Simplified multi-head attention with QKV projection and KV cache.
- **`Expert`**: Single FFN expert with two linear layers and ReLU activation.
- **`MoELayer`**: MoE layer with gating and top-k expert dispatch.
- **`MegaScaleInfer`**: Main model integrating attention, MoE, and pipeline parallelism.
- **`DummyDataset`**: Synthetic dataset for regression task.
- **`train_megascale_infer`**: Training loop with loss computation and optimization.
- **`simulate_m2n_communication`**: Placeholder for M2N communication latency.

## Limitations

- **Single Device**: Runs on one GPU/CPU in Colab, simulating disaggregation rather than using multiple nodes.
- **Simplified Task**: Uses a dummy regression task; real NLP tasks require additional datasets and tokenization.
- **No Real M2N**: Communication is simulated with `time.sleep` due to lack of RDMA or multi-GPU support in Colab.
- **Resource Constraints**: Colab’s free tier (e.g., 12GB GPU memory) limits model size and batch size.

## Extending the Prototype

- **Real Dataset**: Replace `DummyDataset` with a real dataset (e.g., via `!pip install datasets` and `datasets` library).
- **Task Enhancement**: Switch to language modeling by adding a vocabulary and `nn.CrossEntropyLoss`.
- **Profiling**: Use `torch.profiler` for detailed performance analysis.
- **Multi-GPU**: Adapt for a multi-GPU setup outside Colab using `torch.distributed`.

## References

- Zhu, Ruidong et al. "MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism." ByteDance Seed & Peking University, 2025.

## License

This prototype is provided for educational and experimental purposes under the MIT License.

