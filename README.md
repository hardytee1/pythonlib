# Dillema

Dillema is a thin convenience layer over Ray that helps you spin up a local cluster, inspect its status, and deploy an LLM with a single command.

## Installation

To install the project, you can use pip:

```bash
pip install .
```

This pulls in Ray with the Serve and LLM extras. If you plan to run GPU-backed deployments with vLLM, add the optional extras:

```bash
pip install .[llm]
```

Make sure you have Python 3.9–3.12 and a working CUDA/NVIDIA stack before deploying GPU models.

## Usage

Once installed, the CLI is available as `dillema`.

```bash
dillema start --head
```

Start a worker and attach it to the head node:

```bash
dillema start --worker --address "ray://HEAD_NODE_IP:10001"
```

Check the cluster status:

```bash
dillema status
```

Deploy an LLM via Ray Serve (defaults mirror the sample Ray Serve script):

```bash
dillema deploy --model "Qwen/Qwen2.5-14B-Instruct-AWQ"
```

Additional switches let you override the HTTP endpoint, model identifier, tensor/pipeline parallelism, GPU memory utilization, maximum context length, and networking interface used for collective backends. Run `dillema --help` for details.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.