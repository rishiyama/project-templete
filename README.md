# project-templete

## Environment Setup

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stroke-decomposition
   ```

2. Install dependencies:
   ```bash
   uv sync
   uv pip install -e .
   ```

### Optional: Using Docker

Alternatively, you can use Docker to set up the environment:

1. Build the Docker image:
   ```bash
   docker build -t stroke-decomposition .
   ```

2. Run the container:
   ```bash
   docker run -it --rm -v $(pwd):/app stroke-decomposition
   ```

3. Inside the Docker container, install dependencies:
   ```bash
   uv sync
   uv pip install -e .
   ```

## Visualization

For details on how to visualize the model's predictions, refer to the [Visualization Guide](docs/visualization.md).