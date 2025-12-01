# Documentation

This file provides commands to launch the project components: dataset labeling, model training, and the frontend application.

## Setup

```bash
git clone https://github.com/lolyhop/gnn-wiki-project
cd gnn-wiki-project
pip install -r requirements.txt
```

## Data Setup

The pipelines require two datasets available on Google Drive:
[Google Drive Link](https://drive.google.com/drive/folders/1meiA2D5PodBqxQUNpKZLzJsVJWprVvcR)

1. Create a `data/` directory in the project root.
2. Download and place the following files into `data/`:
   - `wiki_graph_restored_knn.json` (Raw format of Wiki graph)
   - `wiki_it_graph_scibert_feats.pt` (Processed PyTorch Geometric tensors)

---

## 1. Dataset Preparation

To reproduce the node labeling process (assigning taxonomy classes using embedding models):

1. Open `src/dataset_preparation/nodes_labeling.py`.
2. Update the `INPUT_GRAPH` constant to point to your raw parsed graph.
3. Run the module from the project root:

```bash
python -m src.dataset_preparation.nodes_labeling
```

## 2. Model Training

**Note:** Before running, open `src/model_training/gat.py` or `src/model_training/baseline.py` and ensure the `PATH` constant points to your local `wiki_it_graph_scibert_feats.pt` file.

### Train MLP Baseline
Trains a text-only Multi-Layer Perceptron on SciBERT features.

```bash
python -m src.model_training.baseline
```

### Train GAT
Trains the Graph Attention Network.

```bash
python -m src.model_training.gat
```

## 3. Frontend Application

To launch the interactive visualization and search dashboard:

1. Open `src/frontend/app.py`.
2. Update the path constants to match your local environment:

```python
# src/frontend/app.py

# Path to raw JSON
JSON_PATH = Path("data/wiki_graph_restored_knn.json")

# Path to PyG Tensors
PT_PATH = Path("data/wiki_it_graph_scibert_feats.pt")

# Path to trained model weights
MODEL_PATH = Path("models/model.pt")
```

3. Run the application:

```bash
streamlit run src/frontend/app.py
```

The app will be available at `http://localhost:8501`.

## Jupyter Notebooks

The `notebooks/` directory contains the research/experimental code used for:
*   Data crawling and processing;
*   MLP/GAT training;
*   LLM-as-a-judge analysis.
