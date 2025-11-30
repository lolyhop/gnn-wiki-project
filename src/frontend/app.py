import json
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import networkx as nx
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from streamlit_agraph import agraph, Node, Edge, Config

# Configuration
st.set_page_config(
    page_title="Wikipedia GNN Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
JSON_PATH = Path(
    "/Users/chrnegor/Documents/code/gnn-wiki-project/data/wiki_graph_restored_knn.json"
)
PT_PATH = Path(
    "/Users/chrnegor/Documents/code/gnn-wiki-project/data/wiki_it_graph_scibert_feats.pt"
)
MODEL_PATH = Path("/Users/chrnegor/Downloads/model.pt")

GRAPH_SAMPLE_SIZE = 350


# Device Setup
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# Model Definition
class GAT(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 8
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.25)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.25,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_embeds: bool = False
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if return_embeds:
            return x
        x = self.conv2(x, edge_index)
        return x


# Data Loading
@st.cache_resource(show_spinner=False)
def load_system() -> Tuple[Data, List[str], Dict[str, Any], np.ndarray]:
    """Loads all artifacts and precomputes embeddings."""
    try:
        # 1. Load Metadata
        with open(JSON_PATH, "r") as f:
            raw_data = json.load(f)
        nodes_dict = raw_data["nodes"]
        titles = list(nodes_dict.keys())

        # 2. Load Graph
        data = torch.load(PT_PATH, map_location=DEVICE, weights_only=False)

        # 3. Load Model
        loaded_obj = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        # Initialize architecture
        model = GAT(in_channels=768, hidden_channels=128, out_channels=8, heads=8).to(
            DEVICE
        )

        if isinstance(loaded_obj, nn.Module):
            model.load_state_dict(loaded_obj.state_dict())
        elif isinstance(loaded_obj, dict):
            model.load_state_dict(loaded_obj)
        else:
            raise ValueError("Unknown model file format")

        model.eval()

        # 4. Generate Embeddings
        with torch.no_grad():
            embeddings = (
                model(data.x, data.edge_index, return_embeds=True).cpu().numpy()
            )

        return data, titles, nodes_dict, embeddings

    except Exception as e:
        st.error(f"System Load Error: {e}")
        st.stop()


# Graph Sampling
@st.cache_data
def get_global_subgraph(num_nodes: int, _data: Data) -> nx.Graph:
    """
    Samples a dense subgraph for visualization.
    Uses Degree Centrality to pick the most 'important' nodes.
    """
    G_full = to_networkx(_data, to_undirected=True)

    # Pick top N nodes by degree to ensure connectivity
    degrees = dict(G_full.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:num_nodes]

    G_sub = G_full.subgraph(top_nodes)
    return G_sub


# UI
def render_home_page(titles: List[str], nodes_dict: Dict, data: Data):
    st.title("🌌 Wikipedia Knowledge Graph")

    col1, col2 = st.columns([4, 1])

    with col2:
        st.subheader("Navigation")

        selected = st.selectbox(
            "Find Article:",
            [""] + titles,
            format_func=lambda x: "Start typing..." if x == "" else x,
        )
        if selected:
            st.session_state["selected_node"] = selected
            st.rerun()

        st.divider()

        # Legend
        CATEGORY_COLORS = {
            "Programming & Software": "#FF6B6B",
            "AI & Machine Learning": "#4ECDC4",
            "Data Mgmt & Databases": "#FFE66D",
            "Computer Security": "#FF9F43",
            "Networking & Protocols": "#54A0FF",
            "Systems & Hardware": "#5F27CD",
            "Math & Formal Methods": "#00D2D3",
            "Other": "#888888",
        }

        legend_html = "<div style='font-size: 14px; line-height: 1.6;'>"
        for cat_name, color in CATEGORY_COLORS.items():
            legend_html += (
                f"<span style='color:{color}; font-size:16px'>●</span> {cat_name}<br>"
            )
        legend_html += "</div>"

        st.caption("🎨 **Legend:**")
        st.markdown(legend_html, unsafe_allow_html=True)
        st.info("👆 **Click on any node** to navigate.")

    with col1:
        G_sub = get_global_subgraph(GRAPH_SAMPLE_SIZE, data)
        d = dict(G_sub.degree)
        min_deg, max_deg = min(d.values()), max(d.values())

        nodes = []
        edges = []

        FULL_CATEGORY_COLORS = {
            "Programming & Software": "#FF6B6B",
            "Artificial Intelligence & Machine Learning": "#4ECDC4",
            "Data Management & Databases": "#FFE66D",
            "Computer Security": "#FF9F43",
            "Networking & Protocols": "#54A0FF",
            "Systems & Hardware": "#5F27CD",
            "Mathematics & Formal Methods": "#00D2D3",
            "Other": "#888888",
        }

        for node_id in G_sub.nodes():
            t = titles[node_id]
            cat = nodes_dict[t]["category"]

            degree = d[node_id]
            norm_size = 15 + (degree - min_deg) * (20 / (max_deg - min_deg + 1e-9))
            color = FULL_CATEGORY_COLORS.get(cat, "#888888")

            nodes.append(
                Node(
                    id=t,
                    label=t,
                    size=norm_size,
                    color=color,
                    title=f"Category: {cat}",
                    font={
                        "color": "white",
                        "size": 14,
                        "face": "arial",
                        "strokeWidth": 2,
                        "strokeColor": "#000000",
                    },
                )
            )

        for src, dst in G_sub.edges():
            edges.append(Edge(source=titles[src], target=titles[dst], color="#555555"))

        config = Config(
            width="100%",
            height=750,
            directed=False,
            physics=True,
            hierarchical=False,
            backgroundColor="#0E1117",
            nodeHighlightBehavior=True,
            highlightColor="#F7A072",
            collapsible=False,
            gravity=-200,
            central_gravity=0.005,
            spring_length=200,
            spring_strength=0.02,
            node_spacing=600,
        )

        return_value = agraph(nodes=nodes, edges=edges, config=config)

        if return_value:
            st.session_state["selected_node"] = return_value
            st.rerun()


def render_content_page(
    article_title: str,
    titles: List[str],
    nodes_dict: Dict,
    embeddings: np.ndarray,
    data: Data,
):
    # Setup
    node_idx = titles.index(article_title)
    meta = nodes_dict[article_title]

    # Back Button
    if st.button("← Back to Global Graph"):
        st.session_state["selected_node"] = None
        st.rerun()

    st.title(article_title)
    st.caption(f"📂 Category: {meta.get('category', 'General')}")
    st.divider()

    # Layout
    col_content, col_sidebar = st.columns([2.5, 1.2])

    with col_content:
        # 1. Main Text
        st.subheader("📖 Abstract")
        st.markdown(f"{meta.get('text', 'No text available.')}")

        st.divider()

        # 2. Find neighbors in Edge Index
        neighbor_mask = data.edge_index[0] == node_idx
        neighbor_indices = data.edge_index[1][neighbor_mask].tolist()

        st.subheader(f"🔗 References ({len(neighbor_indices)})")

        if neighbor_indices:
            cols = st.columns(3)
            for i, idx in enumerate(neighbor_indices):
                neighbor_title = titles[idx]
                if cols[i % 3].button(neighbor_title, key=f"link_{idx}"):
                    st.session_state["selected_node"] = neighbor_title
                    st.rerun()
        else:
            st.warning("No outgoing links in dataset.")

    with col_sidebar:
        # 3. GNN Recommendations
        st.markdown("### 🧠 You might want to read")
        st.info("AI-powered suggestions based on Graph Embeddings.")

        # Cosine Sim
        target_vec = embeddings[node_idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, embeddings)[0]

        # Top 6 excluding self
        top_indices = np.argsort(sims)[::-1][1:7]

        for idx in top_indices:
            rec_title = titles[idx]
            score = sims[idx]

            with st.container():
                st.write(f"**{rec_title}**")
                st.progress(float(score), text=f"Match: {int(score*100)}%")
                if st.button("Read", key=f"rec_{idx}"):
                    st.session_state["selected_node"] = rec_title
                    st.rerun()
                st.divider()


# Main App Controller
def main():
    # 1. Load Data
    with st.spinner("Starting GNN Engine..."):
        data, titles, nodes_dict, embeddings = load_system()

    # 2. Session State Initialization
    if "selected_node" not in st.session_state:
        st.session_state["selected_node"] = None

    # 3. Router
    if st.session_state["selected_node"] is None:
        render_home_page(titles, nodes_dict, data)
    else:
        render_content_page(
            st.session_state["selected_node"], titles, nodes_dict, embeddings, data
        )


if __name__ == "__main__":
    main()
