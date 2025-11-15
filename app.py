from shiny import App, Inputs, Outputs, Session, ui, render, reactive
from shinyswatch import theme
from shinywidgets import output_widget, render_widget
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------- Data Loading Helpers ----------

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ART_PATH = DATA_DIR / "articles.tsv"
LNK_PATH = DATA_DIR / "links.tsv"


def sample_data():
    """
    Small demo graph so the app works out-of-the-box.
    Creates a network of "speedster" characters with connections between them.
    """
    articles = pd.DataFrame({"Article": [
        "Flash", "Quicksilver", "Sonic", "Road Runner", "Dash", "Zoom",
        "Speedy Gonzales", "Jetstream", "Velocity", "Lightning"
    ]})
    links = pd.DataFrame({
        "Source_Article": ["Flash", "Flash", "Quicksilver", "Sonic", "Sonic",
                           "Road Runner", "Dash", "Speedy Gonzales",
                           "Velocity", "Lightning", "Jetstream", "Flash"],
        "Target_Article": ["Quicksilver", "Sonic", "Sonic", "Road Runner", "Dash",
                           "Speedy Gonzales", "Velocity", "Jetstream",
                           "Lightning", "Flash", "Flash", "Speedy Gonzales"]
    })
    return articles, links


def _read_table(path: Path):
    """
    Read TSV files, skipping comment lines starting with #.
    Automatically detects if file has 1 column (articles) or 2 columns (links).
    """
    if not path.exists():
        print(f"Path does not exist: {path}")
        return None
    try:
        # Read all lines and filter out comments
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Total lines in {path.name}: {len(lines)}")

        # Filter out comment lines (starting with #) and empty lines
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

        if not data_lines:
            print(f"No data found in {path.name}, only comments")
            return None

        print(f"First data line: {data_lines[0][:80]}")
        print(f"Second data line: {data_lines[1][:80] if len(data_lines) > 1 else 'N/A'}")

        # Write filtered data to a temporary string and read with pandas
        from io import StringIO
        data_str = ''.join(data_lines)

        # Determine if this is articles (1 column) or links (2 columns) based on tabs in first line
        num_tabs = data_lines[0].count('\t')

        if num_tabs == 0:
            # Single column file (articles) - no header row in file
            df = pd.read_csv(StringIO(data_str), sep="\t", header=None, names=['Article'], engine="python",
                             encoding='utf-8')
        else:
            # Two column file (links) - no header row in file
            df = pd.read_csv(StringIO(data_str), sep="\t", header=None, names=['Source_Article', 'Target_Article'],
                             engine="python", encoding='utf-8')

        # Strip whitespace from column names and values
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        print(f"Successfully loaded {path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")

        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------- Graph Building & Analysis Helpers ----------

def build_graph(articles: pd.DataFrame, links: pd.DataFrame, directed: bool):
    """
    Build a NetworkX graph from articles and links dataframes.
    Can create either directed or undirected graphs based on user selection.
    """
    # Create directed or undirected graph based on user choice
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes from articles dataframe
    if "Article" in articles.columns:
        nodes = articles["Article"].dropna().astype(str).tolist()
        G.add_nodes_from(nodes)
        print(f"Added {len(nodes)} nodes from articles")

    # Add edges from links dataframe
    if "Source_Article" in links.columns and "Target_Article" in links.columns:
        edge_count = 0
        for a, b in links[["Source_Article", "Target_Article"]].dropna().astype(str).itertuples(index=False):
            if a and b and a != "nan" and b != "nan":
                G.add_edge(a, b)
                edge_count += 1
        print(f"Added {edge_count} edges from links")
    else:
        print(f"Links columns: {links.columns.tolist()}")

    return G


def compute_metrics(G: nx.Graph):
    """
    Compute network centrality metrics for all nodes.
    Returns a dataframe with degree, centrality, betweenness, closeness, and clustering.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "degree", "degree_centrality", "betweenness", "closeness", "clustering"])

    # Use undirected version for stability in metric calculations
    GU = G.to_undirected()

    # Calculate various centrality metrics
    deg = dict(G.degree())  # Number of connections per node
    cdeg = nx.degree_centrality(GU)  # Normalized degree
    btw = nx.betweenness_centrality(GU, normalized=True)  # Bridge importance
    close = nx.closeness_centrality(GU)  # Average distance to other nodes
    clust = nx.clustering(GU)  # How connected neighbors are

    return pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [deg.get(n, 0) for n in G.nodes()],
        "degree_centrality": [cdeg.get(n, 0) for n in G.nodes()],
        "betweenness": [btw.get(n, 0) for n in G.nodes()],
        "closeness": [close.get(n, 0) for n in G.nodes()],
        "clustering": [clust.get(n, 0) for n in G.nodes()],
    })


def detect_communities(G: nx.Graph):
    """
    Detect communities (clusters) in the graph using greedy modularity.
    Returns a dictionary mapping node names to community IDs.
    """
    if G.number_of_nodes() == 0:
        return {}
    GU = G.to_undirected()
    # Find communities using modularity optimization
    comms = list(nx.algorithms.community.greedy_modularity_communities(GU)) if GU.number_of_nodes() else []
    cmap = {}
    for cid, cset in enumerate(comms):
        for n in cset:
            cmap[n] = cid
    return cmap


def layout_coords(G: nx.Graph, algo: str):
    """
    Calculate node positions for visualization using different layout algorithms.
    Each algorithm emphasizes different graph properties.
    """
    if G.number_of_nodes() == 0:
        return {}
    if algo == "spring":
        # Force-directed layout - nodes repel, edges attract
        return nx.spring_layout(G, seed=42)
    if algo == "kamada-kawai":
        # Energy minimization - tries to make edge lengths uniform
        return nx.kamada_kawai_layout(G)
    if algo == "circular":
        # Nodes arranged in a circle
        return nx.circular_layout(G)
    if algo == "spectral":
        # Uses graph Laplacian eigenvectors
        return nx.spectral_layout(G)
    return nx.spring_layout(G, seed=42)


# ---------- User Interface ----------

app_ui = ui.page_fluid(
    ui.h2("WikiSpeedsters — Interactive Graph (Python Shiny)"),
    ui.row(
        # Left sidebar with controls
        ui.column(3,
                  # Data upload section
                  ui.card(
                      ui.card_header("Data"),
                      ui.input_file("articles_file", "Upload Articles TSV/CSV", accept=[".tsv", ".csv"]),
                      ui.input_file("links_file", "Upload Links TSV/CSV", accept=[".tsv", ".csv"]),
                      ui.div({"class": "text-muted small"},
                             "Tip: If ./data/articles.tsv and ./data/links.tsv exist, the app will load them automatically."),
                      ui.output_text_verbatim("load_status", placeholder=True)
                  ),
                  # Interactive controls for filtering and visualization
                  ui.card(
                      ui.card_header("Controls"),
                      ui.input_switch("directed", "Directed graph", value=True),
                      # Filter controls to reduce graph size for performance
                      ui.input_slider("max_nodes", "Maximum nodes to display", 50, 1000, 200, step=50),
                      ui.input_select("node_filter", "Show nodes by",
                                      {"top_degree": "Top by degree",
                                       "top_betweenness": "Top by betweenness",
                                       "min_degree": "Minimum degree filter"}),
                      ui.input_slider("min_degree", "Min degree threshold", 0, 50, 5, step=1),
                      # Visualization controls
                      ui.input_select("layout_algo", "Layout",
                                      ["spring", "kamada-kawai", "circular", "spectral"]),
                      ui.input_select("color_by", "Color by",
                                      ["community", "degree", "degree_centrality", "betweenness", "closeness",
                                       "clustering"]),
                      ui.input_select("size_by", "Node size by",
                                      ["degree", "degree_centrality", "betweenness", "closeness", "clustering"])
                  )
                  ),
        # Main content area with visualization and table
        ui.column(9,
                  ui.layout_column_wrap(
                      # Interactive Plotly graph with zoom/pan capabilities
                      ui.card(
                          ui.card_header("Graph View (Plotly)"),
                          output_widget("graph_plot", height="520px")
                      ),
                      # Dynamic table showing top nodes by selected metric
                      ui.card(
                          ui.card_header("Top Nodes by Metric"),
                          ui.output_data_frame("metric_table")
                      )
                  )
                  )
    ),
    # Bottom section with summary statistics
    ui.row(
        ui.column(12,
                  ui.card(
                      ui.card_header("Quick Stats"),
                      ui.output_text_verbatim("stats_text")
                  )
                  )
    ),
    theme=theme.materia()  # Apply Material Design theme
)


# ---------- Server Logic ----------

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def data_frames():
        """
        Reactive function that loads data with the following priority:
        1) User uploads (if provided)
        2) Local ./data/articles.tsv and ./data/links.tsv (if present)
        3) Built-in sample data
        """
        art = None
        lnk = None
        source = "sample"

        # Check for user uploads first
        if input.articles_file():
            fa = Path(input.articles_file()[0]['datapath'])
            # Determine separator based on file extension
            sep_a = "\t" if fa.suffix.lower() == ".tsv" else ","
            art = pd.read_csv(fa, sep=sep_a)
            art.columns = art.columns.str.strip()
            source = "uploaded articles"

        if input.links_file():
            fl = Path(input.links_file()[0]['datapath'])
            # Determine separator based on file extension
            sep_l = "\t" if fl.suffix.lower() == ".tsv" else ","
            lnk = pd.read_csv(fl, sep=sep_l)
            lnk.columns = lnk.columns.str.strip()
            if source == "sample":
                source = "uploaded links"
            elif source == "uploaded articles":
                source = "uploaded"

        # If no uploads, try local TSV files
        if art is None:
            print(f"Looking for articles at: {ART_PATH}")
            print(f"File exists: {ART_PATH.exists()}")
            art = _read_table(ART_PATH)
            if art is not None:
                source = "local data/articles.tsv"

        if lnk is None:
            print(f"Looking for links at: {LNK_PATH}")
            print(f"File exists: {LNK_PATH.exists()}")
            lnk = _read_table(LNK_PATH)
            if lnk is not None and source == "sample":
                source = "local data/links.tsv"
            elif lnk is not None and "local" in source:
                source = "local data files"

        # Fallback to sample if still None
        if art is None or lnk is None:
            art_sample, lnk_sample = sample_data()
            if art is None:
                print("Using sample articles data")
                art = art_sample
            if lnk is None:
                print("Using sample links data")
                lnk = lnk_sample
            source = "sample data"

        return art, lnk, source

    @reactive.Calc
    def graph_and_metrics():
        """
        Reactive function that builds the graph and computes metrics.
        Applies user-selected filters to reduce graph size for performance.
        """
        art, lnk, _ = data_frames()
        G = build_graph(art, lnk, input.directed())

        # Apply filtering strategy based on user selection
        filter_method = input.node_filter()
        max_nodes = int(input.max_nodes())
        min_deg = int(input.min_degree())

        if G.number_of_nodes() == 0:
            return G, pd.DataFrame(
                columns=["node", "degree", "degree_centrality", "betweenness", "closeness", "clustering", "community"])

        if filter_method == "min_degree":
            # Filter by minimum degree threshold, then limit to max_nodes
            if min_deg > 0:
                keep = [n for n, d in G.degree() if d >= min_deg]
                G = G.subgraph(keep).copy()
                # Further limit if still too many nodes
                if G.number_of_nodes() > max_nodes:
                    degrees = dict(G.degree())
                    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                    G = G.subgraph([n for n, _ in top_nodes]).copy()

        elif filter_method == "top_degree":
            # Keep only the top N most connected nodes (fastest method)
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        elif filter_method == "top_betweenness":
            # Keep top N nodes by betweenness centrality (identifies bridge nodes)
            GU = G.to_undirected()
            btw = nx.betweenness_centrality(GU, normalized=True)
            top_nodes = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        # Compute metrics and detect communities
        metrics = compute_metrics(G)
        if len(metrics) > 0:
            metrics["community"] = [detect_communities(G).get(n, -1) for n in metrics["node"]]
        return G, metrics

    @output
    @render.text
    def load_status():
        """Display which data source is currently being used."""
        _, _, source = data_frames()
        return f"Data source: {source}"

    @output
    @render_widget
    def graph_plot():
        """
        Create interactive Plotly graph visualization with:
        - Zoom and pan capabilities
        - Color coding by selected metric or community
        - Node sizing by selected metric
        - Hover tooltips showing node details
        """
        G, metrics = graph_and_metrics()
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title="No nodes to display")
            return fig

        # Calculate node positions using selected layout algorithm
        pos = layout_coords(G, input.layout_algo())

        # Create edge traces (lines connecting nodes)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            xa, ya = pos[a]
            xb, yb = pos[b]
            edge_x += [xa, xb, None]  # None creates a break in the line
            edge_y += [ya, yb, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            hoverinfo="none", line=dict(width=1), opacity=0.5
        )

        # Add position coordinates to metrics dataframe
        m2 = metrics.copy()
        m2["x"] = m2["node"].map(lambda n: pos[n][0])
        m2["y"] = m2["node"].map(lambda n: pos[n][1])

        # Apply user-selected color and size mappings
        color_by = input.color_by()
        size_by = input.size_by()
        svals = m2[size_by]
        # Scale node sizes between 10 and 40 based on metric values
        sizes = [20] * len(svals) if svals.max() == svals.min() else \
            (10 + 30 * (svals - svals.min()) / (svals.max() - svals.min()))

        # Create node trace with hover information
        node_trace = px.scatter(
            m2, x="x", y="y",
            hover_name="node",
            hover_data=["degree", "degree_centrality", "betweenness", "closeness", "clustering", "community"],
            color=color_by, size=sizes
        ).data[0]

        # Combine edges and nodes into final figure
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=True, margin=dict(l=10, r=10, t=10, b=10),
            dragmode="pan"  # Enable panning by default
        )
        return fig

    @output
    @render.data_frame
    def metric_table():
        """
        Display dynamic table of top 15 nodes ranked by the selected size metric.
        Updates automatically when user changes the "Node size by" dropdown.
        """
        _, metrics = graph_and_metrics()
        if len(metrics) == 0:
            return pd.DataFrame()
        col = input.size_by()
        return metrics.sort_values(col, ascending=False).head(15).reset_index(drop=True)

    @output
    @render.text
    def stats_text():
        """Display summary statistics about the current graph."""
        G, _ = graph_and_metrics()
        if G.number_of_nodes() == 0:
            return "No graph loaded."
        GU = G.to_undirected()
        dens = nx.density(GU)  # How connected the graph is (0-1)
        comps = nx.number_connected_components(GU)  # Number of disconnected groups
        return f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Density: {dens:.4f} | Connected components: {comps}"


# Create and run the Shiny app
app = App(app_ui, server)