from shiny import App, Inputs, Outputs, Session, ui, render, reactive
from shinyswatch import theme
from shinywidgets import output_widget, render_widget
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import random

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


def compute_modularity(G: nx.Graph, communities_dict: dict):
    """
    Compute modularity Q for the given graph and community assignment.
    Returns modularity score (float).
    """
    if G.number_of_nodes() == 0 or not communities_dict:
        return 0.0

    GU = G.to_undirected()

    # Convert dict to list of sets (format required by nx.modularity)
    comm_ids = set(communities_dict.values())
    communities = [set([n for n, c in communities_dict.items() if c == cid]) for cid in comm_ids]

    return nx.algorithms.community.modularity(GU, communities)


def run_si_diffusion(G: nx.Graph, seed_node: str, max_steps: int):
    """
    Run SI (Susceptible-Infected) diffusion model.

    Args:
        G: NetworkX graph
        seed_node: Starting infected node
        max_steps: Maximum number of diffusion steps

    Returns:
        dict: {node: infection_time}, where uninfected nodes have time = -1
    """
    if seed_node not in G.nodes():
        return {n: -1 for n in G.nodes()}

    # Use undirected version for diffusion
    GU = G.to_undirected()

    # Track infection times
    infection_time = {n: -1 for n in GU.nodes()}
    infection_time[seed_node] = 0

    # Track newly infected nodes at each step
    newly_infected = {seed_node}

    for step in range(1, max_steps + 1):
        # Find all neighbors of newly infected nodes
        next_infected = set()
        for node in newly_infected:
            for neighbor in GU.neighbors(node):
                if infection_time[neighbor] == -1:  # Not yet infected
                    infection_time[neighbor] = step
                    next_infected.add(neighbor)

        # If no new infections, stop early
        if not next_infected:
            break

        newly_infected = next_infected

    return infection_time


def check_reachability(G: nx.Graph, seed_node: str):
    """
    Check if all nodes are reachable from the seed node.

    Returns:
        (bool, int): (all_reachable, num_unreachable)
    """
    if seed_node not in G.nodes():
        return False, G.number_of_nodes()

    GU = G.to_undirected()

    # Find all nodes reachable from seed
    reachable = nx.node_connected_component(GU, seed_node)
    num_unreachable = G.number_of_nodes() - len(reachable)

    return num_unreachable == 0, num_unreachable


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
                                      ["degree", "degree_centrality", "betweenness", "closeness",
                                       "clustering", "diffusion_time"]),
                      ui.input_select("size_by", "Node size by",
                                      ["degree", "degree_centrality", "betweenness", "closeness", "clustering"]),

                      ui.input_select("degree_type", "Histogram degree type",
                                      {"total": "Total degree", "in": "In-degree", "out": "Out-degree"},
                                      selected="total")
                  ),

                  # Graph Density Controls
                  ui.card(
                      ui.card_header("Graph Density Controls"),
                      ui.input_slider("edge_sample_pct", "Show % of edges", 10, 100, 100, step=5),
                      ui.input_select("community_filter", "Filter by community",
                                      choices={"all": "All communities"}),
                      ui.div({"class": "text-muted small"},
                             "Reduce edge density or filter to specific communities for clarity.")
                  ),

                  # Diffusion Parameters
                  ui.card(
                      ui.card_header("Diffusion"),
                      ui.input_select("seed_node", "Seed node", choices=[]),
                      ui.input_slider("diffusion_steps", "Diffusion steps", 1, 20, 5, step=1),
                      ui.input_switch("random_seed", "Random seed", value=False),
                      ui.input_action_button("run_diffusion", "Run diffusion", class_="btn-primary"),
                      ui.div(
                          {"class": "text-muted small mt-1"},
                          "To visualize diffusion, first set 'Color by' to 'diffusion_time', "
                          "then choose a seed node and click 'Run diffusion'."
                      ),
                      ui.output_text("diffusion_warning"),
                  )
                  ),

        # Main content area with visualization and table
        ui.column(9,
                  # First row: Graph and Table side by side
                  ui.layout_column_wrap(
                      # Interactive Plotly graph with zoom/pan capabilities
                      ui.card(
                          ui.card_header("Graph View (Plotly)"),
                          output_widget("graph_plot", height="520px"),
                          ui.div(
                              {"class": "text-muted small mt-2"},
                              ui.output_text("graph_caption")
                          ),
                          ui.div(
                              {"class": "text-muted small mt-1"},
                              ui.output_text("text_legend")
                          )
                      ),
                      # Dynamic table showing top nodes by selected metric
                      ui.card(
                          ui.card_header("Top Nodes by Metric"),
                          ui.output_data_frame("metric_table")
                      ),
                  ),

                  # Second row: Histogram spans full width below
                  ui.card(
                      ui.card_header("Degree Distribution Histogram"),
                      ui.div(
                          {"class": "text-muted small mb-1"},
                          "How to use: Shows the degree distribution for all nodes in the current graph. "
                          "Use the 'Histogram degree type' dropdown on the left to switch "
                          "between total degree, in-degree, or out-degree."
                      ),
                      output_widget("degree_histogram", height="350px"),
                  ),
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
    # Reactive value to store diffusion results
    diffusion_results = reactive.Value({})

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

        # Check for user uploads first - use same parsing logic as _read_table
        if input.articles_file():
            fa = Path(input.articles_file()[0]['datapath'])
            art = _read_table(fa)  # Use our robust parser
            if art is not None:
                source = "uploaded articles"

        if input.links_file():
            fl = Path(input.links_file()[0]['datapath'])
            lnk = _read_table(fl)  # Use our robust parser
            if lnk is not None:
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
        G_full = build_graph(art, lnk, input.directed())

        # Apply filtering strategy based on user selection
        try:
            filter_method = input.node_filter()
            max_nodes = int(input.max_nodes())
            min_deg = int(input.min_degree())
            edge_sample = int(input.edge_sample_pct())
            community_filt = input.community_filter()
        except (TypeError, ValueError):
            filter_method = "top_degree"
            max_nodes = 200
            min_deg = 5
            edge_sample = 100
            community_filt = "all"

        if G_full.number_of_nodes() == 0:
            return G_full, pd.DataFrame(
                columns=["node", "degree", "degree_centrality", "betweenness", "closeness", "clustering",
                         "community"]), {}

        # STEP 1: Detect communities on the FULL graph (before any filtering)
        # This ensures we have complete community info for the dropdown
        communities_dict_full = detect_communities(G_full)

        # STEP 2: Apply community filter FIRST (if selected)
        G = G_full
        if community_filt != "all":
            try:
                target_comm = int(community_filt)
                keep_nodes = [n for n, c in communities_dict_full.items() if c == target_comm]
                if keep_nodes:
                    G = G.subgraph(keep_nodes).copy()
            except ValueError:
                pass

        # STEP 3: Apply node count filtering
        if filter_method == "min_degree":
            if min_deg > 0:
                keep = [n for n, d in G.degree() if d >= min_deg]
                G = G.subgraph(keep).copy()
                if G.number_of_nodes() > max_nodes:
                    degrees = dict(G.degree())
                    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                    G = G.subgraph([n for n, _ in top_nodes]).copy()

        elif filter_method == "top_degree":
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        elif filter_method == "top_betweenness":
            GU = G.to_undirected()
            btw = nx.betweenness_centrality(GU, normalized=True)
            top_nodes = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        # STEP 4: Detect communities on the FILTERED graph for display
        communities_dict = detect_communities(G)

        # Compute metrics
        metrics = compute_metrics(G)

        if len(metrics) > 0:
            metrics["community"] = [communities_dict.get(n, -1) for n in metrics["node"]]

        # STEP 5: Edge sampling
        if edge_sample < 100 and G.number_of_edges() > 0:
            num_edges_keep = max(1, int(G.number_of_edges() * edge_sample / 100))
            edges_list = list(G.edges())
            random.seed(42)
            sampled_edges = random.sample(edges_list, num_edges_keep)
            G_new = nx.DiGraph() if input.directed() else nx.Graph()
            G_new.add_nodes_from(G.nodes())
            G_new.add_edges_from(sampled_edges)
            G = G_new

        # Return communities from FULL graph so dropdown stays populated
        return G, metrics, communities_dict_full

    # Update seed node dropdown and community filter when graph changes
    @reactive.Effect
    def _():
        G, metrics, communities_dict_full = graph_and_metrics()

        # Update seed node choices
        if G.number_of_nodes() > 0:
            node_list = sorted(G.nodes())
            choices = {n: n for n in node_list}
            ui.update_select("seed_node", choices=choices)

        # Update community filter choices based on FULL graph communities
        # This keeps all communities available even when filtering
        if communities_dict_full:
            communities = sorted(set(communities_dict_full.values()))
            comm_choices = {"all": "All communities"}
            comm_choices.update({str(c): f"Community {c}" for c in communities})

            # Only update if choices have changed (prevents reset loop)
            current_choice = input.community_filter()
            ui.update_select("community_filter", choices=comm_choices, selected=current_choice)

    # Run diffusion when button is clicked
    @reactive.Effect
    @reactive.event(input.run_diffusion)
    def _():
        G, _, _ = graph_and_metrics()

        if G.number_of_nodes() == 0:
            return

        # Determine seed node
        if input.random_seed():
            seed = random.choice(list(G.nodes()))
        else:
            seed = input.seed_node()
            if not seed or seed not in G.nodes():
                seed = list(G.nodes())[0]

        # Run diffusion
        infection_times = run_si_diffusion(G, seed, input.diffusion_steps())
        diffusion_results.set(infection_times)

    @output
    @render.text
    def load_status():
        """Display which data source is currently being used."""
        _, _, source = data_frames()
        return f"Data source: {source}"

    @output
    @render.text
    def diffusion_warning():
        """Display warning if seed node cannot reach all nodes."""
        G, _, _ = graph_and_metrics()

        if G.number_of_nodes() == 0:
            return ""

        seed = input.seed_node()
        if not seed or seed not in G.nodes():
            return ""

        all_reachable, num_unreachable = check_reachability(G, seed)

        if not all_reachable:
            return f"⚠️ Warning: {num_unreachable} nodes cannot be reached from '{seed}' due to disconnected components."

        return ""

    @output
    @render.text
    def graph_caption():
        """Short explanation of how to read the current visualization."""
        color_by = input.color_by()
        size_by = input.size_by()
        directed = "directed" if input.directed() else "undirected"

        return (
            "How to read this view: Each node represents a page in the "
            f"{directed} graph. Color shows {color_by}, size reflects {size_by}, "
            f"and edges represent links. Hover over nodes for details."
        )

    @output
    @render.text
    def text_legend():
        """Readable text legend explaining node color, size, and edges."""
        color_by = input.color_by()
        size_by = input.size_by()
        directed = "directed" if input.directed() else "undirected"

        color_label = color_by.replace("_", " ").title()
        size_label = size_by.replace("_", " ").title()

        arrow = "→" if input.directed() else "—"

        legend_text = (
            f"Legend:\n"
            f"• Node color: {color_label}"
        )

        if color_by == "diffusion_time":
            legend_text += " (green = source article, yellow = farthest infected, purple = infected earliest)"

        legend_text += f"\n• Node size: {size_label}\n• Edge {arrow}: link between pages"

        return legend_text

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
        G, metrics, _ = graph_and_metrics()
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title="No nodes to display")
            return fig

        # Calculate node positions using selected layout algorithm
        pos = layout_coords(G, input.layout_algo())

        color_by = input.color_by()
        size_by = input.size_by()

        # Create edge traces (lines connecting nodes)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            xa, ya = pos[a]
            xb, yb = pos[b]
            edge_x += [xa, xb, None]
            edge_y += [ya, yb, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none",
            line=dict(width=1),
            opacity=0.5,
            name="Links between pages",
            showlegend=True,
        )

        # Add position coordinates to metrics dataframe
        m2 = metrics.copy()
        m2["x"] = m2["node"].map(lambda n: pos[n][0])
        m2["y"] = m2["node"].map(lambda n: pos[n][1])

        diff_results = diffusion_results.get()
        if diff_results and color_by == "diffusion_time":
            m2["diffusion_time"] = m2["node"].map(lambda n: diff_results.get(n, -1))
            # Mark source node with special value for distinct coloring
            m2["is_source"] = m2["diffusion_time"] == 0

        # Apply user-selected color and size mappings
        svals = m2[size_by]
        sizes = (
            [20] * len(svals)
            if svals.max() == svals.min()
            else (10 + 30 * (svals - svals.min()) / (svals.max() - svals.min()))
        )

        # Create hover data
        hover_cols = ["degree", "degree_centrality", "betweenness", "closeness", "clustering", "community"]
        if "diffusion_time" in m2.columns:
            hover_cols.append("diffusion_time")

        # Create node trace with hover information
        if color_by == "diffusion_time" and "is_source" in m2.columns:
            # Split into source and non-source nodes for distinct coloring
            source_nodes = m2[m2["is_source"] == True]
            other_nodes = m2[m2["is_source"] == False]

            traces = [edge_trace]

            # Non-source nodes (regular diffusion coloring)
            if len(other_nodes) > 0:
                other_sizes = [sizes[i] for i in other_nodes.index]
                node_trace_others = px.scatter(
                    other_nodes,
                    x="x",
                    y="y",
                    hover_name="node",
                    hover_data=hover_cols,
                    color="diffusion_time",
                    size=other_sizes,
                ).data[0]
                node_trace_others.name = "Infected nodes"
                node_trace_others.showlegend = True
                if getattr(node_trace_others, "marker", None) and getattr(node_trace_others.marker, "colorbar", None):
                    node_trace_others.marker.colorbar.title = "Steps from Source"
                traces.append(node_trace_others)

            # Source node (distinct bright color - orange/red)
            if len(source_nodes) > 0:
                source_sizes = [sizes[i] for i in source_nodes.index]
                node_trace_source = go.Scatter(
                    x=source_nodes["x"],
                    y=source_nodes["y"],
                    mode="markers",
                    marker=dict(
                        size=[s * 1.3 for s in source_sizes],  # Make slightly larger
                        color="rgb(0, 255, 0)",  # Bright Green
                        line=dict(width=2, color="white")  # White border for extra emphasis
                    ),
                    text=source_nodes["node"],
                    hovertemplate="<b>SOURCE: %{text}</b><br>" +
                                  "<br>".join([f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]) +
                                  "<extra></extra>",
                    customdata=source_nodes[hover_cols].values,
                    name="Source article",
                    showlegend=True
                )
                traces.append(node_trace_source)

            fig = go.Figure(data=traces)
        else:
            # Regular coloring for non-diffusion views
            node_trace = px.scatter(
                m2,
                x="x",
                y="y",
                hover_name="node",
                hover_data=hover_cols,
                color=color_by if color_by in m2.columns else "community",
                size=sizes,
            ).data[0]

            node_trace.name = f"Pages (color: {color_by}, size: {size_by})"
            node_trace.showlegend = True

            # Optional: give colorbar a nicer title for continuous metrics
            if getattr(node_trace, "marker", None) and getattr(node_trace.marker, "colorbar", None):
                node_trace.marker.colorbar.title = color_by.replace("_", " ").title()

            fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                itemclick=False,
                itemdoubleclick=False,
            ),
            margin=dict(l=10, r=10, t=10, b=60),
            dragmode="pan",
        )
        return fig

    @output
    @render_widget
    def degree_histogram():
        """
        Each bar shows how many nodes have a given degree.
        Use the 'Histogram degree type' control on the left
        to switch between total, in-, and out-degree.
        """
        art, lnk, _ = data_frames()
        G_full = build_graph(art, lnk, input.directed())

        # no graph = show an empty frame
        if G_full.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(
                title="No nodes to display",
                xaxis_title="Degree",
                yaxis_title="Count of Nodes",
            )
            return fig

        # "total", "in", or "out", which?
        degree_mode = input.degree_type()

        # compute degrees for each node
        if G_full.is_directed():
            if degree_mode == "in":
                degree_dict = dict(G_full.in_degree())
            elif degree_mode == "out":
                degree_dict = dict(G_full.out_degree())
            else:
                # "total" degree = in-degree + out-degree
                degree_dict = {
                    n: G_full.in_degree(n) + G_full.out_degree(n)
                    for n in G_full.nodes()
                }
        else:
            degree_dict = dict(G_full.degree())

        df = pd.DataFrame({"degree": list(degree_dict.values())})

        mode_label = {
            "total": "Total degree",
            "in": "In-degree",
            "out": "Out-degree",
        }.get(degree_mode, "Degree")

        # Plot histogram
        fig = px.histogram(df, x="degree", nbins=40)

        fig.update_layout(
            title=f"{mode_label} distribution",
            xaxis_title="Degree (k)",
            yaxis_title="Count of Nodes",
            bargap=0.1,
            margin=dict(l=40, r=10, t=40, b=40),
        )

        return fig

    @output
    @render.data_frame
    def metric_table():
        """
        Display dynamic table of top 15 nodes ranked by the selected size metric.
        Updates automatically when user changes the "Node size by" dropdown.
        """
        _, metrics, _ = graph_and_metrics()
        if len(metrics) == 0:
            return pd.DataFrame()
        col = input.size_by()
        return metrics.sort_values(col, ascending=False).head(15).reset_index(drop=True)

    @output
    @render.text
    def stats_text():
        """Display summary statistics about the current graph."""
        G, metrics, communities_dict_full = graph_and_metrics()
        if G.number_of_nodes() == 0:
            return "No graph loaded."

        GU = G.to_undirected()
        dens = nx.density(GU)  # How connected the graph is (0-1)
        comps = nx.number_connected_components(GU)  # Number of disconnected groups

        # FIXED: Detect communities on the FILTERED graph (G), not full graph
        # This ensures all nodes in communities exist in G
        communities_dict_filtered = detect_communities(G)
        modularity_q = compute_modularity(G, communities_dict_filtered)

        return (f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
                f"Density: {dens:.4f} | Connected components: {comps} | "
                f"Modularity Q: {modularity_q:.4f}")



# Create and run the Shiny app
if __name__ == "__main__":
    import shiny

    # Create and run the Shiny app
    app = App(app_ui, server)

    shiny.run_app(app, port=8000, reload=False)  # reload=False avoids any reload quirks