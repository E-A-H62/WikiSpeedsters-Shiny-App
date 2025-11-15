from shiny import App, Inputs, Outputs, Session, ui, render, reactive
from shinyswatch import theme
from shinywidgets import output_widget, render_widget
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------- Helpers ----------

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"  # <-- will look here automatically
ART_PATH = DATA_DIR / "articles.tsv"
LNK_PATH = DATA_DIR / "links.tsv"

def sample_data():
    """Small demo graph so the app works out-of-the-box."""
    articles = pd.DataFrame({"Article": [
        "Flash","Quicksilver","Sonic","Road Runner","Dash","Zoom",
        "Speedy Gonzales","Jetstream","Velocity","Lightning"
    ]})
    links = pd.DataFrame({
        "Source_Article": ["Flash","Flash","Quicksilver","Sonic","Sonic",
                           "Road Runner","Dash","Speedy Gonzales",
                           "Velocity","Lightning","Jetstream","Flash"],
        "Target_Article": ["Quicksilver","Sonic","Sonic","Road Runner","Dash",
                           "Speedy Gonzales","Velocity","Jetstream",
                           "Lightning","Flash","Flash","Speedy Gonzales"]
    })
    return articles, links

def _read_table(path: Path):
    """Always read TSV files with tab separator."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", comment="#", engine="python")
        # If your original notebook skipped header lines (e.g., skiprows=12), uncomment the next line:
        # df = pd.read_csv(path, sep="\t", skiprows=12, comment="#", engine="python")
        df = df.rename(columns={c.strip(): c.strip() for c in df.columns})
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def build_graph(articles: pd.DataFrame, links: pd.DataFrame, directed: bool):
    """Build a NetworkX (Di)Graph."""
    G = nx.DiGraph() if directed else nx.Graph()
    if "Article" in articles.columns:
        G.add_nodes_from(articles["Article"].dropna().astype(str).tolist())
    if {"Source_Article","Target_Article"}.issubset(links.columns):
        for a, b in links[["Source_Article","Target_Article"]].dropna().astype(str).itertuples(index=False):
            if a and b and a != "nan" and b != "nan":
                G.add_edge(a, b)
    return G

def compute_metrics(G: nx.Graph):
    """Compute centralities & clustering on undirected view for stability."""
    GU = G.to_undirected()
    deg = dict(G.degree())
    cdeg = nx.degree_centrality(GU)
    btw = nx.betweenness_centrality(GU, normalized=True)
    close = nx.closeness_centrality(GU)
    clust = nx.clustering(GU)
    return pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [deg.get(n,0) for n in G.nodes()],
        "degree_centrality": [cdeg.get(n,0) for n in G.nodes()],
        "betweenness": [btw.get(n,0) for n in G.nodes()],
        "closeness": [close.get(n,0) for n in G.nodes()],
        "clustering": [clust.get(n,0) for n in G.nodes()],
    })

def detect_communities(G: nx.Graph):
    """Greedy modularity communities on undirected view."""
    GU = G.to_undirected()
    comms = list(nx.algorithms.community.greedy_modularity_communities(GU)) if GU.number_of_nodes() else []
    cmap = {}
    for cid, cset in enumerate(comms):
        for n in cset:
            cmap[n] = cid
    return cmap

def layout_coords(G: nx.Graph, algo: str):
    """Return layout positions for nodes."""
    if G.number_of_nodes() == 0:
        return {}
    if algo == "spring":
        return nx.spring_layout(G, seed=42)
    if algo == "kamada-kawai":
        return nx.kamada_kawai_layout(G)
    if algo == "circular":
        return nx.circular_layout(G)
    if algo == "spectral":
        return nx.spectral_layout(G)
    return nx.spring_layout(G, seed=42)

# ---------- UI ----------

app_ui = ui.page_fluid(
    ui.h2("WikiSpeedsters — Interactive Graph (Python Shiny)"),
    ui.row(
        ui.column(3,
            ui.card(
                ui.card_header("Data"),
                ui.input_file("articles_file", "Upload Articles TSV/CSV", accept=[".tsv",".csv"]),
                ui.input_file("links_file", "Upload Links TSV/CSV", accept=[".tsv",".csv"]),
                ui.div({"class":"text-muted small"},
                       "Tip: If ./data/articles.tsv and ./data/links.tsv exist, the app will load them automatically.")
            ),
            ui.card(
                ui.card_header("Controls"),
                ui.input_switch("directed", "Directed graph", value=True),
                ui.input_select("layout_algo", "Layout",
                                ["spring","kamada-kawai","circular","spectral"]),
                ui.input_select("color_by", "Color by",
                                ["community","degree","degree_centrality","betweenness","closeness","clustering"]),
                ui.input_slider("min_degree", "Filter: minimum degree", 0, 10, 0, step=1),
                ui.input_select("size_by", "Node size by",
                                ["degree","degree_centrality","betweenness","closeness","clustering"])
            )
        ),
        ui.column(9,
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Graph View (Plotly)"),
                    output_widget("graph_plot", height="520px")
                ),
                ui.card(
                    ui.card_header("Top Nodes by Metric"),
                    ui.output_data_frame("metric_table")
                )
            )
        )
    ),
    ui.row(
        ui.column(12,
            ui.card(
                ui.card_header("Quick Stats"),
                ui.output_text_verbatim("stats_text")
            )
        )
    ),
    theme=theme.materia()
)

# ---------- Server ----------

def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Calc
    def data_frames():
        """
        Load precedence:
        1) User uploads (if provided)
        2) ./data/articles.tsv and ./data/links.tsv (if present)
        3) Built-in sample
        """
        # 1) uploads?
        if input.articles_file() or input.links_file():
            if input.articles_file():
                fa = Path(input.articles_file()[0]['datapath'])
                sep_a = "\t" if fa.suffix.lower() == ".tsv" else None
                art = pd.read_csv(fa, sep=sep_a)
            else:
                # if only links uploaded, still try local or sample for articles
                art = _read_table(ART_PATH) or sample_data()[0]

            if input.links_file():
                fl = Path(input.links_file()[0]['datapath'])
                sep_l = "\t" if fl.suffix.lower() == ".tsv" else None
                lnk = pd.read_csv(fl, sep=sep_l)
            else:
                lnk = _read_table(LNK_PATH) or sample_data()[1]
        else:
            # 2) local data/ fallback to 3) sample
            art = _read_table(ART_PATH)
            lnk = _read_table(LNK_PATH)
            if art is None or lnk is None:
                art, lnk = sample_data()

        # normalize column names
        art = art.rename(columns={c: c.strip() for c in art.columns})
        lnk = lnk.rename(columns={c: c.strip() for c in lnk.columns})
        return art, lnk

    @reactive.Calc
    def graph_and_metrics():
        """Build graph + metrics + community labels, filter by min degree."""
        art, lnk = data_frames()
        G = build_graph(art, lnk, input.directed())
        md = int(input.min_degree())
        if md > 0 and G.number_of_nodes():
            keep = [n for n, d in G.degree() if d >= md]
            G = G.subgraph(keep).copy()
        metrics = compute_metrics(G)
        metrics["community"] = [detect_communities(G).get(n, -1) for n in metrics["node"]]
        return G, metrics

    @output
    @render_widget
    def graph_plot():
        """Interactive Plotly graph (zoom/pan)."""
        G, metrics = graph_and_metrics()
        if G.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title="No nodes to display")
            return fig

        pos = layout_coords(G, input.layout_algo())
        # edges
        edge_x, edge_y = [], []
        for a, b in G.edges():
            xa, ya = pos[a]
            xb, yb = pos[b]
            edge_x += [xa, xb, None]
            edge_y += [ya, yb, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            hoverinfo="none", line=dict(width=1), opacity=0.5
        )

        m2 = metrics.copy()
        m2["x"] = m2["node"].map(lambda n: pos[n][0])
        m2["y"] = m2["node"].map(lambda n: pos[n][1])

        color_by = input.color_by()
        size_by = input.size_by()
        svals = m2[size_by]
        sizes = [20]*len(svals) if svals.max() == svals.min() else \
                (10 + 30*(svals - svals.min())/(svals.max()-svals.min()))

        node_trace = px.scatter(
            m2, x="x", y="y",
            hover_name="node",
            hover_data=["degree","degree_centrality","betweenness","closeness","clustering","community"],
            color=color_by, size=sizes
        ).data[0]

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=True, margin=dict(l=10,r=10,t=10,b=10),
            dragmode="pan"
        )
        return fig

    @output
    @render.data_frame
    def metric_table():
        """Dynamic table of top nodes by selected metric."""
        _, metrics = graph_and_metrics()
        col = input.size_by()
        return metrics.sort_values(col, ascending=False).head(15).reset_index(drop=True)

    @output
    @render.text
    def stats_text():
        """Quick graph stats."""
        G, _ = graph_and_metrics()
        if G.number_of_nodes() == 0:
            return "No graph loaded."
        GU = G.to_undirected()
        dens = nx.density(GU)
        comps = nx.number_connected_components(GU)
        return f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Density: {dens:.4f} | Connected components: {comps}"

app = App(app_ui, server)
