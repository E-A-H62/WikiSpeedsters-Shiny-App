# WikiSpeedsters — Interactive Graph Visualization

This repository contains a Python Shiny application for visualizing and analyzing the WikiSpeedia dataset. The app provides interactive network analysis tools to explore article relationships, compute centrality metrics, detect communities, and simulate information diffusion through the Wikipedia article network.

**Dataset Source:** [WikiSpeedia Dataset](https://snap.stanford.edu/data/wikispeedia.html)

**Source Code:** [Google Colab](https://colab.research.google.com/drive/1xWRNpcRNGNQXUj-2NjVzzITSlm6LanbH?usp=sharing)

## Features

### Interactive Graph Visualization
- **Interactive Plotly Graph**: Zoom, pan, and hover over nodes to explore the network
- **Multiple Layout Algorithms**: Spring force-directed layout for optimal node positioning
- **Customizable Node Appearance**: Color and size nodes by various network metrics
- **Edge Sampling**: Reduce edge density for clearer visualization of large networks

### Network Analysis Metrics
- **Degree Centrality**: Number of connections per node
- **Betweenness Centrality**: Importance as a bridge between other nodes
- **Closeness Centrality**: Average distance to all other nodes
- **Clustering Coefficient**: How connected a node's neighbors are to each other
- **Community Detection**: Automatic detection of node communities using modularity optimization

### Information Diffusion Simulation
- **SI (Susceptible-Infected) Diffusion Model**: Simulate how information spreads through the network
- **Customizable Seed Nodes**: Choose starting points or use random selection
- **Diffusion Visualization**: Color nodes by infection time to see propagation patterns
- **Reachability Analysis**: Check if all nodes are reachable from the seed node

### Data Management
- **Multiple Data Sources**: 
  - Upload custom TSV/CSV files
  - Load from local `./data/` directory
  - Use built-in sample data (speedster characters network)
- **Flexible Data Format**: Automatically handles TSV files with comment lines

### Analysis Tools
- **Degree Distribution Histogram**: Visualize the distribution of node degrees (total, in-degree, or out-degree)
- **Top Nodes Table**: Dynamic table showing top 15 nodes ranked by selected metrics
- **Graph Statistics**: Summary statistics including node count, edge count, density, connected components, and modularity

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Required Packages
Install the required dependencies:

```bash
pip install shiny shinyswatch shinywidgets pandas networkx plotly
```

Or install from a requirements file (if provided):

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Navigate to the application directory:**
   ```bash
   cd WikiSpeedsters-Shiny-App
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Access the application:**
   - The app will start on `http://localhost:8000`
   - Open this URL in your web browser

### Data Setup

The application supports three data loading methods (in priority order):

1. **User Upload**: Upload TSV or CSV files through the web interface
2. **Local Files**: Place your data files in the `./data/` directory:
   - `./data/articles.tsv` - List of article names (one per line)
   - `./data/links.tsv` - List of links (two columns: source and target articles)
3. **Sample Data**: If no data is provided, the app uses a built-in sample network

### Data Format

#### Articles File (`articles.tsv`)
- Single column file with article names
- One article per line
- Comment lines starting with `#` are automatically ignored
- Example:
  ```
  Article
  Flash
  Quicksilver
  Sonic
  ```

#### Links File (`links.tsv`)
- Two-column TSV file with source and target articles
- Format: `Source_Article\tTarget_Article`
- Comment lines starting with `#` are automatically ignored
- Example:
  ```
  Source_Article	Target_Article
  Flash	Quicksilver
  Flash	Sonic
  Quicksilver	Sonic
  ```

## User Interface Guide

### Left Sidebar Controls

#### Data Section
- **Upload Articles TSV/CSV**: Upload your articles file
- **Upload Links TSV/CSV**: Upload your links file
- **Data Source Status**: Shows which data source is currently active

#### Controls Section
- **Directed graph**: Toggle between directed and undirected graph representation
- **Maximum nodes to display**: Limit the number of nodes shown (50-1000, default: 200)
- **Show nodes by**: Filter nodes by:
  - Top by degree
  - Top by betweenness
  - Minimum degree filter
- **Min degree threshold**: Minimum number of connections required (0-50, default: 5)
- **Layout**: Graph layout algorithm (currently: spring)
- **Color by**: Node coloring metric:
  - degree
  - degree_centrality
  - betweenness
  - closeness
  - clustering
  - diffusion_time
- **Node size by**: Node sizing metric (same options as color, excluding diffusion_time)
- **Histogram degree type**: Choose total, in-degree, or out-degree for histogram

#### Graph Density Controls
- **Show % of edges**: Reduce edge density (10-100%, default: 100%)
- **Filter by community**: Select specific communities to display

#### Diffusion Section
- **Seed node**: Select starting node for diffusion simulation
- **Diffusion steps**: Number of propagation steps (1-20, default: 5)
- **Random seed**: Toggle to use a random starting node
- **Run diffusion**: Execute the diffusion simulation

### Main Display Area

#### Graph View (Plotly)
- Interactive network visualization
- Hover over nodes to see detailed metrics
- Zoom and pan to explore different regions
- Color and size encoding based on selected metrics

#### Top Nodes by Metric Table
- Displays top 15 nodes ranked by the selected "Node size by" metric
- Shows all computed metrics for each node
- Updates automatically when filters change

#### Degree Distribution Histogram
- Visualizes the distribution of node degrees
- Switch between total, in-degree, and out-degree views
- Helps identify network structure patterns

#### Quick Stats
- **Nodes**: Total number of nodes in the current graph
- **Edges**: Total number of edges/links
- **Density**: Network density (0-1 scale)
- **Connected components**: Number of disconnected subgraphs
- **Modularity Q**: Community structure quality metric

## How to Use Diffusion Visualization

1. Set **"Color by"** to `diffusion_time` in the Controls section
2. Select a **seed node** from the dropdown (or enable "Random seed")
3. Adjust **diffusion steps** if needed
4. Click **"Run diffusion"**
5. The graph will update showing:
   - **Green nodes**: Source article (seed node)
   - **Yellow/Purple nodes**: Infected articles, colored by infection time
   - **Gray nodes**: Uninfected articles (if any)

## Technical Details

### Network Metrics Explained

- **Degree**: Number of direct connections a node has
- **Degree Centrality**: Normalized degree (0-1 scale)
- **Betweenness Centrality**: Measures how often a node lies on shortest paths between other nodes
- **Closeness Centrality**: Average shortest path distance to all other nodes
- **Clustering Coefficient**: Probability that two neighbors of a node are also connected
- **Modularity**: Measures the strength of community structure (-1 to 1, higher is better)

### Graph Algorithms

- **Community Detection**: Uses greedy modularity maximization algorithm
- **Layout Algorithm**: Spring force-directed layout with fixed seed for reproducibility
- **Diffusion Model**: Susceptible-Infected (SI) model with deterministic propagation

## Citation

When using the WikiSpeedia dataset, please cite:

1. Robert West and Jure Leskovec: *Human Wayfinding in Information Networks*. 21st International World Wide Web Conference (WWW), 2012.

2. Robert West, Joelle Pineau, and Doina Precup: *Wikispeedia: An Online Game for Inferring Semantic Distances between Concepts*. 12th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2013.

## Troubleshooting

### App won't start
- Ensure all dependencies are installed: `pip install shiny shinyswatch shinywidgets pandas networkx plotly`
- Check that Python 3.8+ is being used: `python --version`

### Data not loading
- Verify file paths are correct
- Check that TSV files are properly formatted (tab-separated)
- Ensure comment lines start with `#` if present
- Check console output for error messages

### Graph appears empty
- Increase "Maximum nodes to display" slider
- Lower "Min degree threshold" filter
- Check that your data files contain valid nodes and edges

### Performance issues with large graphs
- Reduce "Maximum nodes to display"
- Lower "Show % of edges" percentage
- Use filtering options to focus on specific communities or high-degree nodes

## Deployment on Render

This application can be deployed on [Render](https://render.com) as a web service. Follow these steps:

### Prerequisites
- A GitHub account
- A Render account (free tier available)
- Your code pushed to a GitHub repository

### Important Notes

- **Port Configuration**: The app automatically uses the `PORT` environment variable provided by Render. No manual configuration needed.
- **Data Files**: If you're using local data files in the `data/` directory, ensure they are committed to your repository.
- **Free Tier Limitations**: 
  - Free services spin down after 15 minutes of inactivity
  - First request after spin-down may take 30-60 seconds to respond
  - Consider upgrading to a paid plan for production use
- **Environment Variables**: You can add custom environment variables in the Render dashboard under "Environment" if needed.

### Troubleshooting Deployment

- **Build Fails**: Check that all dependencies are listed in `requirements.txt`
- **App Won't Start**: Verify the start command is `python app.py`
- **Port Errors**: Ensure the app uses `os.environ.get("PORT", 8000)` for port configuration
- **Data Not Loading**: Verify data files are committed to the repository and paths are correct

### Files Required for Deployment

- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `render.yaml` - Render configuration (optional but recommended)
- `data/` directory - Data files (if using local data)
