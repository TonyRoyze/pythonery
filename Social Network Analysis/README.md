# Social Network Analysis Project

This project contains a comprehensive social network analysis pipeline for analyzing the 2027 social network dataset, implementing all 7 tasks from the assessment brief.

## 📊 Analysis Overview

The complete analysis includes:

1. **Network Overview** - Basic network statistics and structural properties
2. **Centrality Analysis** - Node importance metrics and rankings (Degree, Betweenness, Closeness, Clustering Coefficient)
3. **Community Detection** - User group identification using Louvain and Spectral algorithms
4. **Influence Propagation** - Information spread simulation and analysis using PyTorch Geometric
5. **Machine Learning** - Graph-based link prediction using GraphSAGE, GNN, and traditional ML
6. **Sentiment Analysis** - Text sentiment analysis integrated with network structure
7. **Topology Analysis** - Comprehensive network topology and scalability analysis
8. **Comprehensive Visualizations** - Network plots and statistical charts

## 📁 Project Structure

### Main Analysis Files
- `run_analysis.py` - Complete analysis pipeline script (1167 lines)
- `run_analysis.ipynb` - Jupyter notebook version of the analysis
- `requirements.txt` - Python dependencies

### Data Files
- `2027-comments.json` - Comments data (4.8MB)
- `user_relationships_2027_aggregated.csv` - User relationship data (223KB)
- `user_relationships_2027_raw.csv` - Raw user relationship data (261KB)
- `2027-comments_author_reply_counts.csv` - Reply count data (399KB)

### Analysis Reports
- `analysis_reports/social_network_analysis_report.md` - Main analysis report
- `analysis_reports/tasks.md` - Task implementation details
- `TASK_EXPLANATIONS.md` - Comprehensive task implementation guide

### Outputs Generated
The analysis generates comprehensive outputs in the `outputs/` directory:

#### Visualizations (`outputs/visualizations/`)
- `network_overview.png` - Overall network structure
- `degree_centrality.png` - Degree centrality distribution
- `betweenness_centrality.png` - Betweenness centrality distribution
- `louvain_communities.png` - Louvain community detection results
- `spectral_communities.png` - Spectral clustering results

#### Metrics (`outputs/metrics/`)
- `centrality_metrics.json` - Complete centrality scores for all nodes

#### Exports (`outputs/exports/`)
- `centrality_metrics.csv` - Exportable centrality data (197KB)
- `community_assignments.csv` - Node-to-community mappings (79KB)
- `model_performance.csv` - Machine learning model performance metrics
- `social_network.graphml` - NetworkX-compatible graph format (734KB)
- `social_network.gexf` - Gephi-compatible graph format (1.2MB)

#### Additional Analysis Outputs
- `outputs/topology_analysis/` - Network topology analysis results
- `outputs/scalability_analysis/` - Scalability analysis results
- `outputs/sentiment_analysis/` - Sentiment analysis results
- `outputs/models/` - Trained machine learning models
- `outputs/documentation/` - Generated documentation

## 🚀 How to Run the Analysis

### Option 1: Python Script (Recommended)
```bash
# Install required packages
pip install -r requirements.txt

# Run the complete analysis pipeline
python run_analysis.py
```

### Option 2: Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook run_analysis.ipynb

# Or use Jupyter Lab
jupyter lab run_analysis.ipynb
```

### Option 3: Individual Components
The analysis is modular and can be run component by component:
```python
# Import specific components
from social_network_analysis.data.data_loader import DataLoader
from social_network_analysis.graph.graph_builder import GraphBuilder
from social_network_analysis.analysis.centrality_calculator import CentralityCalculator
# ... etc
```

## 📈 Implemented Analysis Components

### 1. Network Representation & Centrality (Task 1)
- **NetworkX Graph Construction**: Comprehensive social network graph
- **Centrality Metrics**: Degree, Betweenness, Closeness, Clustering Coefficient
- **Network Statistics**: Density, degree distribution, connected components

### 2. Community Detection (Task 2)
- **Louvain Algorithm**: Fast modularity-based community detection
- **Spectral Clustering**: Matrix-based community detection
- **Community Analysis**: Size, density, and key members analysis

### 3. Influence Propagation (Task 3)
- **PyTorch Geometric Implementation**: Advanced graph neural network approach
- **Information Diffusion Simulation**: Step-by-step propagation modeling
- **Seed Node Selection**: High-centrality node identification
- **Community Impact Analysis**: How communities affect information flow

### 4. Machine Learning for Link Prediction (Task 4)
- **GraphSAGE**: Inductive graph neural network
- **Graph Neural Networks (GNN)**: Advanced deep learning on graphs
- **Traditional ML Baselines**: Random Forest, SVM for comparison
- **Performance Evaluation**: Comprehensive metrics and model comparison

### 5. Scalability Analysis (Task 5)
- **Performance Benchmarking**: Analysis of algorithm scalability
- **Memory Usage Analysis**: Resource consumption patterns
- **Optimization Recommendations**: Performance improvement strategies

### 6. Sentiment Analysis (Task 6)
- **Text Sentiment Analysis**: Using TextBlob for sentiment scoring
- **Network Integration**: Combining sentiment with network structure
- **Community Sentiment**: Sentiment analysis by community

### 7. Topology Analysis (Task 7)
- **Network Topology**: Structural properties and characteristics
- **Small-World Analysis**: Network efficiency and clustering
- **Robustness Analysis**: Network resilience to node/edge removal

## 🎯 Key Features

### Comprehensive Coverage
- All 7 assessment tasks fully implemented
- Multiple algorithms for comparison and validation
- Statistical validation and interpretation

### Advanced Analytics
- Graph neural networks for link prediction
- PyTorch Geometric for influence propagation
- Sentiment analysis integration
- Scalability and topology analysis

### Professional Outputs
- Publication-ready visualizations
- Exportable data formats (CSV, JSON, GraphML, GEXF)
- Comprehensive documentation and reports

### Reproducible Research
- Complete methodology documentation
- All parameters and settings recorded
- Modular, maintainable code structure

## 🔍 Understanding the Results

### Network Structure
The analysis reveals:
- Overall network connectivity and density
- Small-world properties and efficiency
- Central vs. peripheral user identification

### Community Organization
Community detection shows:
- Natural user groupings and characteristics
- Community sizes and internal structure
- Inter-community relationships and boundaries

### Influence Dynamics
Propagation analysis demonstrates:
- Information spread patterns through the network
- Most influential users and seed node effectiveness
- Community impact on information flow

### Machine Learning Insights
Link prediction provides:
- Relationship prediction accuracy
- Feature importance in social connections
- Model performance comparisons

## 📚 Technical Requirements

### Python Packages
- `networkx` - Graph analysis and manipulation
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Static visualizations
- `seaborn` - Statistical plotting
- `torch` - PyTorch for deep learning
- `torch_geometric` - Graph neural networks
- `scikit-learn` - Traditional machine learning
- `textblob` - Sentiment analysis
- `jupyter` - Notebook environment

### System Requirements
- Python 3.7+
- 8GB+ RAM (for large networks and ML models)
- GPU recommended for GNN training (optional)
- Modern web browser (for interactive visualizations)

## 🎓 Educational Value

This project serves as:
- **Complete Implementation**: All assessment tasks fully addressed
- **Learning Resource**: Comprehensive social network analysis tutorial
- **Reference Implementation**: Production-ready analysis pipeline
- **Research Template**: Adaptable for other datasets and research questions

## 🔧 Customization

The pipeline can be easily customized for:
- Different datasets (modify data loading sections)
- Additional analysis components
- Alternative visualization styles
- Custom machine learning models
- Extended sentiment analysis