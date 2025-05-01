import networkx as nx
import matplotlib.pyplot as plt

def draw_causal_dag():
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node("Channel State", node_type="input")
    G.add_node("Transmission Rate", node_type="intermediate")
    G.add_node("Bit Error Rate", node_type="intermediate")
    G.add_node("Delay", node_type="output")
    G.add_node("Inference Accuracy", node_type="output")
    G.add_node("Energy Consumption", node_type="output")
    G.add_node("Quantization Level", node_type="intervention")
    G.add_node("LDPC Coding Rate", node_type="intervention")
    G.add_node("Bandwidth Allocation", node_type="intervention")

    # Add edges
    G.add_edge("Channel State", "Transmission Rate")
    G.add_edge("Channel State", "Bit Error Rate")
    G.add_edge("Transmission Rate", "Delay")
    G.add_edge("Bit Error Rate", "Inference Accuracy")
    G.add_edge("Quantization Level", "Transmission Rate")
    G.add_edge("LDPC Coding Rate", "Transmission Rate")
    G.add_edge("Bandwidth Allocation", "Transmission Rate")
    G.add_edge("Transmission Rate", "Energy Consumption")
    G.add_edge("Transmission Rate", "Inference Accuracy")

    # Node colors based on node type
    node_colors = {
        "input": "skyblue",
        "intervention": "lightgreen",
        "output": "lightcoral",
        "intermediate": "lightyellow"
    }
    colors = [node_colors[nx.get_node_attributes(G, "node_type")[node]] for node in G.nodes()]

    # Drawing the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=colors, font_size=10, font_weight="bold", arrowsize=20)
    
    # Add a title and legend
    plt.title("Causal DAG for OMNIS Framework", fontsize=16)
    
    # Create legend manually
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='skyblue', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Intervention', markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='lightcoral', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Intermediate', markerfacecolor='lightyellow', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.show()

# Call the function to draw the graph
draw_causal_dag()