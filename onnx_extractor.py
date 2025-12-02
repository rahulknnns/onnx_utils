import onnx
import json
import argparse
import sys

def extract_graph_structure(model_path, output_path):
    print(f"Loading model: {model_path}...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    nodes = []
    edges = []
    
    # Map to track which node produces which tensor
    # key: tensor_name, value: node_name
    tensor_producer_map = {}

    # Identify initializers (weights/biases) so we don't draw them as input nodes
    initializers = {init.name for init in model.graph.initializer}

    print("Analyzing graph inputs...")
    # 1. Process Graph Inputs (The entry points)
    for inp in model.graph.input:
        # Skip if it's actually a weight/bias
        if inp.name in initializers:
            continue
            
        nodes.append({
            "id": inp.name,
            "label": inp.name,
            "type": "Input",
            "inputs": [],
            "outputs": [inp.name]
        })
        # Register this input node as the producer of the input tensor
        tensor_producer_map[inp.name] = inp.name

    print("Analyzing compute nodes...")
    # 2. Process Compute Nodes (Layers)
    for node in model.graph.node:
        # Use node name if available, otherwise generate one based on outputs
        node_id = node.name if node.name else f"{node.op_type}_{node.output[0]}"
        
        nodes.append({
            "id": node_id,
            "label": node_id,
            "type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output)
        })

        # Map outputs to this node
        for output_name in node.output:
            tensor_producer_map[output_name] = node_id

    print("Analyzing graph outputs...")
    # 3. Process Graph Outputs (The exit points)
    for out in model.graph.output:
        # Create a visual node for the output
        output_node_id = f"Output_{out.name}"
        nodes.append({
            "id": output_node_id,
            "label": out.name,
            "type": "Output",
            "inputs": [out.name], # It consumes the final tensor
            "outputs": []
        })

    # 4. Create edges based on inputs
    print("Building connections...")
    for node in nodes:
        for input_name in node['inputs']:
            # If the input comes from a known producer (another node or graph input)
            if input_name in tensor_producer_map:
                producer_id = tensor_producer_map[input_name]
                edges.append({
                    "source": producer_id,
                    "target": node['id']
                })

    # 5. Save to JSON
    structure = {
        "nodes": nodes,
        "edges": edges
    }

    with open(output_path, 'w') as f:
        json.dump(structure, f, indent=2)

    print(f"Done! Extracted {len(nodes)} nodes and {len(edges)} connections.")
    print(f"Graph structure saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ONNX graph structure for visualization.")
    parser.add_argument("model", help="Path to the input .onnx file")
    parser.add_argument("--output", default="graph_structure.json", help="Path to the output .json file")
    
    args = parser.parse_args()
    
    extract_graph_structure(args.model, args.output)