import streamlit as st
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import os
import gc
import json
from torch.utils import dataloader,dataset



class ListLoaderDataset(dataset.Dataset):
    def __init__(self, data_list_path,inputs):
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            self.data_lines = [line.strip() for line in lines if line.strip()]
        self.inputs = inputs

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        data = {}
        data_line = self.data_lines[idx]
        input_paths = data_line.split(' ')
        for input, input_path in zip(self.inputs, input_paths):
            if input_path.endswith('.npy'):
                data[input.name] = np.load(input_path)
            else:
                data[input.name] = np.fromfile(input_path,dtype=input.type).reshape(input.shape)
        return data
    

# --- Helper Functions ---

def get_onnx_inputs_info(sess):
    inputs_info = {}
    for inp in sess.get_inputs():
        name = inp.name
        shape = inp.shape
        clean_shape = []
        for dim in shape:
            if isinstance(dim, (int, float)):
                clean_shape.append(dim)
            else:
                clean_shape.append(1)
        inputs_info[name] = {"shape":clean_shape,
                             "type":np.dtype(ort.numpy_obj_dtype_from_onnx_type(inp.type))}
    return inputs_info

def generate_inputs_dict(inputs_info, seed, method="Random Noise"):
    np.random.seed(seed) 
    inputs = {}
    for name, info in inputs_info.items():
        shape = info["shape"]
        if method == "Random Noise":
            inputs[name] = np.random.rand(*shape).astype(np.float32)
        elif method == "Random Normal":
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        elif method == "Zeros":
            inputs[name] = np.zeros(shape, dtype=np.float32)
        elif method == "Ones":
            inputs[name] = np.ones(shape, dtype=np.float32)
    return inputs

def add_intermediate_outputs(onnx_model):
    for node in onnx_model.graph.node:
        for output in node.output:
            if output not in [x.name for x in onnx_model.graph.output]:
                onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    return onnx_model

def get_percentile_from_hist(counts, bin_edges, percentile):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    cdf = np.cumsum(counts)
    target = total * percentile
    idx = np.searchsorted(cdf, target)
    if idx >= len(bin_edges) - 1:
        idx = len(bin_edges) - 2
    return (bin_edges[idx] + bin_edges[idx+1]) / 2

# --- GUI Setup ---

st.set_page_config(layout="wide", page_title="CUDA ONNX Analyzer")
st.title("⚡ Multi-Input ONNX Analyzer (Histogram Only)")

st.sidebar.header("1. Configuration")
uploaded_file = st.sidebar.file_uploader("Upload ONNX Model", type=["onnx"])

available_providers = ort.get_available_providers()
default_provider_index = 0
if "CUDAExecutionProvider" in available_providers:
    default_provider_index = available_providers.index("CUDAExecutionProvider")

provider = st.sidebar.selectbox("Execution Provider", available_providers, index=default_provider_index)

list_config = json.load(open('onnx_utils/list_config.json'))

NUM_SAMPLES = 10
BIN_COUNT = 1000

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        model = onnx.load(tmp_path)
        model = add_intermediate_outputs(model)
        
        sess = ort.InferenceSession(model.SerializeToString(), providers=[provider])
        inputs_info = get_onnx_inputs_info(sess)
        
        st.sidebar.success(f"Model Loaded! Inputs detected: {len(inputs_info)}")
        with st.sidebar.expander("Input Details"):
            st.json(inputs_info)

        st.sidebar.header("2. Analysis Settings")
        data_lists = [data_list for data_list in list_config.keys() if data_list != ""]
        data_lists.append("Random Noise")            
        data_method = st.sidebar.selectbox(
            "Sample data", 
           data_lists
        )
        
        if st.sidebar.button("Run Deep Analysis (2-Pass)"):
            
            output_names = [x.name for x in sess.get_outputs()]
            
            # --- PASS 1: Calculate Min/Max ---
            progress_bar = st.progress(0, text="Pass 1/2: Calculating Global Min/Max...")
            
            layer_bounds = {name: {'min': float('inf'), 'max': float('-inf')} for name in output_names}
            valid_layers = set()
            if data_method != "Random Noise":
                dataset_obj = ListLoaderDataset(data_list_path=list_config[data_method],
                                                inputs=list(sess.get_inputs()))
                data_loader = dataloader.DataLoader(dataset_obj, batch_size=1, shuffle=False)
                
                total_samples = len(dataset_obj)
                
                for i, batch in enumerate(data_loader):
                    inputs = {k: v.numpy() for k, v in batch.items()}
                    outputs = sess.run(None, inputs)
                    
                    for name, arr in zip(output_names, outputs):
                        if arr.size == 0:
                            continue
                            
                        valid_layers.add(name)
                        local_min = np.min(arr)
                        local_max = np.max(arr)
                        if local_min < layer_bounds[name]['min']:
                            layer_bounds[name]['min'] = float(local_min)
                        if local_max > layer_bounds[name]['max']:
                            layer_bounds[name]['max'] = float(local_max)
                    
                    progress_bar.progress((i + 1) / (total_samples * 2), text=f"Pass 1: Sample {i+1}/{total_samples}")
            else:
               for i in range(NUM_SAMPLES):
                   inputs = generate_inputs_dict(inputs_info, seed=i, method=data_method)
                   outputs = sess.run(None, inputs)
                   for name, arr in zip(output_names, outputs):
                        if arr.size == 0:
                            continue
                        valid_layers.add(name)
                        local_min = np.min(arr)
                        local_max = np.max(arr)
                        if local_min < layer_bounds[name]['min']:
                            layer_bounds[name]['min'] = float(local_min)
                        if local_max > layer_bounds[name]['max']:
                            layer_bounds[name]['max'] = float(local_max)
                
                   progress_bar.progress((i + 1) / (NUM_SAMPLES * 2), text=f"Pass 1: Sample {i+1}/{NUM_SAMPLES}")

            # --- PREPARE HISTOGRAM BINS ---
            layer_hists = {}
            for name in valid_layers:
                bounds = layer_bounds[name]
                vmin, vmax = bounds['min'], bounds['max']
                if vmin == vmax: vmax += 1e-6 
                
                edges = np.linspace(vmin, vmax, BIN_COUNT + 1)
                layer_hists[name] = {
                    'counts': np.zeros(BIN_COUNT, dtype=np.int64),
                    'edges': edges,
                    'min': vmin,
                    'max': vmax
                }

            # --- PASS 2: Calculate Histograms ---
            progress_bar.progress(0.5, text="Pass 2/2: Computing Histograms & Box Plots...")

            for i in range(NUM_SAMPLES):
                inputs = generate_inputs_dict(inputs_info, seed=i, method=data_method)
                outputs = sess.run(None, inputs)
                
                for name, arr in zip(output_names, outputs):
                    if name not in valid_layers or arr.size == 0:
                        continue
                        
                    counts, _ = np.histogram(arr, bins=layer_hists[name]['edges'])
                    layer_hists[name]['counts'] += counts
                
                progress_bar.progress(0.5 + ((i + 1) / (NUM_SAMPLES * 2)), text=f"Pass 2: Sample {i+1}/{NUM_SAMPLES}")
            
            progress_bar.empty()
            
            # --- Compile Final Stats ---
            stats_list = []
            final_viz_data = {} 

            for name in valid_layers:
                hist_data = layer_hists[name]
                counts = hist_data['counts']
                edges = hist_data['edges']
                
                vmin = hist_data['min']
                vmax = hist_data['max']
                
                median = get_percentile_from_hist(counts, edges, 0.50)
                
                # 99% Range Stats (P99.5 - P0.5)
                p_low = get_percentile_from_hist(counts, edges, 0.005)
                p_high = get_percentile_from_hist(counts, edges, 0.995)
                range_99 = p_high - p_low
                
                stats_list.append({
                    "Layer Name": name,
                    "Min": vmin,
                    "Max": vmax,
                    "Full Range": vmax - vmin,
                    "99% Range": range_99,
                    "Median": median
                })
                
                bin_centers = (edges[:-1] + edges[1:]) / 2
                final_viz_data[name] = {
                    "bin_centers": bin_centers,
                    "counts": counts,
                    "median": median,
                    "min": vmin,
                    "max": vmax,
                    "p_low": p_low,
                    "p_high": p_high
                }

            if not stats_list:
                st.warning("No valid layers found (all outputs were empty).")
            else:
                st.session_state['df_results'] = pd.DataFrame(stats_list)
                st.session_state['viz_data'] = final_viz_data
                st.session_state['analysis_complete'] = True
            
            del layer_bounds, layer_hists
            gc.collect()

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Display Results ---

if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
    df = st.session_state['df_results']
    viz_data = st.session_state['viz_data']

    st.divider()
    
    st.subheader("📊 Layer Statistics Summary")
    
    column_config = {
        "Min": st.column_config.NumberColumn(format="%.4f"),
        "Max": st.column_config.NumberColumn(format="%.4f"),
        "Full Range": st.column_config.NumberColumn(format="%.4f"),
        "99% Range": st.column_config.NumberColumn(format="%.4f"),
        "Median": st.column_config.NumberColumn(format="%.4f"),
    }

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config=column_config
    )
    
    selected_rows = event.selection.rows
    
    if selected_rows:
        idx = selected_rows[0]
        layer_name = df.iloc[idx]["Layer Name"]
        data = viz_data[layer_name]
        
        st.subheader(f"Layer Analysis: `{layer_name}`")
        st.markdown("### Interactive Coverage Histogram")
        
        # --- SLIDER LOGIC ---
        min_val = float(data['min'])
        max_val = float(data['max'])
        
        # Default to the calculated 99% range
        default_low = float(data['p_low'])
        default_high = float(data['p_high'])
        
        if default_low < min_val: default_low = min_val
        if default_high > max_val: default_high = max_val
        if min_val == max_val: max_val += 1e-6
        
        # Full Width Slider
        range_sel = st.slider(
            "Adjust Range Lines to calculate coverage:",
            min_value=min_val,
            max_value=max_val,
            value=(default_low, default_high),
            format="%.4f"
        )
        
        # --- CALCULATE COVERAGE ---
        mask = (data['bin_centers'] >= range_sel[0]) & (data['bin_centers'] <= range_sel[1])
        covered_counts = np.sum(data['counts'][mask])
        total_counts = np.sum(data['counts'])
        
        percentage = 0.0
        if total_counts > 0:
            percentage = (covered_counts / total_counts) * 100
            
        st.info(f"Selected Range **[{range_sel[0]:.4f}, {range_sel[1]:.4f}]** covers **{percentage:.2f}%** of the data.")
        
        # --- PREPARE COLOR CODED HISTOGRAM ---
        plot_df = pd.DataFrame({
            'Value': data['bin_centers'],
            'Frequency': data['counts']
        })
        
        plot_df['Status'] = np.where(
            (plot_df['Value'] >= range_sel[0]) & (plot_df['Value'] <= range_sel[1]), 
            'Selected Range', 
            'Outside'
        )
        
        color_map = {'Selected Range': '#FF4B4B', 'Outside': '#262730'}

        fig_hist = px.bar(
            plot_df,
            x='Value', 
            y='Frequency',
            color='Status',
            color_discrete_map=color_map,
            title=f"Distribution with Coverage Highlight",
            template="plotly_dark"
        )
        
        fig_hist.add_vline(x=range_sel[0], line_width=2, line_dash="solid", line_color="white")
        fig_hist.add_vline(x=range_sel[1], line_width=2, line_dash="solid", line_color="white")
        
        fig_hist.update_layout(bargap=0, showlegend=True, legend_title_text="")
        
        # Full width plot
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("👆 Click on a row to see detailed plots.")