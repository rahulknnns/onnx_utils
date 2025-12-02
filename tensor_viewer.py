import streamlit as st
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Tensor Comparator")


# --- Helper Functions ---
def calculate_nmse(cpu_tensor, hwa_tensor):
    """
    Calculates Normalized Mean Square Error.
    Formula: ||CPU - HWA||^2 / ||CPU||^2
    """
    epsilon = 1e-10 # Prevent division by zero
    error_energy = np.sum((cpu_tensor - hwa_tensor) ** 2)
    ref_energy = np.sum(cpu_tensor ** 2) + epsilon
    return 10 * np.log10(error_energy / ref_energy)

def load_and_compute_metrics(cpu_dir, hwa_dir):
    """
    Scans directories, loads tensors, computes metrics, 
    and returns a DataFrame.
    """
    if not os.path.exists(cpu_dir) or not os.path.exists(hwa_dir):
        return None, "One or both directories do not exist."

    cpu_files = set([f for f in os.listdir(cpu_dir) if f.endswith('.npy')])
    hwa_files = set([f for f in os.listdir(hwa_dir) if f.endswith('.npy')])
    
    # Only process files that exist in both directories
    common_files = list(cpu_files.intersection(hwa_files))
    
    if not common_files:
        return None, "No matching .npy files found in both directories."

    data_rows = []

    progress_bar = st.progress(0)
    for i, filename in enumerate(common_files):
        # Update progress
        progress_bar.progress((i + 1) / len(common_files))
        
        # Load paths
        p_cpu = os.path.join(cpu_dir, filename)
        p_hwa = os.path.join(hwa_dir, filename)
        
        try:
            t_cpu = np.load(p_cpu)
            t_hwa = np.load(p_hwa)
            
            # Ensure they are flat for metric calculation
            f_cpu = t_cpu.flatten()
            f_hwa = t_hwa.flatten()

            # Calculate Metrics
            mse = np.mean((f_cpu - f_hwa) ** 2)
            nmse = calculate_nmse(f_cpu, f_hwa)
            
            data_rows.append({
                "Tensor Name": filename,
                "Min (CPU)": float(np.min(f_cpu)),
                "Max (CPU)": float(np.max(f_cpu)),
                "Min (HWA)": float(np.min(f_hwa)),
                "Max (HWA)": float(np.max(f_hwa)),
                "MSE": mse,
                "NMSE (dB)": nmse,
            })
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    progress_bar.empty()
    return pd.DataFrame(data_rows), None
def plot_visuals(cpu_data, hwa_data, tensor_name, bin_count=None, 
                 show_box=True, show_hist=True, show_scatter=True):
    
    # 1. defined inputs
    f_cpu = cpu_data.flatten()
    f_hwa = hwa_data.flatten()
    indices = np.arange(len(f_cpu))

    # 2. Determine which plots are active and their "height weights"
    # Box = 1 unit, Hist = 1.5 units, Scatter = 2 units of height
    active_plots = []
    if show_box: active_plots.append({"type": "box", "weight": 1.0, "title": "Spread Analysis"})
    if show_hist: active_plots.append({"type": "hist", "weight": 1.5, "title": "Distribution Analysis"})
    if show_scatter: active_plots.append({"type": "scatter", "weight": 2.0, "title": "Element-wise Comparison"})

    if not active_plots:
        fig = go.Figure()
        fig.update_layout(title="No plots selected")
        return fig

    # 3. Calculate dynamic row heights
    total_weight = sum(p["weight"] for p in active_plots)
    row_heights = [p["weight"] / total_weight for p in active_plots]
    
    titles = [p["title"] for p in active_plots]

    # 4. Create Subplots
    fig = make_subplots(
        rows=len(active_plots), 
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=row_heights,
        subplot_titles=titles
    )

    # 5. Add Traces Loop
    # We loop through active_plots and add traces to the correct row index (1-based)
    for i, plot in enumerate(active_plots):
        row_idx = i + 1
        p_type = plot["type"]

        if p_type == "box":
            fig.add_trace(go.Box(x=f_cpu, name='CPU (Box)', marker_color='blue', boxpoints=False), row=row_idx, col=1)
            fig.add_trace(go.Box(x=f_hwa, name='HWA (Box)', marker_color='red', boxpoints=False), row=row_idx, col=1)
            fig.update_xaxes(title_text="Value Range", row=row_idx, col=1)

        elif p_type == "hist":
            # Handle Binning Logic
            hist_params = {}
            if bin_count:
                global_min = min(f_cpu.min(), f_hwa.min())
                global_max = max(f_cpu.max(), f_hwa.max())
                if global_max == global_min: global_max += 0.1
                bin_size = (global_max - global_min) / bin_count
                hist_params = dict(xbins=dict(start=global_min, end=global_max, size=bin_size), autobinx=False)
                fig.layout.annotations[i].text += f" ({bin_count} Bins)" # Update title dynamically

            fig.add_trace(go.Histogram(x=f_cpu, name='CPU (Hist)', marker_color='blue', opacity=0.75, showlegend=True, **hist_params), row=row_idx, col=1)
            fig.add_trace(go.Histogram(x=f_hwa, name='HWA (Hist)', marker_color='red', opacity=0.75, showlegend=True, **hist_params), row=row_idx, col=1)
            fig.update_xaxes(title_text="Value Range", row=row_idx, col=1)
            fig.update_yaxes(title_text="Count", row=row_idx, col=1)

        elif p_type == "scatter":
            fig.add_trace(go.Scattergl(x=indices, y=f_cpu, mode='markers', name='CPU (Points)', marker=dict(color='blue', size=3, opacity=0.5)), row=row_idx, col=1)
            fig.add_trace(go.Scattergl(x=indices, y=f_hwa, mode='markers', name='HWA (Points)', marker=dict(color='red', size=3, opacity=0.5)), row=row_idx, col=1)
            fig.update_xaxes(title_text="Flattened Index", row=row_idx, col=1)
            fig.update_yaxes(title_text="Value", row=row_idx, col=1)

    # 6. Linking Logic (Only link X if Box and Hist are both present)
    # We find the row indices for box and hist
    box_row = next((i+1 for i, p in enumerate(active_plots) if p["type"] == "box"), None)
    hist_row = next((i+1 for i, p in enumerate(active_plots) if p["type"] == "hist"), None)

    if box_row and hist_row:
        fig.update_xaxes(matches='x', row=box_row, col=1)
        fig.update_xaxes(matches='x', row=hist_row, col=1)

    fig.update_layout(
        barmode='overlay',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300 + (len(active_plots) * 200), # Dynamic Height based on plot count
        legend=dict(orientation="v", y=1, x=1.02, title_text="Toggle Layers")
    )
    
    return fig
# --- Main UI Layout ---
st.title("CPU vs HWA Tensor Comparator")

# 1. Inputs
col1, col2 = st.columns(2)
with col1:
    cpu_path = st.text_input("CPU Tensors Directory", value="./cpu_tensors")
with col2:
    hwa_path = st.text_input("HWA Tensors Directory", value="./hwa_tensors")

# 2. Load Data Button
if 'df' not in st.session_state:
    st.session_state.df = None

if st.button("Load & Compare Tensors"):
    with st.spinner("Processing tensors..."):
        df, error = load_and_compute_metrics(cpu_path, hwa_path)
        if error:
            st.error(error)
        else:
            st.session_state.df = df

# 3. Display Table & Handle Interaction
if st.session_state.df is not None:
    st.subheader("Comparison Table")
    st.caption("Click on the checkbox/row to view detailed histograms in the sidebar.")
    
    # Use dataframe with selection mode
    event = st.dataframe(
        st.session_state.df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun", # Triggers script rerun when clicked
        selection_mode="single-row"
    )
# ... (inside the table selection block) ...
    if len(event.selection['rows']) > 0:
        selected_index = event.selection['rows'][0]
        selected_row = st.session_state.df.iloc[selected_index]
        tensor_name = selected_row['Tensor Name']
        
# Open Sidebar
        with st.sidebar:
            st.header(f"Details: {tensor_name}")
            
            # --- Statistics ---
            st.write("### Statistics")
            st.write(f"**MSE:** {selected_row['MSE']:.6f}")
            st.write(f"**NMSE:** {selected_row['NMSE (dB)']:.2f} dB")
            st.divider()

            # --- View Settings ---
            st.write("### View Settings")
            
            # 1. Plot Toggles
            c1, c2, c3 = st.columns(3) # Compact layout
            with c1: show_box = st.checkbox("Box", value=True)
            with c2: show_hist = st.checkbox("Hist", value=True)
            with c3: show_scatter = st.checkbox("Pts", value=True)
            
            # 2. Binning Controls
            use_custom_bins = st.toggle("Custom Bin Count", value=False)
            bin_count = None
            if use_custom_bins and show_hist: # Only show slider if Histogram is ON
                max_slider_val = st.number_input("Max Limit", 100, 2000, 500, step=100)
                bin_count = st.slider("Bins", 10, max_slider_val, min(100, max_slider_val), 10)

            # --- Load Data & Plot ---
            try:
                cpu_data = np.load(os.path.join(cpu_path, tensor_name))
                hwa_data = np.load(os.path.join(hwa_path, tensor_name))
                
                # Pass all flags to the function
                fig = plot_visuals(
                    cpu_data, hwa_data, tensor_name, 
                    bin_count=bin_count,
                    show_box=show_box,
                    show_hist=show_hist,
                    show_scatter=show_scatter
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        with st.sidebar:
            st.info("Select a row in the table to see the histogram.")