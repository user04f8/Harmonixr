import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz_utils import MIDISingleDataset, load_model
from model import WraparoundConv3D

def visualize_conv3d_layers(model, num_filters=5, slice_dim=2):
    """
    Visualize 3D convolutional filters from the MIDIClassifier model.

    Args:
        model (MIDIClassifier): The trained MIDIClassifier model.
        num_filters (int): Number of filters to visualize per convolutional layer.
        slice_dim (int): Dimension along which to slice the 3D filter (0: Depth, 1: Height, 2: Width).
    """
    # Identify all 3D convolutional layers in the model
    conv3d_layers = []
    for name, module in model.named_modules():
        if isinstance(module, WraparoundConv3D):
            conv3d_layers.append((name, module))
    
    if not conv3d_layers:
        print("No 3D convolutional layers found in the model.")
        return
    
    print(f"Found {len(conv3d_layers)} 3D convolutional layer(s).")
    
    for layer_idx, (layer_name, conv_layer) in enumerate(conv3d_layers):
        weights = conv_layer.conv.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, D, H, W)
        out_channels, in_channels, D, H, W = weights.shape
        print(f"\nLayer {layer_idx + 1}: {layer_name}")
        print(f"Number of filters: {out_channels}")
        print(f"Filter shape: (in_channels={in_channels}, D={D}, H={H}, W={W})")
        
        # Select a subset of filters to visualize
        selected_filters = min(num_filters, out_channels)
        filters_to_visualize = weights[:selected_filters]
        
        for filter_idx, filter_weights in enumerate(filters_to_visualize):
            # Average across input channels if necessary
            if in_channels > 1:
                # For multi-channel filters, visualize each channel separately or average them
                # Here, we'll average across input channels for simplicity
                filter_3d = np.mean(filter_weights, axis=0)  # Shape: (D, H, W)
            else:
                filter_3d = filter_weights[0]  # Shape: (D, H, W)
            
            # Determine the number of slices based on slice_dim
            num_slices = filter_3d.shape[slice_dim]
            # Limit the number of slices to visualize
            max_slices = 10
            slice_indices = np.linspace(0, num_slices - 1, min(max_slices, num_slices)).astype(int)
            
            # Create subplots for each slice
            fig = make_subplots(
                rows=1, cols=min(max_slices, num_slices),
                subplot_titles=[f"Slice {idx}" for idx in slice_indices],
                horizontal_spacing=0.02
            )
            
            for i, slice_idx in enumerate(slice_indices):
                if slice_dim == 0:
                    slice_2d = filter_3d[slice_idx, :, :]  # Depth slice
                    x_title = "Height"
                    y_title = "Width"
                elif slice_dim == 1:
                    slice_2d = filter_3d[:, slice_idx, :]  # Height slice
                    x_title = "Depth"
                    y_title = "Width"
                else:
                    slice_2d = filter_3d[:, :, slice_idx]  # Width slice
                    x_title = "Depth"
                    y_title = "Height"
                
                # Normalize the slice for better visualization
                slice_min, slice_max = slice_2d.min(), slice_2d.max()
                if slice_max - slice_min > 0:
                    slice_normalized = (slice_2d - slice_min) / (slice_max - slice_min)
                else:
                    slice_normalized = slice_2d - slice_min  # All zeros
                
                fig.add_trace(
                    go.Heatmap(
                        z=slice_normalized,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    row=1, col=i + 1
                )
                
                # Update each subplot's axes
                fig.update_xaxes(title_text=x_title, row=1, col=i + 1)
                fig.update_yaxes(title_text=y_title, row=1, col=i + 1)
            
            # Update the layout of the figure
            fig.update_layout(
                title_text=f"Layer: {layer_name} | Filter {filter_idx + 1}/{selected_filters}",
                height=300,
                width=300 * min(max_slices, num_slices),
                showlegend=False
            )
            
            fig.show()

if __name__ == '__main__':
    print("Loading model")
    checkpoint_path = r'tb_logs/SiaViT/version_34/checkpoints/last.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device=device)

    visualize_conv3d_layers(model, num_filters=1e99, slice_dim=0)

