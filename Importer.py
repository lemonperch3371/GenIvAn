from Refined_File_Reader import *
import pickle
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# LOADING...
file_path = pickle.load(open("file_path.pkl", "rb"))
byte_shift = pickle.load(open("byte_shift.pkl", "rb"))
header = pickle.load(open("header.pkl", "rb"))

long_file_data = get_long_file(file_path, byte_shift)

num_params = number_of_parameters(file_path)
param_list = parameters(file_path)
num_fixed_params = number_of_fixed_parameters(file_path)
fixed_param_list = fixed_parameters(file_path)
total_params = total_number_of_parameters(file_path)
all_param_list = all_parameters(file_path)
grid_dims = grid_dimensions(file_path)
num_points = points(file_path)
channel_list = channels(file_path)
num_channels = number_of_channels(file_path)
partition_size = partitions(file_path)
long_blocks = long_data_blocks(file_path, byte_shift)
bias_array = bias_sweep_array(file_path, byte_shift)
df = df_generator(file_path, byte_shift)
cd = channel_dictionary(file_path, byte_shift)


step = max(1, len(bias_array) // 10)
marks = {i: f"{bias_array[i]:.2f} V" for i in range(0, len(bias_array), step)}

# ---------- Dash App ----------
app = Dash(__name__)
server = app.server  


channel_key = 'LIX 1 omega (A)'

app.layout = html.Div([
    html.H1("Interactive dI/dV Map Viewer", style={'textAlign': 'center', 'color': 'red', 'fontSize': '40px', 'fontFamily': 'Garamond'}),

    html.Div([
        dcc.Slider(id='pt-slider', min=0, max=len(bias_array) - 1, step=1, value=41, marks=marks, tooltip={"placement": "bottom", "always_visible": True})
    ], style={'width': '50%', 'margin': '0 auto'}),

    dcc.Graph(id='didv-map', style={'height': '800px'}),
])


@app.callback(
    Output('didv-map', 'figure'),
    Input('pt-slider', 'value')
)
def update_didv_map(pt):
    full_data = cd[channel_key][:, :, pt]
    avg_val = np.average(full_data)

    fig = go.Figure(data=go.Heatmap(
        z=full_data,
        colorscale='Magma',
        zmin=0.5 * avg_val,
        zmax=1.5 * avg_val,
        colorbar=dict(title='dI/dV (A)')
    ))

    fig.update_layout(
        title=f'dI/dV Map - Bias = {bias_array[pt]:.2f} V',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(scaleanchor="x", scaleratio=1),  
    )


    return fig


if __name__ == '__main__':
    app.run(debug=True)
