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
    marks = {i: f"{bias_array[i]:.3f} V" for i in range(0, len(bias_array), step)}

    # ---------- Dash App ----------
    app = Dash(__name__)
    server = app.server  


    channel_key = 'LIX 1 omega (A)'

    app.layout = html.Div([
        html.H1("Interactive dI/dV Map Viewer", style={'textAlign': 'center', 'color': 'red', 'fontSize': '40px', 'fontFamily': 'Garamond'}),

        html.Div([
            dcc.Slider(id='pt_slider', min=0, max=len(bias_array) - 1, step=1, value=41, marks=marks, tooltip={"placement": "bottom", "always_visible": True})
        ], style={'width': '50%', 'margin': '0 auto'}),

        dcc.Graph(id='didV_map', style={'height': '800px'}),  
        
        dcc.Graph(id = "point_i_V_map", style = {"height":"400px"}),
        
        html.Br(),
        
        dcc.Graph(id = "point_didV_V_map", style = {"height":"400px"})
    ])


    @app.callback(
        Output('didV_map', 'figure'),
        Output('point_i_V_map','figure'),
        Output('point_didV_V_map','figure'),
        Input('pt_slider', 'value'),
        Input('didV_map','clickData')
        
    )


    #di/dV map for a given voltage

    #clickData - on click
    # i-v map for each point

    def didv_pt_didv_pt_iv_maps(pt, clickData):
        full_data = cd[channel_key][:, :, pt]
        avg_val = np.average(full_data)

        didv_figure = go.Figure(data=go.Heatmap(z=full_data,colorscale='Magma',zmin=0.5 * avg_val,zmax=1.5 * avg_val,colorbar=dict(title='dI/dV (A)')))

        didv_figure.update_layout(title=f'dI/dV Map - Bias = {bias_array[pt]:.2f} V',xaxis_title='X',yaxis_title='Y',yaxis=dict(scaleanchor="x", scaleratio=1))
        
        if not clickData:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Click on any point of the di/dV map to see the i-V, di/dV-V, and z maps at that particular point!", xaxis_title = "Bias (V)", yaxis_title = "di/dV (arb.)")
            return(didv_figure,empty_fig, empty_fig)
        
        point_x = clickData['points'][0]['x']
        point_y = clickData['points'][0]['y']
        
        # point_z = clickData['points'][0]['z']
        
        i = cd['Current (A)'][point_x][point_y][:]
        didv = cd['LIX 1 omega (A)'][point_x][point_y][:]
        
        #point_i_v_curve
        point_i_v_figure = go.Figure(go.Scatter(x = bias_array,y = i, mode='lines+markers',line = dict(color = 'blue')))
        point_i_v_figure.update_layout(title = f"i-V curve for Point (x = {point_x}, y = {point_y})", xaxis_title = "Bias (V)", yaxis_title = "Current (A)")
        
        #point_didv_v_curve
        point_didv_v_figure = go.Figure(go.Scatter(x = bias_array,y=didv, mode="lines+markers",line = dict(color = 'red')))
        point_didv_v_figure.update_layout(title = f"di/dV-V curve for Point (x = {point_x}, y = {point_y})", xaxis_title = "Bias (V)", yaxis_title = "di/dV (arb)")
        
        return(didv_figure, point_i_v_figure, point_didv_v_figure)
        
        
    #Main Function here...
    if __name__ == '__main__':
        app.run(debug=True)
