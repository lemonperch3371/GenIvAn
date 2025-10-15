from File_Reader import *
import pickle
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

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

# Dash APP Construction...
app = Dash(__name__)
server = app.server  

### Channel Key Channel Key Channel Key
# channel_key = 'LIX 1 omega (A)'
### Channel Key Channel Key Channel Key
# os.path.basename(file_path)


    
    # Point Map 2 -----
    
# 
app.layout = html.Div([

    # ===== HEADER SECTION =====
    html.Div([
        html.H3(
            f"Interactive Map Viewer for the file: {os.path.basename(file_path)}",
            style={'textAlign': 'center', 'color': 'red', 'fontSize': '40px', 'fontFamily': 'Garamond'}
        ),
        html.H5(
            f"The file path: {file_path}",
            style={'textAlign': 'center', 'color': '#0505A8', 'fontSize': '20px', 'fontFamily': 'Garamond'}
        ),
        html.H5(
            "Click anywhere on the map to begin!",
            style={'textAlign': 'center', 'color': 'blue', 'fontSize': '30px', 'fontFamily': 'Garamond'}
        ),
    ], style={'width': '100%', 'display': 'block', 'marginBottom': '40px'}),



        
    
        # MAIN SECTION MAIN SECTION MAIN SECTION MAIN SECTION
        html.Div([

            # LEFT COLUMN...
            html.Div([ html.H3("Preprocessing Dropdown", style = {'textAlign':'center','color':"#390ED3", 'fontfamily':'Garamond','textAlign':'center'}),
                # Preprocessing Dropdown
                dcc.Dropdown(
                    id='preprocess_dropdown',
                    options=[
                        {'label': 'None', 'value': 'none'},
                        {'label': 'Subtract Average', 'value': 'mean'},
                        {'label': 'Subtract Linear Fit', 'value': 'linear'},
                        {'label': 'Subtract Polynomial Fit', 'value': 'polynomial'},
                        {'label': 'Gaussian Blur (σ)', 'value': 'gaussian'},
                        {'label': 'Savitzky-Golay Filter', 'value': 'sg_filter'},
                        {'label': 'K-Means Clustering', 'value': 'kmeans'},
                    ],
                    value='none',
                    style={'width': '100%', 'display': 'block'}
                ),

                # Dynamic Preprocessing Parameter Sliders
                html.Div([
                    # Gaussian Blur
                    html.Div([
                        html.Label("Gaussian Blur Sigma (σ)", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='gaussian_sigma_slider',
                            min=0.1, max=5.0, step=0.1, value=1.5,
                            marks={i: f"{i:.1f}" for i in np.arange(1, 5.5, 0.5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag',
                        )
                    ], id='gaussian_slider_container', style={'display': 'none'}),

                    # Polynomial Degree
                    html.Div([
                        html.Label("Polynomial Fit Degree (power)", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='polynomial_degree_slider',
                            min=1, max=20, step=1, value=3,
                            marks={i: str(i) for i in range(1, 21)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag',
                        )
                    ], id='polynomial_degree_container', style={'display': 'none', 'width':'500px'}),

                    # Savitzky-Golay: Window Length
                    html.Div([
                        html.Label("Savitzky-Golay Window Length (odd)", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='sg_window_slider',
                            min=3, max=33, step=2, value=7,
                            marks={i: str(i) for i in range(3, 34, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag',
                        )
                    ], id='sg_window_container', style={'width':'500px','display': 'none'}),

                    # Savitzky-Golay: Polynomial Order
                    html.Div([
                        html.Label("Savitzky-Golay Polynomial Order", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='sg_polyorder_slider',
                            min=1, max=30, step=1, value=3,
                            marks={i: str(i) for i in range(1, 31)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag',
                        )
                    ], id='sg_polyorder_container', style={'width': '100%', 'display': 'none'}),

                    # KMeans Clusters
                    html.Div([
                        html.Label("K-Means Number of Clusters", style={'textAlign': 'center', 'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='kmeans_clusters_slider',
                            min=2, max=20, step=1, value=5,
                            marks={i: str(i) for i in range(2, 21)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='drag',
                        )
                    ], id='kmeans_clusters_container', style={'width': '70%', 'margin': '0 auto 30px auto', 'display': 'none'}),

                ], id='preprocess_params_container'),

                # Main Channel Selector
                dcc.Dropdown(
                    id='channel_dropdown',
                    options=[{'label': key, 'value': key} for key in list(cd.keys())],
                    value=list(cd.keys())[0],
                    style={'width': '70%', 'margin': '20px auto', 'display': 'block'}
                ),

                # Point Slider
                html.Div([
                    dcc.Slider(
                        id='pt_slider',
                        min=0, max=len(bias_array) - 1, step=1, value=1,
                        marks=marks,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'
                    )
                ], style={'width': '100%', 'margin': '0 auto 40px auto', 'display': 'block'}),

                # Main Channel Map
                dcc.Graph(id='selected_channel_map_fig', style={'height': '600px', 'width': '600px'}),

                html.Div(style={'height': '20px'}),

                # FFT Map
                dcc.Graph(id='selected_channel_fft_fig', style={'height': '600px', 'width': '600px'}),
            ], style={'width': '60%', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'baseline',  'rightPadding':'4000px', 'leftPadding':'1000px'}), #XYZXYZ 1 STYLING MAIN MAPS (Inside the XYZXYZ 1 AND XYZXYZ 2 CONTAINER)

            ######### POINT MAPS POINT MAPS POINT MAPS ########
            
            
            
            html.Div([ html.Br(), 
                html.H3("First Point Map", style={'textAlign': 'center', 'color': '#0BB0B0', 'fontSize': '30px', 'fontFamily': 'Garamond'}),
                
                #Dropdown for Filtering the Point Maps_1
                html.Div([html.H5("This part contains something..."), 
                         html.Br(),
                         dcc.Dropdown(id = 'point_channel_filters_dropdown_1',
                                      options = [{"label":"None", "value":"none"},
                                                 {'label':"Savitzky-Golay Filter", "value":"sav_gol_pt_1"},
                                                 {"label":"Butterworth Filter", "value":"butterworth_pt_1"},
                                                 {"label":"Gaussian Filter", "value":"gaussian_pt_1"},
                                                 ], style = {"width": "300px"})
                         ]), #Dropdown for filtering individual plots...
                
                
                #POINT MAP 1 DROPDOWN
                dcc.Dropdown(
                    id='point_channel_dropdown_1',
                    options=[{'label': key, 'value': key} for key in list(cd.keys())],
                    value=list(cd.keys())[0],
                    style={'width': '300px', 'margin': '10px', 'display': 'block'}
                ),
                
                #POINT MAP 1
                dcc.Graph(id="point_map_1", style={"height": "500px", "width": "500px"}),
                
                
                #SPACING BETWEEN THE PLOTS
                html.Div(style={'height': '30px'}),



                #POINT MAP 2 BEGINS...
                html.H3("Second Point Map", style={'textAlign': 'center', 'color': '#CFAA06', 'fontSize': '30px', 'fontFamily': 'Garamond'}),
                
                #POINT MAP 2 DROPDOWN
                dcc.Dropdown(
                    id='point_channel_dropdown_2',
                    options=[{'label': key, 'value': key} for key in list(cd.keys())],
                    value=list(cd.keys())[0],
                    style={'width': '300px', 'margin': '10px', 'display': 'block'}
                ),
                
                
                #POINT MAP 2
                dcc.Graph(id="point_map_2", style={"height": "500px", "width": "500px"}),
                
                
            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'paddingLeft': '100px'}), #FINAL STYLING FOR [POINT] MAPS XYZXYZ 2



        # DIFFERENTIATED PLOTS SECTION 
        html.Div([ html.Br(), 
            html.H3("First Point Map: Differentiated", style={'textAlign': 'center', 'color': "#54600B", 'fontSize': '30px', 'fontFamily': 'Garamond'}),
            dcc.Graph(id='point_map_1_diff', style={"height": "500px", "width": "500px"}),

            html.Div(style={'height': '20px'}),

            html.H3("Second Point Map: Differentiated", style={'textAlign': 'center', 'color': "#B05A06", 'fontSize': '30px', 'fontFamily': 'Garamond'}),
            dcc.Graph(id="point_map_2_diff", style={"height": "500px", "width": "500px"}),
        ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'align-items':'center',}), #FINAL STYLING FOR DIFFERENTIATED MAPS XYZXYZ 3
        
        
        
    ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'}, #for XYZXYZ 1, XYZXYZ 2, XYZXYZ 3 COMBINED; XYZXYZ 4
        )
        ]
    )







@app.callback(
    Output('gaussian_slider_container', 'style'),
    Output('polynomial_degree_container', 'style'),
    Output('sg_window_container', 'style'),
    Output('sg_polyorder_container', 'style'),
    Output('kmeans_clusters_container', 'style'),
    
    
    Input('preprocess_dropdown', 'value')
)
def toggle_preprocess_params(selected):
    hidden = {'display': 'none'}
    visible = {'width': '200px', 'margin': '0 auto 30px auto', 'display': 'block'}

    if selected == 'gaussian':
        gaussian_visible = visible
        polynomial_visible = hidden
        sg_window_visible = hidden
        sg_polyorder_visible = hidden
        kmeans_visible = hidden
        
    elif selected == 'polynomial':
        gaussian_visible = hidden
        polynomial_visible = visible
        sg_window_visible = hidden
        sg_polyorder_visible = hidden
        kmeans_visible = hidden
        
    elif selected == 'sg_filter':
        gaussian_visible = hidden
        polynomial_visible = hidden
        sg_window_visible = visible
        sg_polyorder_visible = visible
        kmeans_visible = hidden
        
    elif selected == 'kmeans':
        gaussian_visible = hidden
        polynomial_visible = hidden
        sg_window_visible = hidden
        sg_polyorder_visible = hidden
        kmeans_visible = visible
        
    else:
        gaussian_visible = hidden
        polynomial_visible = hidden
        sg_window_visible = hidden
        sg_polyorder_visible = hidden
        kmeans_visible = hidden

    return gaussian_visible, polynomial_visible, sg_window_visible, sg_polyorder_visible, kmeans_visible




@app.callback(
    Output('selected_channel_map_fig', 'figure'),
    Output('point_map_1','figure'),
    Output('point_map_2','figure'),
    Output('selected_channel_fft_fig','figure'),
    Output('point_map_1_diff', 'figure'),
    Output('point_map_2_diff', 'figure'),
    
    
    Input('pt_slider', 'value'),
    Input('selected_channel_map_fig','clickData'),
    Input('channel_dropdown','value'),
    Input('point_channel_dropdown_1','value'),
    Input('point_channel_dropdown_2','value'),
    Input('preprocess_dropdown', 'value'),
    Input('gaussian_sigma_slider','value'),
    Input('polynomial_degree_slider', 'value'),
    Input('sg_window_slider', 'value'),
    Input('sg_polyorder_slider', 'value'),
    Input('kmeans_clusters_slider', 'value'),
)



#di/dV map for a given voltage

#clickData - on click
# i-v map for each point
def update_all_maps(pt, clickData, main_channel, point_channel_1, point_channel_2,preprocess_method, gaussian_sigma, polynomial_degree, sg_window, sg_polyorder, kmeans_clusters):
    # Main map and FFT map (using main_channel)
    # selected_channel_values = cd[main_channel][:, :, pt]
    
    
    selected_channel_values = cd[main_channel][:, :, pt]

    # Apply preprocessing
    if preprocess_method == 'mean':
        selected_channel_values = selected_channel_values - np.mean(selected_channel_values)

    elif preprocess_method == 'linear':

        y, x = np.indices(selected_channel_values.shape)
        Linear_Data = np.stack([x.ravel(), y.ravel()], axis=1)
        model = LinearRegression().fit(Linear_Data, selected_channel_values.ravel())
        trend = model.predict(Linear_Data).reshape(selected_channel_values.shape)
        
        selected_channel_values = selected_channel_values - trend

    elif preprocess_method == 'polynomial':

        y, x = np.indices(selected_channel_values.shape)
        X = np.stack([x.ravel(), y.ravel()], axis=1)
        poly = PolynomialFeatures(degree=polynomial_degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, selected_channel_values.ravel())
        trend = model.predict(X_poly).reshape(selected_channel_values.shape)
        
        selected_channel_values = selected_channel_values - trend
        
    elif preprocess_method == 'gaussian':
        # Apply a 2D Gaussian filter with user-defined sigma
        sigma = gaussian_sigma  # This should be passed in from the callback
        selected_channel_values = gaussian_filter(selected_channel_values, sigma=sigma)
        
    elif preprocess_method == 'sg_filter': 
        # Apply Savitzky-Golay filter

        
        window_length_x = sg_window  
        window_length_y = sg_window  
        polyorder = sg_polyorder

        
        if window_length_x <= polyorder:
            window_length_x = polyorder + 2  
            if window_length_x % 2 == 0:
                window_length_x += 1

        if window_length_y <= polyorder:
            window_length_y = polyorder + 2
            if window_length_y % 2 == 0:
                window_length_y = window_length_y + 1

        smooth_x = savgol_filter(selected_channel_values, window_length=window_length_x, polyorder=polyorder, axis=1, mode='mirror')
        smooth_xy = savgol_filter(smooth_x, window_length=window_length_y, polyorder=polyorder, axis=0, mode='mirror')

        selected_channel_values = smooth_xy
    
    
    elif preprocess_method == 'kmeans':
        
        
        z = selected_channel_values.copy()
        z_flat = z.reshape(-1, 1)

        # KMeans with 4 clusters (can tweak)
        kmeans = KMeans(n_clusters=kmeans_clusters, n_init='auto', random_state=0)
        labels = kmeans.fit_predict(z_flat)
        centers = kmeans.cluster_centers_.flatten()
        
        clustered = centers[labels].reshape(z.shape)
        selected_channel_values = clustered
        
    # Calculate the average value for scaling    
    avg_val = np.average(selected_channel_values)
    selected_channel_map_fig = go.Figure(data=go.Heatmap(z=selected_channel_values,colorscale='Magma',zmin=0.1 * avg_val,zmax=1.7 * avg_val,colorbar=dict()))
    selected_channel_map_fig.update_layout(title=f'{main_channel} Map - Bias = {bias_array[pt]:.3f} mV',xaxis_title='X',yaxis_title='Y',yaxis=dict(scaleanchor="x", scaleratio=1), xaxis=dict(scaleanchor="y", scaleratio=1))


    fft2d = np.fft.fftshift(np.fft.fft2(selected_channel_values))
    fft_magnitude_spectrum = np.log(np.abs(fft2d))  
    selected_channel_fft_fig = go.Figure(data=go.Heatmap(z=fft_magnitude_spectrum,colorscale='Viridis',colorbar=dict()))
    selected_channel_fft_fig.update_layout(title=f"2D FFT logged Magnitude Spectrum - Bias = {bias_array[pt]:.3f} V",xaxis_title="k_x",yaxis_title="k_y",yaxis=dict(scaleanchor="x", scaleratio=1))

    # If no clickData, return empty figures for the point channels
    if not clickData:
        empty_fig_1 = go.Figure()
        empty_fig_1.update_layout(title=f"Click on any point of the {point_channel_1} map to see the \n{point_channel_1}-V curve!", xaxis_title="Bias (V)",yaxis_title=point_channel_1)
        empty_fig_2 = go.Figure()
        empty_fig_2.update_layout(title=f"Click on any point of the {point_channel_2} map to see the \n{point_channel_2}-V curve!",xaxis_title="Bias (V)",yaxis_title=point_channel_2)
        empty_diff_fig_1 = go.Figure()
        empty_diff_fig_1.update_layout(title=f"Differentiated plot 1 will appear here")
        empty_diff_fig_2 = go.Figure()
        empty_diff_fig_2.update_layout(title=f"Differentiated plot 2 will appear here")
        return selected_channel_map_fig, empty_fig_1, empty_fig_2, selected_channel_fft_fig, empty_diff_fig_1, empty_diff_fig_2

    point_x = clickData['points'][0]['x'] #Throws back the JSON File
    point_y = clickData['points'][0]['y'] #Throws back the JSON File, which contains x,y,z, curveNumber, pointNumber and things like these... it is inbuilt in the plotly go figure


    channel_1_data = cd[point_channel_1][point_y][point_x][:]
    channel_2_data = cd[point_channel_2][point_y][point_x][:]
    
    diff_channel_1 = np.gradient(channel_1_data,bias_array)
    diff_channel_2 = np.gradient(channel_2_data,bias_array)
    #Channel 1 and Channel 2 Point Maps at the clicked points
    point_map_1_fig = go.Figure(go.Scatter(x=bias_array,y=channel_1_data,mode='lines+markers',line=dict(color='red')))
    point_map_1_fig.update_layout(title=f"{point_channel_1}-V curve for Point (x={point_x}, y={point_y})",xaxis_title="Bias (V)",yaxis_title=point_channel_1)

    point_map_2_fig = go.Figure(go.Scatter(x=bias_array, y=channel_2_data, mode='lines+markers', line=dict(color='blue')))
    point_map_2_fig.update_layout(title=f"{point_channel_2}-V curve for Point (x={point_x}, y={point_y})",xaxis_title="Bias (V)",yaxis_title=point_channel_2)
    
    
    
    #Channel 1 and Channel 2 Differentiated Point Maps at the clicked points

    
    # Create differentiated plots
    point_map_1_diff_fig = go.Figure(go.Scatter(x=bias_array, y=diff_channel_1, mode='lines+markers', line=dict(color='green')))
    point_map_1_diff_fig.update_layout(title=f"Differentiated {point_channel_1}-V curve for Point (x={point_x}, y={point_y})",
                                    xaxis_title="Bias (V)", yaxis_title=f"d({point_channel_1})/dV")

    point_map_2_diff_fig = go.Figure(go.Scatter(x=bias_array, y=diff_channel_2, mode='lines+markers', line=dict(color='purple')))
    point_map_2_diff_fig.update_layout(title=f"Differentiated {point_channel_2}-V curve for Point (x={point_x}, y={point_y})",
                                    xaxis_title="Bias (V)", yaxis_title=f"d({point_channel_2})/dV")
        
    
    
    
    return selected_channel_map_fig, point_map_1_fig, point_map_2_fig, selected_channel_fft_fig, point_map_1_diff_fig, point_map_2_diff_fig

    
#Main Function here...
if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload = True)
    
    
    