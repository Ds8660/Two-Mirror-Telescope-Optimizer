#!/usr/bin/env python3
"""
Two-Mirror Telescope Simulation
Demonstrates multivariable calculus optimization of mirror shapes
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

def get_shape_name(e):
    """Get conic section name from eccentricity"""
    if e == 0.0:
        return "Sphere"
    elif e == 1.0:
        return "Parabola"
    elif e < 1.0:
        return "Ellipse"
    else:
        return "Hyperbola"

def conic_sag(x, y, e, R, offset=0):
    """Conic surface equation: z = f(x,y,e,R)"""
    r2 = x**2 + y**2
    if abs(e) < 1e-10:
        if r2 > R**2:
            return np.nan
        return offset + (R - np.sqrt(R**2 - r2))
    elif abs(e - 1.0) < 1e-6:
        return offset + r2 / (2*R)
    else:
        k = -e**2
        c = 1.0/R
        disc = 1 - (1+k)*c**2*r2
        if disc < 0:
            return np.nan
        return offset + c*r2/(1 + np.sqrt(disc))

def conic_normal(x, y, e, R):
    """Surface normal vector"""
    h = 1e-6
    dzdx = (conic_sag(x+h, y, e, R, 0) - conic_sag(x-h, y, e, R, 0))/(2*h)
    dzdy = (conic_sag(x, y+h, e, R, 0) - conic_sag(x, y-h, e, R, 0))/(2*h)
    n = np.array([-dzdx, -dzdy, 1.0])
    return n / np.linalg.norm(n)

def reflect(d, n):
    """Reflection law: r = d - 2(d·n)n"""
    return d - 2*np.dot(d, n)*n

def trace_ray(x_ray, y_ray, config, use_secondary=True, field_angle=0.0):
    """Trace single ray through telescope.
    
    field_angle: off-axis field angle in degrees (tilts incoming ray).
    """
    path = []
    theta = np.deg2rad(field_angle)
    
    prim = config['primary']
    z_prim = conic_sag(x_ray, y_ray, prim['e'], prim['R'], prim['position'])
    
    if np.isnan(z_prim):
        return None, None
    
    z_start = z_prim + 2.0
    p_start = np.array([x_ray, y_ray, z_start])
    p_prim = np.array([x_ray, y_ray, z_prim])
    d_inc = np.array([np.sin(theta), 0, -np.cos(theta)])
    
    path.append(('incident', p_start, p_prim))
    
    n_prim = conic_normal(x_ray, y_ray, prim['e'], prim['R'])
    d_after_prim = reflect(d_inc, n_prim)
    
    if not use_secondary:
        if abs(d_after_prim[2]) > 1e-6:
            t = -z_prim / d_after_prim[2]
            if t > 0:
                p_final = p_prim + d_after_prim * t
                path.append(('primary_to_focus', p_prim, p_final))
                return path, p_final
        return path, None
    
    sec = config['secondary']
    sec_z = sec['position']
    
    if abs(d_after_prim[2]) > 1e-6:
        t = (sec_z - z_prim) / d_after_prim[2]
        if t > 0:
            p_at_sec = p_prim + d_after_prim * t
            x_sec, y_sec = p_at_sec[0], p_at_sec[1]
            
            if x_sec**2 + y_sec**2 <= sec['aperture']**2:
                z_sec = conic_sag(x_sec, y_sec, sec['e'], sec['R'], sec_z)
                if not np.isnan(z_sec):
                    p_sec = np.array([x_sec, y_sec, z_sec])
                    path.append(('primary_to_secondary', p_prim, p_sec))
                    
                    n_sec = conic_normal(x_sec, y_sec, sec['e'], sec['R'])
                    d_after_sec = reflect(d_after_prim, n_sec)
                    
                    if abs(d_after_sec[2]) > 1e-6:
                        t_final = -z_sec / d_after_sec[2]
                        if t_final > 0:
                            p_final = p_sec + d_after_sec * t_final
                            path.append(('secondary_to_focus', p_sec, p_final))
                            return path, p_final
    
    return path, None

def calculate_rms(config, n_rays=9, field_angle=0.0, use_secondary=True):
    """Calculate RMS spot size"""
    aperture = config['primary']['aperture']
    x_positions = np.linspace(-aperture*0.9, aperture*0.9, n_rays)
    focal_points = []
    
    for x_ray in x_positions:
        if use_secondary and abs(x_ray) < config['secondary']['aperture']:
            continue
        
        path, focal = trace_ray(x_ray, 0, config, use_secondary, field_angle)
        if focal is not None:
            focal_points.append(focal[:2])
    
    if len(focal_points) < 3:
        return 999.0
    
    focal_points = np.array(focal_points)
    centroid = focal_points.mean(axis=0)
    distances = np.sqrt(((focal_points - centroid)**2).sum(axis=1))
    return np.sqrt((distances**2).mean())

def calculate_combined_aberration(config, w_sph=0.6, w_coma=0.4, use_secondary=True):
    """Combined spherical + coma aberration"""
    rms_sph = calculate_rms(config, n_rays=9, field_angle=0.0, use_secondary=use_secondary)
    rms_coma = calculate_rms(config, n_rays=7, field_angle=3.0, use_secondary=use_secondary)
    return w_sph * rms_sph + w_coma * rms_coma, rms_sph, rms_coma

def optimize_system(config_template, optimize_params, w_sph=0.6, w_coma=0.4, 
                   learning_rate=0.05, max_iter=50):
    """Gradient descent optimization"""
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in config_template.items()}
    history = []
    
    param_bounds = {'e': (0.0, 2.0), 'R': (0.5, 5.0), 'position': (0.5, 2.5)}
    
    for iteration in range(max_iter):
        combined, sph, coma = calculate_combined_aberration(config, w_sph, w_coma)
        
        state = {param: config[mirror][param] for mirror, param in optimize_params}
        state.update({'iter': iteration, 'combined': combined, 'sph': sph, 'coma': coma})
        history.append(state)
        
        h = 0.005
        gradients = []
        
        for mirror, param in optimize_params:
            original = config[mirror][param]
            
            config[mirror][param] = original + h
            combined_plus, _, _ = calculate_combined_aberration(config, w_sph, w_coma)
            
            config[mirror][param] = original - h
            combined_minus, _, _ = calculate_combined_aberration(config, w_sph, w_coma)
            
            config[mirror][param] = original
            
            grad = (combined_plus - combined_minus) / (2*h)
            gradients.append(grad)
        
        grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
        if grad_magnitude < 0.0001:
            break
        
        adaptive_lr = learning_rate / (1 + 0.1 * iteration)
        
        for (mirror, param), grad in zip(optimize_params, gradients):
            bounds = param_bounds.get(param, (0.0, 2.0))
            new_val = config[mirror][param] - adaptive_lr * grad
            new_val = np.clip(new_val, bounds[0], bounds[1])
            config[mirror][param] = new_val
        
        new_combined, _, _ = calculate_combined_aberration(config, w_sph, w_coma)
        if new_combined >= combined and iteration > 5:
            learning_rate *= 0.8
    
    final_combined, final_sph, final_coma = calculate_combined_aberration(config, w_sph, w_coma)
    history.append({
        **{param: config[mirror][param] for mirror, param in optimize_params},
        'iter': len(history),
        'combined': final_combined,
        'sph': final_sph,
        'coma': final_coma
    })
    
    return config, history

def plot_telescope(config, n_rays=9, show_ray_types=None, show_secondary=True):
    """Create telescope visualization with FIXED bounds"""
    if show_ray_types is None:
        show_ray_types = {'incident', 'primary_reflected', 'secondary_reflected'}
    
    fig = go.Figure()
    
    prim = config['primary']
    sec = config['secondary']
    
    x_prim = np.linspace(-prim['aperture'], prim['aperture'], 200)
    z_prim = [conic_sag(x, 0, prim['e'], prim['R'], prim['position']) for x in x_prim]
    z_prim = [z if not np.isnan(z) else None for z in z_prim]
    
    prim_name = get_shape_name(prim['e'])
    fig.add_trace(go.Scatter(
        x=x_prim, y=z_prim,
        mode='lines',
        line=dict(color='#2E86AB', width=4),
        name=f'Primary: {prim_name} (e₁={prim["e"]:.2f})'
    ))
    
    x_sec = np.linspace(-sec['aperture'], sec['aperture'], 100)
    z_sec = [conic_sag(x, 0, sec['e'], sec['R'], sec['position']) for x in x_sec]
    z_sec = [z if not np.isnan(z) else None for z in z_sec]
        
    sec_name = get_shape_name(sec['e'])
    fig.add_trace(go.Scatter(
        x=x_sec, y=z_sec,
        mode='lines',
        line=dict(color='#E63946', width=4),
        name=f'Secondary: {sec_name} (e₂={sec["e"]:.2f})'
        ))
    
    show_any_rays = len(show_ray_types) > 0
    if show_any_rays:
        aperture = prim['aperture']
        x_rays = np.linspace(-aperture*0.9, aperture*0.9, n_rays)
        
        for x_ray in x_rays:
            if show_secondary and abs(x_ray) < sec['aperture']:
                continue
            
            path, focal = trace_ray(x_ray, 0, config, show_secondary)
            
            if path:
                colors = {
                    'incident': '#FF6B9D',
                    'primary_to_secondary': '#06FFA5',
                    'primary_to_focus': '#06FFA5',
                    'secondary_to_focus': '#FFB703'
                }
                
                type_mapping = {
                    'incident': 'incident',
                    'primary_to_secondary': 'primary_reflected',
                    'primary_to_focus': 'primary_reflected',
                    'secondary_to_focus': 'secondary_reflected'
                }
                
                for segment_type, p1, p2 in path:
                    display_type = type_mapping.get(segment_type, None)
                    if display_type and display_type in show_ray_types:
                        fig.add_trace(go.Scatter(
                            x=[p1[0], p2[0]],
                            y=[p1[2], p2[2]],
                            mode='lines',
                            line=dict(color=colors.get(segment_type, 'gray'), width=1.5),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        focal_points = []
        for x_ray in x_rays:
            if show_secondary and abs(x_ray) < sec['aperture']:
                continue
            path, focal = trace_ray(x_ray, 0, config, show_secondary)
            if focal is not None:
                focal_points.append(focal)
        
        if len(focal_points) > 2:
            focal_points = np.array(focal_points)
            fig.add_trace(go.Scatter(
                x=focal_points[:, 0],
                y=focal_points[:, 2],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name='Focal Points',
                hovertemplate='x=%{x:.4f}, z=%{y:.4f}<extra></extra>'
            ))
            
            centroid = focal_points.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[centroid[0]],
                y=[centroid[2]],
                mode='markers',
                marker=dict(color='purple', size=12, symbol='circle'),
                name='Ideal Focus'
            ))
    
    fig.add_hline(y=0, line=dict(color='gray', dash='dash', width=1))
    
    combined, sph, coma = calculate_combined_aberration(config, use_secondary=show_secondary)
    
    if show_secondary:
        title_text = f'<b>{prim_name} + {sec_name} Telescope</b><br>' + \
                     f'<sub>e₁={prim["e"]:.2f}, e₂={sec["e"]:.2f} | Sph={sph:.4f}, Coma={coma:.4f}</sub>'
    else:
        title_text = f'<b>Single {prim_name} Mirror</b><br>' + \
                     f'<sub>e₁={prim["e"]:.2f} | Sph={sph:.4f}</sub>'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='<b>Position (x)</b>',
        yaxis_title='<b>Height (z)</b>',
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            range=[-1.5, 1.5],
            fixedrange=False
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            range=[-0.3, 2.0],
            fixedrange=False
        ),
        uirevision='constant'
    )
    
    return fig

def compute_spot_points(config, field_angle=0.0, n_rays=7, use_secondary=True):
    """Generate focal plane intercepts for a grid of rays."""
    prim = config['primary']
    sec = config['secondary']
    aperture = prim['aperture']
    radii = np.linspace(0, aperture*0.9, n_rays)
    angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    points = []
    
    for r in radii:
        for ang in angles:
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            if use_secondary and np.hypot(x, y) < sec['aperture']:
                continue
            _, focal = trace_ray(x, y, config, use_secondary=use_secondary, field_angle=field_angle)
            if focal is not None:
                points.append(focal[:2])
    return np.array(points) if points else np.empty((0, 2))

def plot_spot_diagram(config, n_rays=7, show_secondary=True, coma_field_angle=3.0):
    """Spot diagram with on-axis (spherical) and off-axis (coma) rays."""
    on_axis = compute_spot_points(config, field_angle=0.0, n_rays=n_rays, use_secondary=show_secondary)
    off_axis = compute_spot_points(config, field_angle=coma_field_angle, n_rays=n_rays, use_secondary=show_secondary)
    
    fig = go.Figure()
    if len(on_axis):
        fig.add_trace(go.Scatter(
            x=on_axis[:, 0], y=on_axis[:, 1],
            mode='markers', name='On-axis (spherical)',
            marker=dict(color='#2E86AB', size=6, symbol='circle')
        ))
    if len(off_axis):
        fig.add_trace(go.Scatter(
            x=off_axis[:, 0], y=off_axis[:, 1],
            mode='markers', name=f'Off-axis {coma_field_angle}° (coma)',
            marker=dict(color='#E63946', size=6, symbol='x')
        ))
    
    if len(on_axis) + len(off_axis) > 0:
        all_pts = np.vstack([p for p in (on_axis, off_axis) if len(p)])
        span = max(0.05, np.max(np.abs(all_pts)) * 1.2)
    else:
        span = 0.1
    
    fig.add_shape(type='circle', x0=-span, y0=-span, x1=span, y1=span,
                  line=dict(color='lightgray', dash='dash'))
    
    fig.update_layout(
        title='Spot Diagram (focal plane)',
        xaxis_title='x @ focus',
        yaxis_title='y @ focus',
        xaxis=dict(scaleanchor='y', scaleratio=1, range=[-span, span], gridcolor='lightgray'),
        yaxis=dict(range=[-span, span], gridcolor='lightgray'),
        height=320,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
    )
    return fig

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Two-Mirror Telescope", style={'margin': '0'}),
        html.P("using Gradients to optimize mirror design", style={'margin': '5px 0', 'fontSize': '14px'})
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa',
             'borderBottom': '3px solid #2E86AB'}),
    
    html.Div([
        # Left panel
        html.Div([
            html.H3("Primary Mirror"),
            html.Label("Eccentricity e₁:"),
            dcc.Slider(id='e_primary', min=0.0, max=2.0, step=0.001, value=1.0,
                      marks={0: '0', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2'},
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Label("Curvature R₁:", style={'marginTop': '10px'}),
            dcc.Slider(id='R_primary', min=0.5, max=5.0, step=0.01, value=3.0,
                      marks={0.5: '0.5', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Hr(),
            
            html.H3("Secondary Mirror"),
            html.Label("Eccentricity e₂:"),
            dcc.Slider(id='e_secondary', min=0.0, max=2.0, step=0.001, value=1.5,
                      marks={0: '0', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2'},
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Label("Curvature R₂:", style={'marginTop': '10px'}),
            dcc.Slider(id='R_secondary', min=0.1, max=2.0, step=0.01, value=0.6,
                      marks={0.1: '0.1', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2'},
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Hr(),
            
            html.H3("Ray Settings"),
            html.Label("Number of rays:"),
            dcc.Slider(id='n_rays', min=5, max=25, step=2, value=17,
                      marks={5: '5', 11: '11', 17: '17', 23: '23', 25: '25'},
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Label("Ray Display:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Checklist(
                id='ray_options',
                options=[
                    {'label': ' Incident (Pink)', 'value': 'incident'},
                    {'label': ' Primary Reflected (Green)', 'value': 'primary_reflected'},
                    {'label': ' Secondary Reflected (Yellow)', 'value': 'secondary_reflected'}
                ],
                value=['incident', 'primary_reflected', 'secondary_reflected'],
                style={'fontSize': '13px'}
            ),
            
            html.Hr(),
            
            html.H3("Optimization"),
            dcc.Checklist(
                id='opt_params',
                options=[
                    {'label': ' Optimize e₁ (Primary eccentricity)', 'value': 'e1'},
                    {'label': ' Optimize e₂ (Secondary eccentricity)', 'value': 'e2'},
                    {'label': ' Optimize R₁ (Primary curvature)', 'value': 'R1'},
                    {'label': ' Optimize R₂ (Secondary curvature)', 'value': 'R2'}
                ],
                value=['e1', 'e2', 'R1', 'R2'],
                style={'fontSize': '13px', 'marginBottom': '10px'}
            ),
            
            html.Br(),
            html.Hr(style={'margin': '15px 0'}),
            
            html.Button('Optimize', id='opt_btn', n_clicks=0,
                       style={'width': '100%', 'padding': '15px', 'fontSize': '16px',
                             'backgroundColor': '#E63946', 'color': 'white',
                             'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
            
            html.Div(id='opt_result', style={'marginTop': '15px', 'padding': '15px',
                    'backgroundColor': '#fff3cd', 'borderRadius': '5px',
                    'fontSize': '12px', 'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'})
            
        ], style={'width': '320px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRight': '2px solid #dee2e6', 'overflowY': 'auto', 'height': '100vh'}),
        
        # Right panel
        html.Div([
            dcc.Graph(id='telescope_plot', style={'height': '600px'}),
            dcc.Graph(id='spot_plot', style={'height': '340px'})
        ], style={'flex': 1, 'padding': '20px'})
        
    ], style={'display': 'flex'})
])

@app.callback(
    Output('telescope_plot', 'figure'),
    Output('spot_plot', 'figure'),
    Input('e_primary', 'value'),
    Input('e_secondary', 'value'),
    Input('R_primary', 'value'),
    Input('R_secondary', 'value'),
    Input('ray_options', 'value'),
    Input('n_rays', 'value')
)
def update_plot(e1, e2, R1, R2, ray_options, n_rays):
    show_ray_types = set(ray_options)
    show_secondary = True
    
    config = {
        'primary': {'e': e1, 'R': R1, 'aperture': 1.0, 'position': 0.0},
        'secondary': {'e': e2, 'R': R2, 'aperture': 0.25, 'position': 1.4}
    }
    
    fig = plot_telescope(config, n_rays=n_rays, show_ray_types=show_ray_types, 
                        show_secondary=show_secondary)
    spot_fig = plot_spot_diagram(config, n_rays=5, show_secondary=show_secondary, coma_field_angle=3.0)
    
    combined, sph, coma = calculate_combined_aberration(config, w_sph=0.5, w_coma=0.5, use_secondary=show_secondary)
    
    return fig, spot_fig

@app.callback(
    Output('opt_result', 'children'),
    [Output('e_primary', 'value', allow_duplicate=True),
     Output('e_secondary', 'value', allow_duplicate=True),
     Output('R_primary', 'value', allow_duplicate=True),
     Output('R_secondary', 'value', allow_duplicate=True)],
    Input('opt_btn', 'n_clicks'),
    State('e_primary', 'value'),
    State('e_secondary', 'value'),
    State('R_primary', 'value'),
    State('R_secondary', 'value'),
    State('opt_params', 'value'),
    prevent_initial_call=True
)
def run_optimization(n_clicks, e1, e2, R1, R2, opt_params):
    if n_clicks == 0:
        return "", e1, e2, R1, R2
    
    config = {
        'primary': {'e': e1, 'R': R1, 'aperture': 1.0, 'position': 0.0},
        'secondary': {'e': e2, 'R': R2, 'aperture': 0.25, 'position': 1.4}
    }
    
    w_sph, w_coma = 0.5, 0.5
    
    optimize_params = []
    if 'e1' in opt_params:
        optimize_params.append(('primary', 'e'))
    if 'e2' in opt_params:
        optimize_params.append(('secondary', 'e'))
    if 'R1' in opt_params:
        optimize_params.append(('primary', 'R'))
    if 'R2' in opt_params:
        optimize_params.append(('secondary', 'R'))
    
    if not optimize_params:
        return "Select at least one parameter", e1, e2, R1, R2
    
    opt_config, history = optimize_system(config, optimize_params, w_sph, w_coma)
    method_name = "Gradient Descent"
    
    result_lines = ["Optimization Results:",
                    f"Method: {method_name}",
                    f"Iterations: {len(history)-1}\n"]
    
    if 'e1' in opt_params or 'R1' in opt_params:
        result_lines.append(f"PRIMARY MIRROR:")
        if 'e1' in opt_params:
            e1_start = history[0].get('e', e1)
            e1_final = opt_config['primary']['e']
            result_lines.append(f"  e₁: {e1_start:.4f} → {e1_final:.4f}")
        if 'R1' in opt_params:
            R1_start = history[0].get('R', config['primary']['R'])
            R1_final = opt_config['primary']['R']
            result_lines.append(f"  R₁: {R1_start:.4f} → {R1_final:.4f}")
        result_lines.append("")
    
    if 'e2' in opt_params or 'R2' in opt_params:
        result_lines.append(f"SECONDARY MIRROR:")
        if 'e2' in opt_params:
            e2_start = history[0].get('e', e2)
            e2_final = opt_config['secondary']['e']
            result_lines.append(f"  e₂: {e2_start:.4f} → {e2_final:.4f}")
        if 'R2' in opt_params:
            R2_start = history[0].get('R', config['secondary']['R'])
            R2_final = opt_config['secondary']['R']
            result_lines.append(f"  R₂: {R2_start:.4f} → {R2_final:.4f}")
        result_lines.append("")
    
    result_lines.append(f"ABERRATIONS:")
    result_lines.append(f"  Sph:  {history[0]['sph']:.5f} → {history[-1]['sph']:.5f}")
    result_lines.append(f"  Coma: {history[0]['coma']:.5f} → {history[-1]['coma']:.5f}")
    result_lines.append(f"  Combined: {history[0]['combined']:.5f} → {history[-1]['combined']:.5f}\n")
    
    improvement = ((history[0]['combined']-history[-1]['combined'])/history[0]['combined']*100)
    result_lines.append(f"Improvement: {improvement:.1f}%\n")
    
    result = "\n".join(result_lines)
    
    return result, opt_config['primary']['e'], opt_config['secondary']['e'], \
           opt_config['primary']['R'], opt_config['secondary']['R']

if __name__ == "__main__":
    print("Open on browser: http://localhost:8056")
    print("="*60 + "\n")
    app.run(debug=True, port=8056)

