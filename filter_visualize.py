import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px

def sdsi(dw,re,rd,mu2):
    return 4*mu2*re*rd/(dw**4 + dw**2*((1+rd)**2+(1+re)**2-2*mu2)+(mu2+(1+re)*(1+rd))**2)

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    children=[
        html.H1("Maximum drop-port transmission contour", style={'textAlign': 'center'}),
        html.H2(children="Paper: New passband shapes that minimize the insertion loss of coupled-resonator bandpass filters", style={'textAlign': 'center'}),
        html.Label("Enter the normalized bandwidth Δω/ro then press PLOT:"),
        dcc.Input(id="width", type="number", value=6),
        html.Button(id='plot-button-state', n_clicks=0, children='PLOT'),
        dcc.Graph(id="contour-graph"),
        html.Br(),
        html.H3("Arbitrary spectrum spectrum for given S and M (left):"),
        html.Label("Enter the S and M values:"),
        dcc.Input(id="svalue", type="number", value=0),
        dcc.Input(id="mvalue", type="number", value=0),
        html.Button(id='spectrum-button-state', n_clicks=0, children='PLOT'),
        html.Br(),
        html.Br(),
        html.Div(id="check_s_m_value"),
        dcc.Graph(id="spectrum"),
        #html.Label("Best M for the above given S value is"),
        html.Div(id="optimized_m_value"),
        html.P("which gives a spectrum as above (right)."),
    ]
)

# Define the callback function to update the contour plot only after press "PLOT"
@app.callback(
    Output("contour-graph", "figure"),
    Input("plot-button-state","n_clicks"),
    State("width", "value")
)
def update_contour_plot(n_clicks,w):
    if n_clicks is None:
        return go.Figure()

    # Generate S and M values
    s = np.linspace(-0.999999, 1, 500)
    m = np.linspace(-1, 1, 500)
    S, M = np.meshgrid(s, m)
    
    # Compute the D function
    D = 4 * (S + M**2)/((S + 1)**2) * (1 - M**2 - 4/w * np.sqrt(S - 1 + np.sqrt(2*(S**2 + 1))) + 4/(w**2) * (S - 1 + np.sqrt(2*(S**2 + 1))))
    
    # Red region, eq. S15
    Red = np.greater(M**2,-S)

    # Gray region, eq. S16
    Graycheck = 1 - 2 / w * np.sqrt(S - 1 + np.sqrt(2 * S**2 + 2))
    Gray = np.greater(Graycheck, np.abs(M))

    # modify D according to the two restrictions:
    restric = np.logical_and(Red, Gray) # mask array
    D[~restric] = -1

    # always need 6 contour lines, self-determine distances
    contourSize = np.nanmax(D) / 6.0

    # Plot the contour plot
    contour_plot = go.Contour(
        x=s,
        y=m,
        z=D,
        #colorscale="",
        contours=dict(start=0, end=np.nanmax(D), size = contourSize),
        hovertemplate="S=%{x:.3f}<br>M=%{y:.3f}<br>D=%{z:.3f}<extra></extra>",
        #showscale=False
    )
    
    # Set the layout of the graph
    layout = go.Layout(
        title="Contour for D (D=-1 means unphysical device with negative coupling, see Eq. S15 & S16)",
        xaxis=dict(title="Shape Parameter S"),
        yaxis=dict(title="Impedence Match M")
    )
    
    # Create the figure
    fig = go.Figure(data=[contour_plot], layout=layout)
    
    return fig

# Define the callback function to update the spectrum only after press "PLOT"
@app.callback(
    Output("check_s_m_value", "children"),
    Output("spectrum", "figure"),
    Output("optimized_m_value", "children"),
    Input("spectrum-button-state","n_clicks"),
    Input("plot-button-state","n_clicks"),
    State("width", "value"),
    State("svalue", "value"),
    State("mvalue", "value")
)
def update_spectrum(contour_click,spectrum_click,w,s,m):
    if (contour_click is None) or (spectrum_click is None):
        return go.Figure(), ''

    dwrange = np.linspace(-2.5*w, 2.5*w, 500)
    # Check S and M values is valid or not
    if np.greater(m*m,-s) and np.greater((1 - 2 / w * np.sqrt(s - 1 + np.sqrt(2 * s**2 + 2))), np.abs(m)):
        check_s_m = 'Valid.'
        # Generate re, rd, mu2 based on S and M values
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
        # Compute the transfer function
        transf = sdsi(dwrange,re,rd,mu2)
        # Compute the optimized M and transfer function
        optimized_m2 = -(s-1)/2 + 2/(w*w)*(s-1+math.sqrt(2*s*s+2)) - 2/w*math.sqrt(s-1+math.sqrt(2*s*s+2))
        if optimized_m2 > 0:
            optimized_m = math.sqrt(optimized_m2)
        else:
            optimized_m = 0
        optimized_re = (1+optimized_m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) + 1
        optimized_rd = (1-optimized_m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        optimized_mu2 = w/2 * math.sqrt(w*w/2 + (2+optimized_re+optimized_rd)**2) - w*w/4 - (1+optimized_re)*(1+optimized_rd)
        optimized_transf = sdsi(dwrange,optimized_re,optimized_rd,optimized_mu2)
    else:
        check_s_m = 'INVALID!'
        transf = np.zeros(500)
        optimized_m = np.nan
        optimized_transf = np.zeros(500)

    # Plot the spectrum
    spectrum_plot = go.Scatter(
        x=dwrange,
        y=transf,
        mode='lines',
        name="Spectrum with given S & M"
    )
    o_spectrum_plot = go.Scatter(
        x=dwrange,
        y=optimized_transf,
        mode='lines',
        xaxis="x2",
        name="Optimized spectrum for given S"
    )

    # Set the layout of the graph
    layout = go.Layout(
        title="Spectrum",
        xaxis=dict(domain=[0, 0.475],title="Δω/ro"),
        xaxis2=dict(domain=[0.525, 1],title="Δω/ro"),
        yaxis=dict(title="|sd/si|^2")
    )

    # Create the figure
    fig = go.Figure(data=[spectrum_plot,o_spectrum_plot], layout=layout)
    
    return 'The chosen S and M value is {}'.format(check_s_m), fig, 'Best M for the above given S value is "±{}",'.format(optimized_m)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
