import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import math
import plotly.graph_objects as go

def sdsi(dw,re,rd,mu2):
    return 4*mu2*re*rd/(dw**4 + dw**2*((1+rd)**2+(1+re)**2-2*mu2)+(mu2+(1+re)*(1+rd))**2)

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    children=[
        html.H1("Maximum drop-port transmission contour", style={'textAlign': 'center'}),
        html.H2(children="Paper: New passband shapes that minimize the insertion loss of coupled-resonator bandpass filters", style={'textAlign': 'center'}),
        html.Label("Enter the normalized bandwidth Δω/ro then press PLOT: "),
        dcc.Input(id="width", type="number", value=6),
        html.Button(id='plot-button-state', n_clicks=0, children='PLOT'),
        dcc.Graph(id="contour-graph"),
        html.Br(),

        html.H2("Spectrum Optimization"),
        html.Label("For Δω/ro being specified above, enter the values for parameters. Specify both S and M, OR any two in re, rd and µ^2:"),
        html.Div([
            html.Div(children=[
                html.Label("S : "),
                dcc.Input(id="svalue", type="number", min=-0.9999999, max=1, value=0),
                html.Br(),html.Br(),
                html.Div(dcc.Slider(id="svalue_slider",step=0.001,min=-0.999,max=1,value=0,marks={-0.999:"-1",0:"0",1:"1"},updatemode="drag"),style={'width': '30%'}),
                html.Label("M: "),
                dcc.Input(id="mvalue", type="number", min=-0.9999999, max=1, value=0),
                html.Br(),html.Br(),
                html.Div(dcc.Slider(id="mvalue_slider",step=0.001,min=-0.999,max=1,value=0,marks={-0.999:"-1",0:"0",1:"1"},updatemode="drag"),style={'width': '30%'}),
                html.Div(id="check_s_m_value"),
            ], style={'padding': 20, 'flex': 1}),

            html.Div(children=[
                html.Label("re: "),
                dcc.Input(id="revalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label("rd: "),
                dcc.Input(id="rdvalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label("µ^2: "),
                dcc.Input(id="mu2value", type="number"),
            ], style={'padding': 20, 'flex': 1})
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.Div(style={'display': 'flex', 'justify-content': 'center'},children=[dcc.Graph(id="spectrum",style={'width': '60%'})]),
        #html.Label("Best M for the above given S value is"),
        # html.Div(id="optimized_m_value"),
        #html.Br(),
        html.P("For above chosen S & M values (blue),"),
        html.Div(id="IL_current", style={'textAlign': 'center'}),
        html.P("For the S value above, best M is (red):"),
        html.Div(id="optimized_m_value", style={'textAlign': 'center'}),
        html.P("and it gives"),
        html.Div(id="IL_best_M", style={'textAlign': 'center'}),
        # html.P("For the specified Δω/ro, lowest IL comes from"),
        # html.Div(id="optimized_s_value", style={'textAlign': 'center'}),
        # html.Div(id="IL_best_S", style={'textAlign': 'center'}),

        html.Br(),html.Br(),html.Br(),html.Br()
    ],
    style={'margin-left': '50px','margin-right': '50px'}
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

# Define the callback function to update the spectrum
@app.callback(
    Output("check_s_m_value", "children"),
    Output("IL_current", "children"),
    Output("IL_best_M","children"),
    Output("spectrum", "figure"),
    Output("optimized_m_value", "children"),
    Output("svalue", "value"),
    Output("mvalue", "value"),
    Output("svalue_slider", "value"),
    Output("mvalue_slider", "value"),
    Output("revalue", "value"),
    Output("rdvalue", "value"),
    Output("mu2value", "value"),
    Input("plot-button-state","n_clicks"),
    Input("svalue", "value"),
    Input("mvalue", "value"),
    Input("svalue_slider", "value"),
    Input("mvalue_slider", "value"),
    Input("revalue", "value"),
    Input("rdvalue", "value"),
    Input("mu2value", "value"),
    State("width", "value"),
)
def update_spectrum(contour_click,s_input,m_input,s_slider,m_slider,re_input,rd_input,mu2_input,w):

    dwrange = np.linspace(-2.5*w, 2.5*w, 500)

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    s = s_input
    m = m_input
    if trigger_id == "svalue":
        s = s_input
        s_slider_value = s
        m = m_input
        m_slider_value = m
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
    elif trigger_id == "mvalue":
        m = m_input
        m_slider_value = m
        s = s_input
        s_slider_value = s
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
    elif trigger_id == "svalue_slider":
        s_slider_value = s_slider
        s = s_slider_value
        m = m_input
        m_slider_value = m
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
    elif trigger_id == "mvalue_slider":
        m_slider_value = m_slider
        m = m_slider_value
        s = s_input
        s_slider_value = s
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
    elif trigger_id == "revalue":
        re = re_input
        rd = rd_input
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
        s = (4*mu2-(re-rd)**2)/((re+rd+2)**2)
        s_slider_value = s
        m = (re-rd)/(re+rd+2)
        m_slider_value = m
    elif trigger_id == "rdvalue":
        rd = rd_input
        re = re_input
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)
        s = (4*mu2-(re-rd)**2)/((re+rd+2)**2)
        s_slider_value = s
        m = (re-rd)/(re+rd+2)
        m_slider_value = m
    elif trigger_id == "mu2value":
        mu2 = mu2_input
        rd = rd_input
        re = re_input
        s = (4*mu2-(re-rd)**2)/((re+rd+2)**2)
        s_slider_value = s
        m = (re-rd)/(re+rd+2)
        m_slider_value = m
    else: # changing width
        s = s_input
        s_slider_value = s
        m = m_input
        m_slider_value = m
        re = (1+m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        rd = (1-m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        mu2 = w/2 * math.sqrt(w*w/2 + (2+re+rd)**2) - w*w/4 - (1+re)*(1+rd)

    # Check S and M values is valid or not
    if np.greater(m*m,-s) and np.greater((1 - 2 / w * np.sqrt(s - 1 + np.sqrt(2 * s**2 + 2))), np.abs(m)):
        check_s_m = 'Valid.'
        
        # Compute the transfer function
        transf = sdsi(dwrange,re,rd,mu2)
        insertionloss = 10.0*math.log10(sdsi(0.0,re,rd,mu2))
        # Compute the optimized M and transfer function
        optimized_m2 = -(s-1)/2 + 2/(w*w)*(s-1+math.sqrt(2*s*s+2)) - 2/w*math.sqrt(s-1+math.sqrt(2*s*s+2))
        if optimized_m2 > 0:
            optimized_m = math.sqrt(optimized_m2)
        else:
            optimized_m = 0
        optimized_re = (1+optimized_m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        optimized_rd = (1-optimized_m) * w / (2*math.sqrt(s-1+math.sqrt(2*s*s+2))) - 1
        optimized_mu2 = w/2 * math.sqrt(w*w/2 + (2+optimized_re+optimized_rd)**2) - w*w/4 - (1+optimized_re)*(1+optimized_rd)
        optimized_transf = sdsi(dwrange,optimized_re,optimized_rd,optimized_mu2)
        bestMinsertionloss = 10.0*math.log10(sdsi(0,optimized_re,optimized_rd,optimized_mu2))
    else:
        check_s_m = 'INVALID! No spectrum is generated.'
        transf = np.zeros(500)
        insertionloss = np.nan
        optimized_m = np.nan
        optimized_transf = np.zeros(500)
        bestMinsertionloss = np.nan

    # Plot the spectrum
    spectrum_plot = go.Scatter(
        x=dwrange,
        y=transf,
        mode='lines',
        name="Spectrum with <br>given S & M"
    )
    o_spectrum_plot = go.Scatter(
        x=dwrange,
        y=optimized_transf,
        mode='lines',
        name="Optimized spectrum <br>for given S"
    )

    # Set the layout of the graph
    layout = go.Layout(
        title="Spectrum",
        xaxis=dict(title="Δω/ro"), # domain=[0, 0.475],
        yaxis=dict(title="|sd/si|^2"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        legend=dict(
            traceorder='normal',itemwidth=30  # Adjust the item width as per your preference
        )
    )

    # Create the figure
    fig = go.Figure(data=[spectrum_plot,o_spectrum_plot], layout=layout)
    
    return 'The chosen S and M value is {}'.format(check_s_m), 'Insertion Loss = {:.2f} dB.'.format(insertionloss), 'Insertion Loss = {:.2f} dB.'.format(bestMinsertionloss), fig, 'M= ±{},'.format(optimized_m), s, m, s_slider_value, m_slider_value,re,rd,mu2

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
