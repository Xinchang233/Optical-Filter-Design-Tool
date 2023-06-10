import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import math
import plotly.graph_objects as go

def sdsi(dw,re,rd,mu2):
    return 4*mu2*re*rd/(dw**4 + dw**2*((1+rd)**2+(1+re)**2-2*mu2)+(mu2+(1+re)*(1+rd))**2)

def circle_equation(x, radius):
    return np.sqrt(radius - x * x)

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    children=[
        html.H1(children="New passband shapes that minimize the insertion loss of coupled-resonator bandpass filters", style={'textAlign': 'center'}),

        html.H3("Spectrum Optimization"),
        html.Label(children=["Enter the normalized bandwidth Δω/2r",html.Sub(children="o")," then press CONFIRM: "]),
        dcc.Input(id="width", type="number", value=3), # should be half-width, just change from w/r0 to w/2r0
        html.Button(id='plot-button-state', n_clicks=0, children='CONFIRM'),
        html.Br(),html.Br(),
        html.Label(children=["Then specify EITHER (S, M) OR (r",html.Sub(children="e"),"/r",html.Sub(children="o"),", r",html.Sub(children="d"),"/r",html.Sub(children="o"),"):"]),
        html.Div([
            html.Div(children=[
                html.Label("S : "),
                dcc.Input(id="svalue", type="number", min=-0.9999999, max=1, value=0),
                html.Br(),html.Br(),
                html.Div(dcc.Slider(id="svalue_slider",step=0.001,min=-0.999,max=1,value=0,marks={-0.999:"-1",0:"0",1:"1"},updatemode="drag"),style={'width': '90%'}),
                html.Label("M: "),
                dcc.Input(id="mvalue", type="number", min=-0.9999999, max=1, value=0),
                html.Br(),html.Br(),
                html.Div(dcc.Slider(id="mvalue_slider",step=0.001,min=-0.999,max=1,value=0,marks={-0.999:"-1",0:"0",1:"1"},updatemode="drag"),style={'width': '90%'}),
                html.Br(),html.Br(),
                html.Div(id="check_s_m_value"),
                html.P("Valid range for M is: "),
                html.Div(id="rangeFORm"),
                html.Br(),
                html.Div(id="IL_current"),
            ], style={'padding': 15, 'flex': 3}),

            html.Div(children=[
                dcc.Graph(id="spectrum"),
            ], style={'padding': 1, 'flex': 5}),

            html.Div(children=[
                #dcc.Graph(id="illustration"),
                html.Label(children=["r",html.Sub(children="e"),"/r",html.Sub(children="o"),": "]),
                dcc.Input(id="revalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label(children=["r",html.Sub(children="d"),"/r",html.Sub(children="o"),": "]),
                dcc.Input(id="rdvalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label(children=[html.Span("µ"),"/r",html.Sub(children="o"),": ",html.Div(id="muvalue",style={"display": "inline-block"})]),
                html.Br(),html.Br(),
                html.Label("Graphical illustration: "),
                dcc.Graph(id="illustration"),
            ], style={'padding': 10, 'flex': 2}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.H3("Optimization Analysis"),
        html.Div([
            html.Div(children=[
                html.P("For the S value above, best M which gives lowes insertion loss (red curve)"),
                html.Div(id="optimized_m_value", style={'textAlign': 'center'}),
                html.P(children=["and its corresponding r",html.Sub(children="e"),", r",html.Sub(children="d"), ", ", html.Span("µ")," are"]),
                html.Div(children=["r",html.Sub(children="e"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_m_re1",style={"display": "inline-block"}),", ","r",html.Sub(children="d"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_m_rd1",style={"display": "inline-block"})," OR ","r",html.Sub(children="e"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_m_re2",style={"display": "inline-block"}),", ","r",html.Sub(children="d"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_m_rd2",style={"display": "inline-block"}),"; ","µ","/r",html.Sub(children="o")," = ",html.Div(id="optimized_m_mu",style={"display": "inline-block"}),], style={'textAlign': 'center'}),
                html.P("and it gives"),
                html.Div(id="IL_best_M", style={'textAlign': 'center'}),
            ], style={'padding-right': '10px', 'flex': 2}),

            html.Div(children=[
                html.P(children=["For the specified Δω/2r",html.Sub(children="o"),", lowest IL comes from"]),
                html.Div(id="optimized_s_value", style={'textAlign': 'center'}),
                html.Div(id="approximateORnot"),
                html.P(children=["Its corresponding r",html.Sub(children="e"),", r",html.Sub(children="d"), ", ", html.Span("µ")," are"]),
                html.Div(children=["r",html.Sub(children="e"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_s_re1",style={"display": "inline-block"}),", ","r",html.Sub(children="d"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_s_rd1",style={"display": "inline-block"})," OR ","r",html.Sub(children="e"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_s_re2",style={"display": "inline-block"}),", ","r",html.Sub(children="d"),"/r",html.Sub(children="o")," = ",html.Div(id="optimized_s_rd2",style={"display": "inline-block"}),"; ","µ","/r",html.Sub(children="o")," = ",html.Div(id="optimized_s_mu",style={"display": "inline-block"}),], style={'textAlign': 'center'}),
                #html.Div(id="optimized_s_rerd", style={'textAlign': 'center'}),
                html.P("and it gives"),
                html.Div(id="IL_best_S", style={'textAlign': 'center'}),
            ], style={'padding-right': '10px', 'flex': 2}),

            html.Div(children=[
                html.Div(children=[dcc.Graph(id="bodespectrum")]),
            ], style={'padding-right': '1px', 'flex': 1}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.H3("Contour Plot for Filter Transmission"),
        dcc.Graph(id="contour-graph"),
        html.Br(),

        html.H3(children=['Calculate r',html.Sub(children="o"),': ']),
        html.Label(children=["Input Q",html.Sub(children="o"),": "]),
        dcc.Input(id="Q0", type="number", value=1000000),
        html.Label(children=[",  Input ",html.Span("λ"),html.Sub(children="o"),"(nm): "]),
        dcc.Input(id="lambda0", type="number", value=1515),
        html.Label(children=[",  r",html.Sub(children="o")," = "]),
        html.Div(id="r0",style={"display": "inline-block"}),

        html.Br(),html.Br(),html.Br(),html.Br()
    ],
    style={'margin-left': '50px','margin-right': '50px'}
)

# Define the callback function to update the contour plot only after press "PLOT"
@app.callback(
    Output("r0","children"),
    Output("contour-graph", "figure"),
    Input("Q0","value"),
    Input("lambda0","value"),
    Input("plot-button-state","n_clicks"),
    State("width", "value")
)
def update_contour_plot(Q0,lambda0,n_clicks,w):
    if n_clicks is None:
        return go.Figure()
    
    # Calculate r0
    ro = np.pi * 2.99792458 / lambda0 * 100000000000000000 / Q0

    w = w*2.0
    # Generate S and M values
    s = np.linspace(-0.999999, 1, 1000)
    m = np.linspace(-1, 1, 1000)
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
        hovertemplate="S=%{x:.4f}<br>M=%{y:.4f}<br>D=%{z:.4f}<extra></extra>",
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
    
    return np.round(ro), fig

# Define the callback function to update the spectrum
@app.callback(
    Output("check_s_m_value", "children"),
    Output("rangeFORm", "children"),
    Output("IL_current", "children"),
    Output("IL_best_M","children"),
    Output("IL_best_S","children"),
    Output("spectrum", "figure"),
    Output("bodespectrum", "figure"),
    Output("illustration", "figure"),
    Output("optimized_m_value", "children"),
    # Output("optimized_m_rerd", "children"), # gives the sentence containing re, rd, mu^2
    Output("optimized_m_re1", "children"),
    Output("optimized_m_rd1", "children"),
    Output("optimized_m_re2", "children"),
    Output("optimized_m_rd2", "children"),
    Output("optimized_m_mu", "children"),
    Output("optimized_s_value", "children"),
    Output("approximateORnot", "children"), # if S = -1, take S = -0.9
    Output("optimized_s_re1", "children"),
    Output("optimized_s_rd1", "children"),
    Output("optimized_s_re2", "children"),
    Output("optimized_s_rd2", "children"),
    Output("optimized_s_mu", "children"),
    # Output("optimized_s_rerd", "children"), # gives the sentence containing re, rd, mu^2
    Output("svalue", "value"),
    Output("mvalue", "value"),
    Output("svalue_slider", "value"),
    Output("mvalue_slider", "value"),
    Output("revalue", "value"),
    Output("rdvalue", "value"),
    Output("muvalue", "children"),
    Input("plot-button-state","n_clicks"),
    Input("svalue", "value"),
    Input("mvalue", "value"),
    Input("svalue_slider", "value"),
    Input("mvalue_slider", "value"),
    Input("revalue", "value"),
    Input("rdvalue", "value"),
    State("width", "value"),
)
def update_spectrum(contour_click,s_input,m_input,s_slider,m_slider,re_input,rd_input,w):

    w = w*2.0 # check this. seems no influence because no return values related to w
    dwrange = np.linspace(-2.5*w, 2.5*w, 500)
    pdwranget = np.linspace(-1, 4.5, 250)
    pdwrange = np.power(10,pdwranget)

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
        ptransf = sdsi(pdwrange,re,rd,mu2)
        insertionloss = 10.0*math.log10(sdsi(0.0,re,rd,mu2))
    else:
        check_s_m = '''INVALID!'''
        transf = np.zeros(500)
        ptransf = np.zeros(250)
        insertionloss = np.nan

    # Extract valid M range for input S
    m_for_s = np.linspace(-1, 1, 2001)
    # Red region, eq. S15
    Red = np.greater(m_for_s**2,-s)
    # Gray region, eq. S16
    Graycheck = 1 - 2 / w * np.sqrt(s - 1 + np.sqrt(2 * s**2 + 2))
    Gray = np.greater(Graycheck, np.abs(m_for_s))
    # get mask array:
    restric = np.logical_and(Red, Gray)
    masked_m_for_s = np.ma.array(m_for_s, mask=~restric)
    if np.ma.flatnotmasked_edges(masked_m_for_s) is None:
        m_range = 'No valid M for chosen S'
    else:
        if s > 0:
            edges = np.ma.flatnotmasked_edges(masked_m_for_s)
            m_range = str(np.round(m_for_s[edges[0]],3))+' ≤ M ≤ '+str(np.round(m_for_s[edges[1]],3))
        else:
            edges = np.concatenate([np.ma.flatnotmasked_edges(masked_m_for_s[0:1000]),np.ma.flatnotmasked_edges(masked_m_for_s[1000:2000])])
            m_range = str(np.round(m_for_s[edges[0]],3))+' ≤ M ≤ '+str(np.round(m_for_s[edges[1]],3))+' and '+str(np.round(m_for_s[edges[2]+1000],3))+' ≤ M ≤ '+str(np.round(m_for_s[edges[3]+1000],3))
        #m_range=''

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
    optimized_ptransf = sdsi(pdwrange,optimized_re,optimized_rd,optimized_mu2)
    bestMinsertionloss = 10.0*math.log10(sdsi(0,optimized_re,optimized_rd,optimized_mu2))
    # calculate best S, at point B or A ?
    s_line = np.linspace(0, 1, 1001)
    m_line = 0
    D_line = 4 * (s_line + m_line**2)/((s_line + 1)**2) * (1 - m_line**2 - 4/w * np.sqrt(s_line - 1 + np.sqrt(2*(s_line**2 + 1))) + 4/(w**2) * (s_line - 1 + np.sqrt(2*(s_line**2 + 1))))
    # Red region, eq. S15
    Red = np.greater(0,-s_line)
    # Gray region, eq. S16
    Graycheck = 1 - 2 / w * np.sqrt(s_line - 1 + np.sqrt(2 * s_line**2 + 2))
    Gray = np.greater(Graycheck, 0)
    # modify D according to the two restrictions:
    restric = np.logical_and(Red, Gray) # mask array
    D_line[~restric] = -1
    D_max_line = np.nanmax(D_line) # max from the M=0, 0<S<1 line
    D_max_corner = (1-2/w)*(1-2/w) # max from the M=+-1, S=-1 corner
    if D_max_line > D_max_corner:
        best_s = s_line[np.nanargmax(D_line)]
        best_m = 0
        bestSinsertionloss = 10.0*math.log10(D_max_line)
        best_s_re = (1+0) * w / (2*math.sqrt(best_s-1+math.sqrt(2*best_s*best_s+2))) - 1
        best_s_rd = best_s_re
        best_s_mu2 = w/2 * math.sqrt(w*w/2 + (2+best_s_re+best_s_re)**2) - w*w/4 - (1+best_s_re)*(1+best_s_re)
        approximate = 'these values are achievable. So take them as it is.'
    else:
        best_s = -1
        best_m = '±1'
        # Compute the optimized M and transfer function
        app_optimized_m2 = -(-0.9-1)/2 + 2/(w*w)*(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2)) - 2/w*math.sqrt(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2))
        if app_optimized_m2 > 0:
            app_optimized_m = math.sqrt(app_optimized_m2)
        else:
            app_optimized_m = 0
        approximate = 'but S=-1 will lead to infinite coupling. So take S = -0.9, optimized M = ±'+str(np.round(app_optimized_m,3))
        best_s_re = (1+app_optimized_m) * w / (2*math.sqrt(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2))) - 1
        best_s_rd = (1-app_optimized_m) * w / (2*math.sqrt(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2))) - 1
        best_s_mu2 = w/2 * math.sqrt(w*w/2 + (2+best_s_re+best_s_rd)**2) - w*w/4 - (1+best_s_re)*(1+best_s_rd)
        bestSinsertionloss = 10.0*math.log10(sdsi(0,best_s_re,best_s_rd,best_s_mu2))

    # Plot the spectrum
    spectrum_plot = go.Scatter(
        x=dwrange,
        y=10.0*np.log10(transf),
        mode='lines',
        name="Spectrum <br>with<br>given S & M"
    )
    o_spectrum_plot = go.Scatter(
        x=dwrange,
        y=10.0*np.log10(optimized_transf),
        mode='lines',
        name="<br>Optimized <br>spectrum <br>for given S"
    )
    # Set the layout of the graph
    layout = go.Layout(
        title="Spectrum",
        xaxis=dict(title="Δω/Δω<sub>3dB"), # domain=[0, 0.475],
        yaxis=dict(title="10lg(|s<sub>d</sub>/s<sub>i</sub>|<sup>2)"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        legend=dict(
            traceorder='normal',itemwidth=30,xanchor='right',yanchor='top',bgcolor='rgba(0,0,0,0)' # Adjust the item width as per your preference
        )
    )
    # Create the figure
    fig = go.Figure(data=[spectrum_plot,o_spectrum_plot], layout=layout)
    fig.update_layout(title_text='Spectrum', title_x=0.5)

    # Plot illlustration figure |oo|
    scalefac = 4.0
    radius = 1.0
    illusfig = go.Figure()
    illusfig.update_xaxes(range=[-1.5,1.5], zeroline=False)
    illusfig.update_yaxes(range=[-4,4])
    if re>=0 and rd>=0 and mu2>=0:
        illusfig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-radius, y0=np.exp(-np.sqrt(mu2)/scalefac)/2, x1=+radius, y1=np.exp(-np.sqrt(mu2)/scalefac)/2+radius+radius,
            line_color="LightSeaGreen",
        )
        illusfig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-radius, y0=-np.exp(-np.sqrt(mu2)/scalefac)/2, x1=+radius, y1=-np.exp(-np.sqrt(mu2)/scalefac)/2-radius-radius,
            line_color="LightSeaGreen",
        )
        illusfig.add_shape(type="line",
            x0=-1.5, y0=np.exp(-np.sqrt(mu2)/scalefac)/2+radius+radius+np.exp(-re/scalefac), x1=1.5, y1=np.exp(-np.sqrt(mu2)/scalefac)/2+radius+radius+np.exp(-re/scalefac),
            line=dict(
                color="LightSeaGreen",
                width=2,
            )
        )
        illusfig.add_shape(type="line",
            x0=-1.5, y0=-(np.exp(-np.sqrt(mu2)/scalefac)/2+radius+radius+np.exp(-rd/scalefac)), x1=1.5, y1=-(np.exp(-np.sqrt(mu2)/scalefac)/2+radius+radius+np.exp(-rd/scalefac)),
            line=dict(
                color="LightSeaGreen",
                width=2,
            )
        )

    illusfig.update_layout(width=142.5, height=220, margin=dict(l=30, r=30, t=0, b=0),xaxis=dict(visible=False), yaxis=dict(visible=False),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',dragmode=False)

    # plot the Bode fig
    bodespectrum_plot = go.Scatter(
        x=pdwrange,
        y=10.0*np.log10(ptransf),
        mode='lines',
        name="Spectrum <br>with<br>given S & M"
    )
    o_bodespectrum_plot = go.Scatter(
        x=pdwrange,
        y=10.0*np.log10(optimized_ptransf),
        mode='lines',
        name="<br>Optimized <br>spectrum <br>for given S"
    )
    # Set the layout of the graph
    blayout = go.Layout(
        title="Spectrum Bode Plot",
        xaxis=dict(title="Δω/Δω<sub>3dB"), # domain=[0, 0.475],
        yaxis=dict(title="10lg(|s<sub>d</sub>/s<sub>i</sub>|<sup>2)"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        legend=dict(
            traceorder='normal',itemwidth=30,xanchor='right',yanchor='top',bgcolor='rgba(0,0,0,0)' # Adjust the item width as per your preference
        )
    )
    # Create the figure
    bodefig = go.Figure(data=[bodespectrum_plot,o_bodespectrum_plot], layout=blayout)
    bodefig.update_xaxes(type="log", range=[-1,5]) # log range: 10^0=1, 10^5=100000
    bodefig.update_layout(title_text='Spectrum Bode Plot', title_x=0.5)


    return 'Above (S, M) is {}'.format(check_s_m), '{}'.format(m_range), 'Above (S, M) gives Insertion Loss = {:.2f} dB.'.format(insertionloss), 'Insertion Loss = {:.2f} dB.'.format(bestMinsertionloss), 'Insertion Loss = {:.2f} dB.'.format(bestSinsertionloss), fig, bodefig, illusfig, 'M = ±{:.3f},'.format(optimized_m), np.round(optimized_re,3), np.round(optimized_rd,3),np.round(optimized_rd,3),np.round(optimized_re,3), np.round(np.sqrt(optimized_mu2),3), 'S = {s}, M = {m}'.format(s=best_s,m=best_m), '{ap}.'.format(ap=approximate), np.round(best_s_re,3), np.round(best_s_rd,3),np.round(best_s_rd,3),np.round(best_s_re,3), np.round(np.sqrt(best_s_mu2),3), s, m, s_slider_value, m_slider_value,np.round(re,3),np.round(rd,3),np.round(np.sqrt(mu2),3)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
