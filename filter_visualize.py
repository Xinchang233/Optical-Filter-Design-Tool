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
        html.H1("Maximum drop-port transmission contour", style={'textAlign': 'center'}),
        html.H2(children="Paper: New passband shapes that minimize the insertion loss of coupled-resonator bandpass filters", style={'textAlign': 'center'}),
        html.Label(children=["Enter the normalized bandwidth Δω/2r",html.Sub(children="o")," then press PLOT: "]),
        dcc.Input(id="width", type="number", value=3), # should be half-width, just change from w/r0 to w/2r0
        html.Button(id='plot-button-state', n_clicks=0, children='PLOT'),
        dcc.Graph(id="contour-graph"),
        html.Br(),

        html.H2("Spectrum Optimization"),
        html.Label(children=["For Δω/2r",html.Sub(children="o")," being specified above, enter the values for parameters. Specify both S and M, OR any two in r",html.Sub(children="e"),"/r",html.Sub(children="o"),", r",html.Sub(children="d"),"/r",html.Sub(children="o")," and ",html.Span("µ"),"/r",html.Sub(children="o"),":"]),
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
            ], style={'padding': 5, 'flex': 5}),

            html.Div(children=[
                #dcc.Graph(id="illustration"),
                html.Label(children=["r",html.Sub(children="e"),"/r",html.Sub(children="o"),": "]),
                dcc.Input(id="revalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label(children=["r",html.Sub(children="d"),"/r",html.Sub(children="o"),": "]),
                dcc.Input(id="rdvalue", type="number"),
                html.Br(),html.Br(),html.Br(),
                html.Label(children=[html.Span("µ"),"/r",html.Sub(children="o"),": "]),
                dcc.Input(id="mu2value", type="number"),
                html.Br(),html.Br(),
                html.Label("Graphical illustration: "),
                dcc.Graph(id="illustration"),
            ], style={'padding': 15, 'flex': 2}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

        html.H3("Optimization Analysis"),
        html.Div([
            html.Div(children=[
                html.P("For the S value above, best M which gives lowes insertion loss (red curve) is:"),
                html.Div(id="optimized_m_value", style={'textAlign': 'center'}),
                html.P(children=["and its corresponding r",html.Sub(children="e"),", r",html.Sub(children="d"), ", ", html.Span("µ"),html.Sup(children="2")," are"]),
                html.Div(id="optimized_m_rerd", style={'textAlign': 'center'}),
                html.P("and it gives"),
                html.Div(id="IL_best_M", style={'textAlign': 'center'}),
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.P("For the specified Δω/2ro, lowest IL comes from"),
                html.Div(id="optimized_s_value", style={'textAlign': 'center'}),
                html.Div(id="approximateORnot"),
                html.P(children=["Its corresponding r",html.Sub(children="e"),", r",html.Sub(children="d"), ", ", html.Span("µ"),html.Sup(children="2")," are"]),
                html.Div(id="optimized_s_rerd", style={'textAlign': 'center'}),
                html.P("and it gives"),
                html.Div(id="IL_best_S", style={'textAlign': 'center'}),
            ], style={'padding': 10, 'flex': 1}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),

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
    
    return fig

# Define the callback function to update the spectrum
@app.callback(
    Output("check_s_m_value", "children"),
    Output("rangeFORm", "children"),
    Output("IL_current", "children"),
    Output("IL_best_M","children"),
    Output("IL_best_S","children"),
    Output("spectrum", "figure"),
    Output("illustration", "figure"),
    Output("optimized_m_value", "children"),
    Output("optimized_m_rerd", "children"), # gives the sentence containing re, rd, mu^2
    Output("optimized_s_value", "children"),
    Output("approximateORnot", "children"), # if S = -1, take S = -0.9
    Output("optimized_s_rerd", "children"), # gives the sentence containing re, rd, mu^2
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

    w = w*2.0 # check this. seems no influence because no return values related to w
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
    else:
        check_s_m = '''INVALID!'''
        transf = np.zeros(500)
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
        approximate = 'but S=-1 will lead to infinite coupling. So take S = -0.9, whose optimized M = ±'+str(np.round(app_optimized_m,3))
        best_s_re = (1+app_optimized_m) * w / (2*math.sqrt(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2))) - 1
        best_s_rd = (1-app_optimized_m) * w / (2*math.sqrt(-0.9-1+math.sqrt(2*(-0.9)*(-0.9)+2))) - 1
        best_s_mu2 = w/2 * math.sqrt(w*w/2 + (2+best_s_re+best_s_rd)**2) - w*w/4 - (1+best_s_re)*(1+best_s_rd)
        bestSinsertionloss = 10.0*math.log10(sdsi(0,best_s_re,best_s_rd,best_s_mu2))

    # Plot the spectrum
    spectrum_plot = go.Scatter(
        x=dwrange,
        y=transf,
        mode='lines',
        name="Spectrum <br>with<br>given S & M"
    )
    o_spectrum_plot = go.Scatter(
        x=dwrange,
        y=optimized_transf,
        mode='lines',
        name="<br>Optimized <br>spectrum <br>for given S"
    )
    # Set the layout of the graph
    layout = go.Layout(
        title="Spectrum",
        xaxis=dict(title="Δω/Δω<sub>3dB"), # domain=[0, 0.475],
        yaxis=dict(title="|s<sub>d</sub>/s<sub>i</sub>|<sup>2"),
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
    illusfig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-radius, y0=np.exp(-mu2/scalefac)/2, x1=+radius, y1=np.exp(-mu2/scalefac)/2+radius+radius,
        line_color="LightSeaGreen",
    )
    illusfig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-radius, y0=-np.exp(-mu2/scalefac)/2, x1=+radius, y1=-np.exp(-mu2/scalefac)/2-radius-radius,
        line_color="LightSeaGreen",
    )
    illusfig.add_shape(type="line",
        x0=-1.5, y0=np.exp(-mu2/scalefac)/2+radius+radius+np.exp(-re/scalefac), x1=1.5, y1=np.exp(-mu2/scalefac)/2+radius+radius+np.exp(-re/scalefac),
        line=dict(
            color="LightSeaGreen",
            width=2,
        )
    )
    illusfig.add_shape(type="line",
        x0=-1.5, y0=-(np.exp(-mu2/scalefac)/2+radius+radius+np.exp(-rd/scalefac)), x1=1.5, y1=-(np.exp(-mu2/scalefac)/2+radius+radius+np.exp(-rd/scalefac)),
        line=dict(
            color="LightSeaGreen",
            width=2,
        )
    )

    illusfig.update_layout(width=142.5, height=220, margin=dict(l=30, r=30, t=0, b=0),xaxis=dict(visible=False), yaxis=dict(visible=False),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',dragmode=False)


    return 'Above (S, M) is {}'.format(check_s_m), '{}'.format(m_range), 'Above (S, M) gives Insertion Loss = {:.2f} dB.'.format(insertionloss), 'Insertion Loss = {:.2f} dB.'.format(bestMinsertionloss), 'Insertion Loss = {:.2f} dB.'.format(bestSinsertionloss), fig, illusfig, 'M = ±{:.3f},'.format(optimized_m), 're = {re:.3f}, rd = {rd:.3f} OR re = {rd:.3f}, rd = {re:.3f};  µ^2 = {mu2:.3f}'.format(re=optimized_re,rd=optimized_rd,mu2=optimized_mu2), 'S = {s}, M = {m}'.format(s=best_s,m=best_m), '{ap}.'.format(ap=approximate), 're = {re:.3f}, rd = {rd:.3f} OR re = {rd:.3f}, rd = {re:.3f};  µ^2 = {mu2:.3f}'.format(re=best_s_re,rd=best_s_rd,mu2=best_s_mu2), s, m, s_slider_value, m_slider_value,re,rd,mu2

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
