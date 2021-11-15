import datetime, numpy, pandas, scipy, seaborn,random
from functools import reduce
from snps import SNPs
import streamlit as st
import altair as alt
import os
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize
import emcee
import plotly.figure_factory as ff
import time

try:
    from app_secrets import MINIO_ACCESS_KEY, MINIO_ENCRYPT_KEY
except:
    access_key=os.getenv("MINIO_ACCESS_KEY")
    secret_key=os.getenv("MINIO_SECRET_KEY")

@st.cache(allow_output_mutation=True)
def read_data():
    dZ=pandas.read_csv('data/nl_N-CK-ZE_dz.csv')
    dZ['Delta-Z'] = dZ['Delta-Z'].round(decimals=2)
    dZ['local_x'] = dZ['X'].min() - 10000.0
    dZ['local_y'] = dZ['Y'].min() - 10000.0
    dZ['Distance'] = ((dZ['X']-dZ['local_x']).apply(numpy.square)+(dZ['Y']-dZ['local_y']).apply(numpy.square)).apply(numpy.sqrt)
    dZ['Davg'] = round(dZ['Distance']/1000)*1000
    dZ['Delta-Z_rel'] = 100*dZ['Delta-Z']/dZ['dZ']
    wellnames = dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list()
    total_wells = len(wellnames)
    return (dZ,total_wells)

def main():
    st.set_page_config(page_title="NGUYEN APPS", page_icon='logo.jpg', layout='centered', initial_sidebar_state='auto')
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 90%;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Navigation")
    st.sidebar.header("App Selection")
    goto = st.sidebar.radio('Go to:',['NGUYEN=MC2', 'I DRAW NGUYEN'])
    if goto == 'I DRAW NGUYEN':
        with st.container():
            st.title("I DRAW NGUYEN: Interactive Depth Residual Analysis With NGUYEN")
            st.header("Visualization App for QC & editing depth misties(Delta-Z) between horizons and well tops")
            st.subheader('Parameter Selections:')
            with st.spinner(text='Data loading in progress... Please wait!'):
                time.sleep(1)
            idrawu()
    else:
        with st.container():
            st.title('NGUYEN=MC2: Non-Gaussian UncertaintY EstimatioN by MCMC*')
            st.header("V0,K and Uncertainty Estimation using MCMC - Markov chain Monte Carlo method")
            #CSS to display content correctly
            st.subheader('Parameter Selections:')
            v0kmcmc()

    st.sidebar.info(
            "**Created by:** [KHANH NGUYEN ](mailto:khanhduc@gmail.com)"
            "**©️**2️⃣0️⃣2️⃣0️⃣"
            )

    return None

@st.cache(allow_output_mutation=True)
def generate_vel_data(k, v0, sigma,z,n):
    zi = numpy.linspace(z[0],z[1], n, endpoint=False)
    vi = [k*z + v0 + random.gauss(0, sigma) for z in zi]
    z_scale = numpy.linspace(z[0]-200, z[1]+100, n*4) # for plotting
    return zi, vi, z_scale

def model(zi, k, v0):
    return v0 + k * zi

def l2_loss(tup, zi, vi):
    k, v0 = tup
    delta = model(zi, k,v0) - vi
    return numpy.dot(delta, delta)

#@st.cache(allow_output_mutation=True)
def neg_log_likelihood(tup, zi, vi):
    # Since sigma > 0, we use use log(sigma) as the parameter instead.
    # That way we have an unconstrained problem.
    k, v0, log_sigma = tup
    sigma = numpy.exp(log_sigma)
    delta = model(zi, k, v0) - vi
    return len(zi)/2*numpy.log(2*numpy.pi*sigma**2) + numpy.dot(delta, delta) / (2*sigma**2)

#@st.cache(allow_output_mutation=True)
def log_likelihood(tup, zi, vi):
    return -neg_log_likelihood(tup, zi, vi)

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def MCMC(zi,vi,ndim,nwalkers,niter):
    p0 = [numpy.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood,
                                args=[zi, vi])
    sampler.run_mcmc(p0, niter)
    return sampler

def v0kmcmc():
    col1, col2, col3, col4 = st.columns([2,2,2,4])
    input = col1.selectbox('Select Input Data:',['Synthetic Data', 'Real Data'])
    col2.text('Expand to change parameters for:')
    lqr = col3.checkbox('Least-Squares Regression?', True)
    mlr = col3.checkbox('Max.-Likelihood Regression?', True)
    col4.text('Expand to change parameters and run:')
    with col4.expander('NGUYEN=MC2'):
#            with col3:
        niter = st.number_input('Enter niter',1000,5000,5000,1000)
#            ndim = st.number_input('Enter ndim',1,3,3,1)
        nwalkers = st.number_input('Enter nwalkers',6,10,8,1)
        show_hist = st.checkbox('Show histogram?', True)
        show_rug = st.checkbox('Show rug?', True)
#            with col5:
        show_normal_distribution = st.checkbox('Show normal distribution?', True)
        bin_size = st.number_input('Enter bin size',1,20,3,1)
        view = st.button('Hit Me to Run MCMC Simulation')
        if view:
            with st.spinner(text='Simulation in progress... See snapshots below:'):
                        time.sleep(1)
                        st.markdown(f"""<iframe width="100%" height="276" frameborder="0"
                            src="https://observablehq.com/embed/@cadasa/linear-regression?cell=chart" title="Snapshot of the Simulation"></iframe>
                            """, unsafe_allow_html=True)
    if input == 'Synthetic Data':
        with col2.expander('Synthetic Data'):
#            norm = st.checkbox('Gaussian uncertainty?',False)
            k = st.slider('Slide to select k:',-0.2, 1.0, 0.2,0.01)
            v0 = st.slider('Slide to select V0(m/s):', 1500.0, 5000.0, 2000.0)  #@param {type: "number"}
            sigma = st.slider('Slide to select Sigma', 0.0, 500.0, 300.0)
            z = st.slider('Select depth range (m)', 0.0, 5000.0, (1000.0, 3000.0))
            n = st.number_input('Enter number of data points', 100, 1000, 300,50)
        zi, vi, z_scale = generate_vel_data(k,v0,sigma,z,n)
        Layer_name = 'SYNTHETIC'
    if input == 'Real Data':
        with col2.expander('Real Data'):
            file = st.file_uploader('Choose File')
            if not file:
                st.warning('Please upload a file.')
                st.stop()
            st.success('File uploaded successfully!')

#    zi, vi, z_scale = generate_vel_data(k,v0,sigma,z,n)
    k_hat, v0_hat = scipy.optimize.minimize(l2_loss, (0, 0), args=(zi, vi)).x
    k_hat2, v0_hat2, log_sigma_hat2 = scipy.optimize.minimize(neg_log_likelihood, (0, 0, 0), args=(zi, vi)).x
    fig1 = go.Figure()

    if view :
        ndim=3
        sampler = MCMC(zi,vi,ndim,nwalkers,niter)
# Plot 95% confidence interval
        samples = sampler.chain[:, -4000:, :].reshape((-1, ndim))
#samples = samples[samples[...,1] >0]
#samples = samples[samples[...,1] <10000]
        curves = []
        for k, v0, log_sigma in samples:
            curves.append(
                model(z_scale, k, v0)
                    )
        lo, hi = numpy.percentile(curves, (2.5, 97.5), axis=0)
        fig1.add_trace(go.Scatter(
                        x=z_scale, marker_color = 'red',
                        y=lo,mode='lines',
                        opacity=0.1,line = dict(width=0.05),showlegend=False))
        fig1.add_trace(go.Scatter(
                        x=z_scale, marker_color = 'red',
                        y=hi,mode='lines',fill='tonexty',
                        opacity=0.1,line = dict(width=0.05),showlegend=True,name='95% Confidence Interval'))

# Grab the last 50 from each walker
        samples2 = sampler.chain[:, -100:, :].reshape((-1, ndim))
#samples = samples[samples[...,1] >0]
#samples = samples[samples[...,1] <10000]
        k, v0, log_sigma = samples2[100]
        fig1.add_trace(go.Scatter(x=z_scale, y=model(z_scale, k, v0), mode='lines',opacity=0.2, line = dict(width=1), marker_color = 'green',showlegend=True,name='MCMC sample lines'))
        for k, v0, log_sigma in samples2:
            fig1.add_trace(go.Scatter(x=z_scale, y=model(z_scale, k, v0), mode='lines',opacity=0.2, line = dict(width=.1), marker_color = 'green',showlegend=False,name='MCMC sample lines'))
#        fig1.write_html("fig1.html")
# Plot best fit V0,k
    if lqr:
        fig1.add_trace(go.Scatter(x=z_scale, y=model(z_scale, k_hat, v0_hat), mode='lines',opacity=1, line = dict(width=2), marker_color = 'orange',showlegend=True,name='LQR <b>V<sub>0</sub>='+str(round(v0_hat,4))+', K='+str(round(k_hat,4))))
    if mlr:
        fig1.add_trace(go.Scatter(x=z_scale, y=model(z_scale, k_hat2, v0_hat2), mode='lines',opacity=1, line = dict(width=2), marker_color = 'blue',showlegend=True,name='MLR <b>V<sub>0</sub>='+str(round(v0_hat2,4))+', K='+str(round(k_hat2,4))))
# Plot scatter Vel vs. Zmid
    fig1.add_trace(go.Scatter(x=zi, y=vi, opacity=1.0,mode='markers',showlegend=True,name='V<sub>int</sub>-Z<sub>mid</sub> Xplots for layer: ' +str(Layer_name), marker_color = 'black'))

    fig1.update_layout(
                height=650, width=1000,
                title={'text':"<b>LINEAR REGRESSIONS & NGUYEN=MC<sup>2</sup> FOR LAYER: " +str(Layer_name)+"</b>",
#            yaxis_type="log",
                'xanchor': 'center',
                'yanchor': 'top',
                'x': 0.47,
                'y': .95,
                'font_size':20,},
                yaxis=dict(showline=True,showgrid=True,gridcolor='white',showticklabels=True,linecolor='rgb(204, 204, 204)',
                title='V<sub>int</sub> (m/s)',
                titlefont_size=12,
                zeroline=False,
                showspikes = True,
#            range = (4500,10500),
                tickfont_size=12),
                xaxis=dict(showline=True,showgrid=True,gridcolor='white',showticklabels=True,linecolor='rgb(204, 204, 204)',
                title='Z<sub>mid</sub> (m)',
                titlefont_size=12,
                zeroline=False,
                showspikes = True,
                tickfont_size=12)
                )

    with st.expander('View Chart: ', True):
        st.plotly_chart(fig1,use_container_width=True)
#        if st.button('CREATE HTML REPORT?'):
#        fig1.write_html("fig1.html")

    if view:
# Grab slightly more samples this time
        samples3 = sampler.chain[:, -1000:, :].reshape((-1, ndim))
#samples = samples[samples[...,1] >0]
#samples = samples[samples[...,1] <10000]
        k_samples, v0_samples, log_sigma_samples = samples3.T
        hist_data = [k_samples*1000,v0_samples,numpy.exp(log_sigma_samples)]
        group_labels = ['K*1000', 'V<sub>0</sub>', '\u03C3']
        colors = ['red', 'blue', 'green']
        if show_normal_distribution ==True:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=show_hist, colors=colors, bin_size=bin_size,show_rug=show_rug,curve_type='normal',)
        else:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=show_hist, colors=colors, bin_size=bin_size,show_rug=show_rug,)
        fig.update_layout(
                    height=650, width=1320,
                    title={'text':"<b>Histograms of \u03C3, V<sub>0</sub> and K for layer: " +str(Layer_name)+"</b>",
#            yaxis_type="log",
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'x': 0.5,
                    'y': .95,
                    'font_size':20,},
                    yaxis=dict(showline=True,showgrid=True,gridcolor='white',showticklabels=True,linecolor='rgb(204, 204, 204)',
#            title='Velocity (m/s)',
                    titlefont_size=12,
                    zeroline=False,
                    showspikes = True,
#            range = (4500,10500),
                    tickfont_size=12),
                    xaxis=dict(showline=True,showgrid=True,gridcolor='white',showticklabels=True,linecolor='rgb(204, 204, 204)',
                    title='Values of \u03C3, V<sub>0</sub> and K*1000',
                    titlefont_size=12,
                    zeroline=False,
                    showspikes = True,
                    tickfont_size=12)
                    )

        st.plotly_chart(fig,use_container_width=True)

    return None

def idrawu():
#    st.sidebar.title("Data Selection")
    # upload the user depth residual file
#    st.sidebar.header("Input Data")
#    user_file = st.sidebar.file_uploader("Upload your csv file:")
    # Collapsable user AISNPs DataFrame
#    if user_file is not None:
#        try:
#            with st.spinner("Uploading your csv..."):
#                userdf = SNPs(user_file.getvalue()).snps
#        except Exception as e:
#            st.error(
#                f"Sorry, there was a problem processing your csv file.\n {e}"
#            )
#            user_file = None
#    else:
    dZ,total_wells = read_data()
    wellnames = dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list()
    current_wells = len(wellnames)
    col1, _, col3, _, _ = st.columns(5)
    outlier = col3.checkbox("Remove Outliers?", True)
    multihor = col1.checkbox("Multiple Horizon?", True)

#   saveguard for empty selection
    col1, _, col3, col4, _ = st.columns(5)
    if multihor:
        horizons = dZ.drop_duplicates(subset = ['hor_names'])['hor_names'].to_list()
        horizon = col1.selectbox("Choose Horizon", horizons)
        dZ = dZ[dZ["hor_names"]==horizon]
        wellnames = dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list()
        total_wells = len(wellnames)
        current_wells = len(wellnames)

    st.sidebar.header("Well selection")
    with st.sidebar.expander('👉 Expand to see well list:', False):
        multiselection = st.multiselect("Select/Deselect well(s)", wellnames, default=wellnames)
    dZ = dZ[dZ["well_name"].isin(multiselection)]
    current_wells = len(dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list())

    st.sidebar.success("""\
        """
        f"""⚠️ {current_wells}/{total_wells} """
        """wells are selected"""
        )
    # saveguard for empty selection
    if len(multiselection) == 0:
        return

    if outlier:
        st.sidebar.header("Wells excl. by outlier editing")
        sb_excl_cont = st.sidebar.container()
        rem_type = col3.radio("Removal Type:", ["Absolute value", "Relative value"])
        if rem_type == 'Absolute value':
            abs_outl = col4.slider("Threshold in meters", 0.0, 200.0, 200.0, 1.0)
            exclude_wells = dZ[dZ['Delta-Z'].abs()>abs_outl][['well_name','Delta-Z']].drop_duplicates(subset = ['well_name']).to_records(index=False).tolist()
            dZ = dZ[dZ["Delta-Z"].abs()<=abs_outl]
            wellnames = dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list()
            current_wells = len(wellnames)
            tot_exclude_wells = len(exclude_wells)
        elif rem_type =='Relative value':
            rel_outl = col4.slider("Threshold in %", 0.0, 20.0, 20.0, 0.5)
            exclude_wells = dZ[dZ['Delta-Z_rel'].abs()>rel_outl][['well_name','Delta-Z']].drop_duplicates(subset = ['well_name']).to_records(index=False).tolist()
            dZ = dZ[dZ['Delta-Z_rel'].abs()<=rel_outl]
            wellnames = dZ.drop_duplicates(subset = ['well_name'])['well_name'].to_list()
            current_wells = len(wellnames)
            tot_exclude_wells = len(exclude_wells)

        sb_excl_cont.markdown(f"""\
            <div style="font-size: medium">
            {tot_exclude_wells}
            wells excluded ('Wellname', Delta-Z): \n
            {"".join(str(exclude_wells))}
            </div><br/>
            """, unsafe_allow_html=True)


#Estimate CI using bootstrap
    observations_by_davg = {}
    for davg, y in zip(dZ['Davg'], dZ['Delta-Z']):
        observations_by_davg.setdefault(davg, []).append(y)
    lo_bound = []
    hi_bound = []
    davgs = sorted(observations_by_davg.keys())
    for davg in davgs:
        series = observations_by_davg[davg]
        bootstrapped_means = []
        for i in range(1000):
        # sample with replacement
            bootstrap = [random.choice(series) for _ in series]
            bootstrapped_means.append(numpy.mean(bootstrap))
        lo_bound.append(numpy.percentile(bootstrapped_means, 2.5))
        hi_bound.append(numpy.percentile(bootstrapped_means, 97.5))
    df = pandas.DataFrame({'x':davgs,'lower_bound':lo_bound,'upper_bound':hi_bound})

# interval selection in the histogram plot
    pts = alt.selection(type="interval",encodings=["x"])

# multi selection in the 1st scatter plot
    selector = alt.selection_multi(empty='all', fields=['well_name'])

    Delta_Z_absmax = abs(dZ['Delta-Z']).max()
    Distance_min = dZ['Distance'].min() - 5000
    Distance_max = dZ['Distance'].max() + 5000
    X_min = dZ['X'].min() - 5000
    X_max = dZ['X'].max() + 5000
    Y_min = dZ['Y'].min() - 5000
    Y_max = dZ['Y'].max() + 5000
#box plots
    c = alt.Chart().mark_boxplot(color='green',size=20).encode(
        x=alt.X('Delta-Z:Q',scale=alt.Scale(domain=(-Delta_Z_absmax, Delta_Z_absmax))),
    ).interactive().transform_filter(
        pts
    ).properties(
        width=350, height=60
    )

#histogram
    c2 = alt.Chart().transform_density(
        'mbin',
        as_=['mbin', 'density'],
    ).mark_bar().encode(
        x=alt.X('mbin:Q',axis=alt.Axis(title='Delta-Z'),scale=alt.Scale(domain=(-Delta_Z_absmax, Delta_Z_absmax))),
        y="density:Q",
        color=alt.condition(pts, alt.value("green"), alt.value("lightgray"))
    ).properties(
        width=350,
        height=337
    ).add_selection(pts)
#kde on histogram
    alt.data_transformers.disable_max_rows()
    c2b = alt.Chart().transform_density(
        'mbin',
        as_=['mbin', 'density'],
    ).mark_area(color='red', opacity=0.2).encode(
        x=alt.X('mbin:Q',axis=alt.Axis(title='Delta-Z'),scale=alt.Scale(domain=(-Delta_Z_absmax, Delta_Z_absmax))),
        y="density:Q"
    )
#mean line on histogram
    c2c = alt.Chart().mark_rule(color='blue').encode(
        x=alt.X('mean(Delta-Z):Q',axis=alt.Axis(title='Delta-Z'),scale=alt.Scale(domain=(-Delta_Z_absmax, Delta_Z_absmax))),
        tooltip=['mean(Delta-Z)','stdev(Delta-Z)'],size=alt.value(3)
    ).transform_filter(pts)

#1st scatter plot for dZ and CI
    c3 = alt.Chart().mark_point(filled=True).encode(
        x=alt.X('Distance:Q',axis=alt.Axis(title='Distance to Local Origin(m)'),scale=alt.Scale(domain=(Distance_min, Distance_max))),
        y=alt.Y('Delta-Z:Q',axis=alt.Axis(title='Delta-Z(m)'),scale=alt.Scale(domain=(-Delta_Z_absmax, Delta_Z_absmax))),
        tooltip=['well_name', 'Delta-Z'],
        color=alt.condition(selector,'Delta-Z:Q', alt.value('lightgray'), scale=alt.Scale(scheme='turbo'))
    ).interactive().transform_filter(
        pts
    ).properties(
        width=800,
        height=300).add_selection(selector)
#CI
    c3b = alt.Chart(df).mark_area(opacity=0.15,color='red').encode(
        x='x:Q',
        y='lower_bound:Q',
        y2='upper_bound:Q',
        tooltip=['upper_bound','lower_bound']
    ).interactive().properties(
        width=800,
        height=300)

#scatter plot for map
    c4 = alt.Chart().mark_point(filled=True).encode(
        x=alt.X('X:Q',axis=alt.Axis(title='X(m)'),scale=alt.Scale(domain=(X_min, X_max))),
        y=alt.Y('Y:Q',axis=alt.Axis(title='Y(m)'),scale=alt.Scale(domain=(Y_min, Y_max))),
        tooltip=['well_name', 'Delta-Z'],
        color=alt.Color('Delta-Z:Q',scale=alt.Scale(scheme='turbo'))
    ).interactive().transform_filter(
        pts
        ).properties(
        width=360,
        height=450
    ).transform_filter(
        selector
    )
    c=alt.vconcat(c3b+c3,
    alt.hconcat(c2+c2b+c2c&c, c4),data=dZ
#    ).resolve_scale(
#        color='independent'
    ).transform_bin(
        "mbin",
        field="Delta-Z",
        bin=alt.Bin(maxbins=22)
    )
#combine charts
    st.altair_chart(c, use_container_width=True)
    c.save('chart.html')
    st.markdown(f"""\
        <div style="font-size: medium">
        👉 Hover the cursor over the objects in the charts to see infomation.\n
        <div style="font-size: medium">
        👉 MB1 click on the marker in the top chart to see its location in
        the bottom right map ('shift+MB1' to select multiple markers).\n
        <div style="font-size: medium">
        👉 Click and drag your cursor on the histogram to select different
        ranges of Delta-Z. Click again outside the selection area to reset.
        </div><br/>

    """, unsafe_allow_html=True)

    col1, _ = st.columns([9, 2])
    col1.info("""
        Data are from 'VELMOD' open project in the Netherlands - see my [Map](https://cadasa.github.io/lithostrat_nl/)
        """
        )

    return None
    # ----------------------








if __name__ == "__main__":
    main()
