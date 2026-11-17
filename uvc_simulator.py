"""UVC Vineyard Simulator v2 — educational/engineering design estimator.

Run with: python -m streamlit run uvc_simulator.py
The optical model is deliberately idealized and MUST be radiometrically validated.
"""
from __future__ import annotations
import io
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import gammaln


@dataclass
class Setup:
    mode: str = 'UVC LED'
    pattern: str = 'Regular grid'
    columns: int = 10
    rows: int = 30
    pitch_x: float = 0.015
    pitch_y: float = 0.0394
    stagger_fraction: float = 0.5
    center_y: float = 0.65
    arrays_per_side: int = 3
    sides: int = 2
    array_gap: float = 0.15
    z: float = 0.15
    speed_mph: float = 2.
    y_min: float = 0.05
    y_max: float = 1.25
    target: float = 15.
    wavelength: float = 255.
    beam_angle: float = 90.
    p_nominal_mw: float = 11.5
    p_min_mw: float = 8.
    p_max_mw: float = 15.
    vf: float = 6.
    current: float = 0.1
    unit_cost: float = 2.5
    # Mercury lamp model: vertical extended tube with uniform optical line output.
    lamp_length: float = 1.10
    lamp_elec_w: float = 55.
    lamp_uv_w: float = 18.
    coating_transmission: float = 0.90
    forward_fraction: float = 0.50
    lamp_segments: int = 41
    # Budget assumptions
    led_driver_eff: float = 0.90
    acdc_eff: float = 0.90
    ballast_eff: float = 0.90
    other_load_w: float = 50.
    generator_margin: float = 1.5
    driver_channels: int = 4
    led_series: int = 8
    driver_module_cost: float = 40.
    ballast_lamps_per_unit: int = 1
    ballast_unit_cost: float = 28.
    generator_cost: float = 750.
    supply_cost: float = 200.
    electrical_controls: float = 280.
    per_array_fixture: float = 100.
    cooling_per_array: float = 75.
    other_misc: float = 150.
    contingency_pct: float = 10.
    budget_limit: float = 5000.

    @property
    def v(self): return self.speed_mph * 0.44704
    @property
    def n_per_array(self): return self.columns * self.rows if self.mode == 'UVC LED' else self.columns
    @property
    def total_arrays(self): return self.arrays_per_side * self.sides
    @property
    def n_total(self): return self.n_per_array * self.total_arrays
    @property
    def m(self):
        return -math.log(2) / math.log(math.cos(math.radians(self.beam_angle / 2))) if self.mode == 'UVC LED' else 0.0
    @property
    def single_width(self): return (self.columns-1) * self.pitch_x


def emitter_coords(c: Setup):
    """XY positions: X along travel, Y above ground. Columns are vertical stacks."""
    x = (np.arange(c.columns) - (c.columns-1)/2)*c.pitch_x
    if c.mode == 'UVC LED':
        y = c.center_y + (np.arange(c.rows) - (c.rows-1)/2)*c.pitch_y
        xx, yy = np.meshgrid(x,y)
        if c.pattern == 'Staggered vertical columns':
            yy[:,1::2] += c.stagger_fraction*c.pitch_y
        elif c.pattern == 'Staggered horizontal rows':
            xx[1::2,:] += c.stagger_fraction*c.pitch_x
        return xx.ravel(), yy.ravel()
    return x, np.full(c.columns, c.center_y)


def kernel_integrated(c: Setup, targets_y: np.ndarray, y_sources: np.ndarray, power_w: np.ndarray, m: float):
    """Closed-form complete-pass dose for normalized forward-hemisphere Lambertian emitters.

    Sources may have unequal output. y_sources is source height and power_w is its
    effective optical power emitted INTO the modeled forward hemisphere.
    """
    z=c.z
    dy=targets_y[:,None]-y_sources[None,:]
    a2=z*z+dy*dy
    beta = math.exp(0.5*math.log(math.pi)+gammaln((m+2)/2)-gammaln((m+3)/2))
    coef=(m+1)/(2*math.pi)*beta
    response=coef*z**(m+1)/np.power(a2,(m+2)/2)
    # W/m² integrated over x divided by m/s => J/m²; 0.1 => mJ/cm².
    return 0.1*c.arrays_per_side/c.v*(response @ power_w)


def dose(c: Setup, heights: np.ndarray, optical_override_mw: float|None=None):
    x,y=emitter_coords(c)
    if c.mode=='UVC LED':
        p=(c.p_nominal_mw if optical_override_mw is None else optical_override_mw)/1000
        return kernel_integrated(c,heights,y,np.full(y.size,p),c.m)
    # Discretize each vertical lamp as uniform distributed LINE source.
    # Optical UVC lamp output is measured as total-emitted power. Apply coating
    # transmission and adjustable forward-directed fraction BEFORE kernel.
    seg_count=c.lamp_segments
    lamp_positions=np.repeat(x,seg_count)
    segment_y=c.center_y + np.tile(np.linspace(-c.lamp_length/2+c.lamp_length/(2*seg_count),
                                                 c.lamp_length/2-c.lamp_length/(2*seg_count),seg_count),len(x))
    raw=(c.lamp_uv_w if optical_override_mw is None else optical_override_mw/1000)
    p_eff=raw*c.coating_transmission*c.forward_fraction/seg_count
    return kernel_integrated(c,heights,segment_y,np.full(segment_y.size,p_eff),0.0)


def instantaneous_map(c: Setup, heights: np.ndarray, xx_target: np.ndarray):
    """Irradiance on vertical flat target at a single instant; one canopy side only."""
    x,y=emitter_coords(c)
    if c.mode=='UVC LED':
        xp=x; yp=y; powers=np.full(len(x),c.p_nominal_mw/1000)
        m=c.m
    else:
        k=c.lamp_segments
        xp=np.repeat(x,k)
        yp=c.center_y+np.tile(np.linspace(-c.lamp_length/2+c.lamp_length/(2*k),
                                          c.lamp_length/2-c.lamp_length/(2*k),k),len(x))
        powers=np.full(len(xp),c.lamp_uv_w*c.coating_transmission*c.forward_fraction/k)
        m=0.
    period=c.single_width+c.array_gap
    out=np.zeros((len(heights),len(xx_target)))
    # Vectorize across targets in X, process sources independently for bounded memory.
    for a in range(c.arrays_per_side):
        for src_x,src_y,p in zip(xp,yp,powers):
            d2=(xx_target[None,:]-(src_x+a*period))**2+(heights[:,None]-src_y)**2+c.z*c.z
            out+=p*(m+1)/(2*math.pi)*c.z**(m+1)/np.power(d2,(m+3)/2)*0.1
    return out


def budget_calc(c: Setup):
    if c.mode == 'UVC LED':
        electrical_emitters=c.n_total*c.vf*c.current
        strings=math.ceil(c.n_total/max(1,c.led_series))
        modules=math.ceil(strings/max(1,c.driver_channels))
        driver_cost=modules*c.driver_module_cost
        mains_draw=electrical_emitters/c.led_driver_eff/c.acdc_eff
        components={'LED diodes':c.n_total*c.unit_cost,
                    'Constant-current driver modules':driver_cost,
                    'AC/DC power supply':c.supply_cost}
        current_string_v=c.led_series*c.vf
        detail=f'{strings:,} estimated strings · {modules:,} driver modules · nominal string {current_string_v:.1f} V; check actual driver voltage compliance and string routing'
    else:
        electrical_emitters=c.n_total*c.lamp_elec_w
        ballasts=math.ceil(c.n_total/max(1,c.ballast_lamps_per_unit))
        mains_draw=electrical_emitters/c.ballast_eff
        components={'Mercury lamps (coated assumed)':c.n_total*c.unit_cost,
                    'Compatible electronic ballasts':ballasts*c.ballast_unit_cost}
        detail=f'{ballasts:,} ballasts at {c.ballast_lamps_per_unit} lamp(s) each; actual rated ballast input and starting current must be verified'
    mains_total=mains_draw+c.other_load_w
    generator_running=mains_total*c.generator_margin
    components.update({'Generator estimate':c.generator_cost,
                       'Fixtures / reflectors':c.total_arrays*c.per_array_fixture,
                       'Cooling / thermal hardware':c.total_arrays*c.cooling_per_array,
                       'Safety interlocks / control / cabling':c.electrical_controls,
                       'Miscellaneous':c.other_misc})
    subtotal=sum(components.values())
    contingency=subtotal*c.contingency_pct/100
    components['Contingency']=contingency
    return components, electrical_emitters, mains_total, generator_running, detail


def excel_bytes(c, heights, low, nominal, high, components, e_load, ac_load, gen_rating):
    config=pd.DataFrame({'Parameter':list(vars(c).keys()),'Value':list(vars(c).values())})
    profile=pd.DataFrame({'Canopy height (m)':heights,
                          'Low dose (mJ/cm2)':low,'Nominal dose (mJ/cm2)':nominal,
                          'High dose (mJ/cm2)':high,'Target dose (mJ/cm2)':c.target})
    budget=pd.DataFrame({'Component':list(components),'Estimated cost ($)':list(components.values())})
    power=pd.DataFrame({'Metric':['Emitter electrical input (W)','Estimated generator operating load (W)',
                                   'Suggested generator running rating (W)','Total installed emitters'],
                        'Value':[e_load,ac_load,gen_rating,c.n_total]})
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        config.to_excel(writer,index=False,sheet_name='Inputs')
        profile.to_excel(writer,index=False,sheet_name='Dose profile')
        budget.to_excel(writer,index=False,sheet_name='Budget')
        power.to_excel(writer,index=False,sheet_name='Electrical')
    return buf.getvalue()


st.set_page_config(page_title='UVC Vineyard Simulator',page_icon='💡',layout='wide')
st.title('UVC Vineyard Dosage Simulator')
st.caption('Side-facing panels moving parallel to grapevine rows · LED layouts and mercury lamp arrays · preliminary model')

with st.sidebar:
    st.header('Design inputs')
    mode=st.radio('Light source', ['UVC LED','Mercury lamp'],horizontal=True)
    if mode=='UVC LED':
        st.subheader('LED optical specifications')
        wavelength=st.number_input('Peak wavelength (nm)',200.,400.,255.,step=1.)
        pnom=st.number_input('Nominal optical radiant power per LED (mW)',0.01,10000.,11.5,step=0.5)
        pmin=st.number_input('Minimum optical output (mW)',0.,10000.,8.,step=0.5)
        pmax=st.number_input('Maximum optical output (mW)',0.,10000.,15.,step=0.5)
        beam=st.number_input('Viewing angle, full FWHM (degrees)',1.,179.,90.)
        st.subheader('LED panel pattern')
        pattern=st.selectbox('How are LEDs arranged?', ['Regular grid','Staggered vertical columns','Staggered horizontal rows'])
        st.caption('Each column is a vertical stack. The staggered options shift every other column or row; they keep the same LED count.')
        columns=st.number_input('Number of vertical LED columns',1,120,10)
        rows=st.number_input('LEDs in each column (vertical rows)',1,200,30)
        pitch_x_mm=st.number_input('Spacing between columns (mm)',1.,400.,15.,step=1.)
        pitch_y_mm=st.number_input('Vertical LED spacing (mm)',1.,400.,39.4,step=1.)
        stagger=st.slider('Alternate column/row offset (fraction of pitch)',0.,0.95,0.5,0.05) if pattern!='Regular grid' else 0.
        vf=st.number_input('Forward voltage per LED (V)',0.1,100.,6.,step=0.1)
        current=st.number_input('Current per LED (A)',0.001,20.,0.10,step=0.01,format='%.3f')
        unit_cost=st.number_input('Price per diode ($)',0.,10000.,2.50,step=0.25)
        lamp_length=1.10; lamp_w=55.;lamp_uv=18.;trans=0.9;forward=0.5
    else:
        st.subheader('Mercury lamp specifications')
        st.caption('Use the lamp maker’s *254 nm radiant output* in W; the 55 W rating is electrical, not optical.')
        wavelength=st.number_input('Peak wavelength (nm)',230.,280.,253.7,step=0.1)
        columns=st.number_input('Vertical lamps per array (along travel)',1,80,8)
        rows=1
        lamp_length=st.number_input('Emitting tube length (m)',0.05,3.0,1.10,step=0.05)
        pitch_x_mm=st.number_input('Spacing between lamps along travel (mm)',5.,1000.,100.,step=5.)
        pitch_y_mm=30.;pattern='Regular grid';stagger=0.
        lamp_w=st.number_input('Electrical power per lamp (W)',1.,1000.,55.,step=5.)
        lamp_uv=st.number_input('Uncoated 254 nm optical radiant output / lamp (W)',0.01,500.,18.,step=0.5)
        trans=st.slider('Protective coating UVC transmission',0.0,1.0,0.90,0.01)
        forward=st.slider('Fraction of transmitted UVC sent toward canopy',0.01,1.0,0.50,0.01)
        st.caption('50% forward is a hemispheric idealization; reflector effectiveness and transmission require measurements.')
        unit_cost=st.number_input('Cost of coated lamp ($)',0.,5000.,80.,step=5.)
        beam=120.;pnom=11.5;pmin=8.;pmax=15.;vf=6.;current=0.1
    st.subheader('Placement and operating conditions')
    center_y=st.number_input('Array center above ground (m)',0.0,5.0,0.65,step=0.05)
    arrays=st.number_input('Identical arrays per side',1,30,3)
    sides=st.selectbox('Illuminated sides (hardware totals only)',[1,2],index=1)
    gap=st.number_input('Gap between sequential arrays (m)',0.,10.,0.15,step=0.05)
    z=st.number_input('Panel face to canopy Z (m)',0.01,5.,0.15,step=0.01)
    mph=st.number_input('Tractor speed (mph)',0.1,15.,2.,step=0.1)
    y_min=st.number_input('Canopy minimum height (m)',0.,5.,0.05,step=0.05)
    y_max=st.number_input('Canopy maximum height (m)',0.01,6.,1.25,step=0.05)
    target=st.number_input('Target dose (mJ/cm²)',0.,10000.,15.,step=1.)
    st.subheader('Electrical system and budget')
    with st.expander('Efficiencies and driver / ballast sizing',expanded=False):
        driver_eff=st.number_input('LED driver efficiency',0.1,1.,0.90,step=0.01)
        acdc_eff=st.number_input('AC/DC efficiency',0.1,1.,0.90,step=0.01)
        ballast_eff=st.number_input('Mercury ballast efficiency estimate',0.1,1.,0.90,step=0.01)
        series=st.number_input('LEDs per series string (planning only)',1,100,8)
        channels=st.number_input('Independent strings / driver module',1,40,4)
        module_cost=st.number_input('Price per LED driver module ($)',0.,5000.,40.,step=5.)
        lamps_per_ballast=st.number_input('Lamps / ballast',1,10,1)
        ballast_cost=st.number_input('Price per ballast ($)',0.,5000.,28.,step=2.)
        other_load=st.number_input('Additional operating loads (W)',0.,20000.,50.,step=10.)
        gen_margin=st.number_input('Generator capacity factor',1.,5.,1.5,step=0.1)
    gen_cost=st.number_input('Generator cost ($)',0.,10000.,750.,step=25.)
    supply_cost=st.number_input('AC/DC supply cost ($; LED only)',0.,5000.,200.,step=25.)
    fixture_cost=st.number_input('Fixture / reflector cost per array ($)',0.,5000.,100.,step=10.)
    cooling_cost=st.number_input('Cooling cost per array ($)',0.,5000.,75.,step=10.)
    controls_cost=st.number_input('Wiring, enclosure, interlocks ($)',0.,10000.,280.,step=20.)
    other_misc=st.number_input('Other hardware allowance ($)',0.,10000.,150.,step=10.)
    contingency=st.number_input('Contingency (%)',0.,100.,10.,step=1.)
    budget_limit=st.number_input('Electrical budget limit ($)',0.,1000000.,5000.,step=100.)

if y_max<=y_min:
    st.error('Canopy max height must exceed canopy min height.');st.stop()
if pmin>pmax and mode=='UVC LED':
    st.error('Minimum LED output must be no greater than maximum.');st.stop()

c=Setup(mode=mode,pattern=pattern,columns=int(columns),rows=int(rows),pitch_x=pitch_x_mm/1000,
        pitch_y=pitch_y_mm/1000,stagger_fraction=stagger,center_y=center_y,
        arrays_per_side=int(arrays),sides=sides,array_gap=gap,z=z,speed_mph=mph,
        y_min=y_min,y_max=y_max,target=target,wavelength=wavelength,beam_angle=beam,
        p_nominal_mw=pnom,p_min_mw=pmin,p_max_mw=pmax,vf=vf,current=current,unit_cost=unit_cost,
        lamp_length=lamp_length,lamp_elec_w=lamp_w,lamp_uv_w=lamp_uv,
        coating_transmission=trans,forward_fraction=forward,
        led_driver_eff=driver_eff,acdc_eff=acdc_eff,ballast_eff=ballast_eff,
        other_load_w=other_load,generator_margin=gen_margin,driver_channels=int(channels),
        led_series=int(series),driver_module_cost=module_cost,
        ballast_lamps_per_unit=int(lamps_per_ballast),ballast_unit_cost=ballast_cost,
        generator_cost=gen_cost,supply_cost=supply_cost,electrical_controls=controls_cost,
        per_array_fixture=fixture_cost,cooling_per_array=cooling_cost,other_misc=other_misc,
        contingency_pct=contingency,budget_limit=budget_limit)

heights=np.linspace(c.y_min,c.y_max,121)
nominal=dose(c,heights)
if mode=='UVC LED':
    low=dose(c,heights,c.p_min_mw)
    high=dose(c,heights,c.p_max_mw)
    backcalc_label='Required optical power per LED'
    reference_mw=c.p_nominal_mw
else:
    low=nominal.copy();high=nominal.copy()
    backcalc_label='Required uncoated UVC output per lamp'
    reference_mw=c.lamp_uv_w*1000
min_dose=float(nominal.min());avg_dose=float(nominal.mean())
required_mw=reference_mw*c.target/min_dose if min_dose>0 and c.target>0 else 0.
parts,emit_elec,ac_load,gen_req,architecture=budget_calc(c)
total_cost=sum(parts.values())

k1,k2,k3,k4=st.columns(4)
k1.metric('Emitters per array',f'{c.n_per_array:,}')
k2.metric('Emitters per side',f'{c.n_per_array*c.arrays_per_side:,}')
k3.metric('Total installed emitters',f'{c.n_total:,}')
k4.metric('Minimum predicted dose',f'{min_dose:.3f} mJ/cm²')

tab_dose,tab_layout,tab_heat,tab_budget=st.tabs(['Dose & power','Panel layouts','Overlap heatmap','Electrical & budget'])
with tab_dose:
    fig=go.Figure()
    if mode=='UVC LED':
        fig.add_trace(go.Scatter(x=heights,y=low,name='Low optical power',line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=heights,y=nominal,name='Nominal UVC dose'))
    if mode=='UVC LED':
        fig.add_trace(go.Scatter(x=heights,y=high,name='High optical power',line=dict(dash='dot')))
    fig.add_hline(y=c.target,line_dash='dash',annotation_text='Target dose')
    fig.update_layout(xaxis_title='Target height above ground (m)',yaxis_title='Dose (mJ/cm²)',height=430)
    st.plotly_chart(fig,use_container_width=True)
    a,b,d=st.columns(3)
    a.metric(backcalc_label,f'{required_mw:.1f} mW' if mode=='UVC LED' else f'{required_mw/1000:.2f} W')
    b.metric('Average predicted dose',f'{avg_dose:.3f} mJ/cm²')
    d.metric('Dose / target at worst sampled height',f'{(min_dose/c.target if c.target else 0):.1%}' if c.target else 'Set target')
    st.info('Backcalculation scales each emitter’s *radiant* power at fixed geometry, not electrical input or biological efficacy. It does not guarantee that a commercial device can reach the required optical power.')
    if min_dose<c.target: st.warning('The idealized configuration does not meet the dose goal at every sampled height.')
    else: st.success('The idealized configuration meets the dose goal at all sampled heights.')
with tab_layout:
    st.caption('X is along the tractor’s travel direction; Y is height. **Columns are vertical stacks of LEDs**. A staggered configuration offsets alternate columns or rows; both directions are available to compare.')
    left,right=st.columns(2)
    xp,yp=emitter_coords(c)
    with left:
        st.markdown('**Front view — emitting face of one array**')
        fig=go.Figure()
        if mode=='UVC LED':
            fig.add_trace(go.Scattergl(x=xp,y=yp,mode='markers',marker=dict(size=5,color='#0284c7'),name='LED centers',hovertemplate='X: %{x:.3f} m<br>Y: %{y:.3f} m<extra></extra>'))
            ylow=float(yp.min());yhigh=float(yp.max())
        else:
            for x in xp:
                fig.add_trace(go.Scatter(x=[x,x],y=[c.center_y-c.lamp_length/2,c.center_y+c.lamp_length/2],
                                         mode='lines',line=dict(color='#7c3aed',width=7),showlegend=False,
                                         hovertemplate='Lamp X: '+f'{x:.3f}'+' m<extra></extra>'))
            ylow=c.center_y-c.lamp_length/2;yhigh=c.center_y+c.lamp_length/2
        marginx=max(c.pitch_x*0.4,0.008)
        marginy=max(c.pitch_y*0.3,0.02)
        fig.add_shape(type='rect',x0=float(xp.min())-marginx,x1=float(xp.max())+marginx,
                      y0=ylow-marginy,y1=yhigh+marginy,
                      line=dict(color='#64748b',width=2),fillcolor='rgba(100,116,139,0.07)')
        fig.update_layout(height=500,showlegend=False,xaxis_title='Along-travel X (m)',
                          yaxis_title='Height above ground (m)',margin=dict(t=20,l=15,r=15,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Emitting center span: {c.single_width:.3f} m along travel, '+
                   (f'{max(yp)-min(yp):.3f} m vertically' if mode=='UVC LED' else f'{c.lamp_length:.3f} m lamp length') + '.')
    with right:
        st.markdown('**Top view — sequential arrays facing the canopy**')
        centers=np.arange(c.arrays_per_side)*(c.single_width+c.array_gap)
        xspan=max(c.single_width,0.03)
        xmin=-xspan/2-.12;xmax=centers[-1]+xspan/2+.12
        top=go.Figure()
        for idx,cx in enumerate(centers):
            top.add_shape(type='rect',x0=float(cx-xspan/2),x1=float(cx+xspan/2),y0=-0.008,y1=0.008,
                          line=dict(color='#0369a1'),fillcolor='rgba(2,132,199,0.65)')
            top.add_annotation(x=float(cx),y=-max(.05,.16*c.z),text=f'Array {idx+1}',showarrow=False)
        top.add_shape(type='rect',x0=xmin,x1=xmax,y0=c.z-.012,y1=c.z+.012,
                      line=dict(color='#65a30d'),fillcolor='rgba(101,163,13,0.3)')
        top.add_annotation(x=(xmin+xmax)/2,y=c.z+max(.05,.1*c.z),text='Grapevine target',showarrow=False)
        top.add_annotation(x=(xmin+xmax)/2,y=-max(.10,.35*c.z),text='Tractor travel →',showarrow=False)
        top.update_layout(height=500,xaxis_title='Position along travel X (m)',
                          yaxis_title='Perpendicular distance Z (m)',
                          xaxis_range=[xmin,xmax],yaxis_range=[-max(.2,.6*c.z),c.z+max(.12,.3*c.z)],
                          margin=dict(t=20,l=15,r=15,b=20))
        st.plotly_chart(top,use_container_width=True)
        st.caption('Diagram shows one vine face. Other-side arrays are counted in electrical costs, not double-counted in the dose on this side.')
with tab_heat:
    st.caption('Instantaneous UVC irradiance on a flat vertical canopy at one tractor position. Unlike dose, this is not integrated over time.')
    if c.n_per_array*c.arrays_per_side*(c.lamp_segments if mode=='Mercury lamp' else 1)>9000:
        st.warning('Large geometry: heatmap uses a reduced number of samples to keep the app responsive.')
    extension=max(.2,2*c.z)
    xs=np.linspace(-extension,c.single_width+(c.arrays_per_side-1)*(c.single_width+c.array_gap)+extension,65)
    ys=np.linspace(c.y_min,c.y_max,41)
    with st.spinner('Computing optical overlap...'):
        heat=instantaneous_map(c,ys,xs)
    fig=go.Figure(go.Heatmap(x=xs,y=ys,z=heat,colorscale='Viridis',colorbar=dict(title='mW/cm²')))
    fig.update_layout(xaxis_title='Position along tractor travel (m)',yaxis_title='Canopy height (m)',height=500)
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f'Maximum instantaneous sampled irradiance: {float(heat.max()):.4f} mW/cm². Model assumes no canopy shadowing or foliage-angle effects.')
with tab_budget:
    a,b,d=st.columns(3)
    a.metric('Emitters electrical load',f'{emit_elec/1000:.2f} kW')
    b.metric('Suggested generator running rating',f'{gen_req/1000:.2f} kW')
    d.metric('Estimated electrical system total',f'${total_cost:,.0f}',delta=f'${c.budget_limit-total_cost:,.0f} remaining')
    st.write(architecture)
    st.caption('Budget uses user-editable planning prices and efficiencies. Generator is sized from estimated running power with margin, NOT starting surge, ballast power factor, or vendor-verified ratings.')
    df=pd.DataFrame([{'Item':key,'Estimated cost ($)':val} for key,val in parts.items()])
    st.dataframe(df,hide_index=True,use_container_width=True)
    if total_cost>c.budget_limit: st.warning(f'Provisional estimate exceeds the ${c.budget_limit:,.0f} ceiling by ${total_cost-c.budget_limit:,.0f}.')
    else: st.success(f'Provisional estimate is within the entered ${c.budget_limit:,.0f} ceiling, subject to actual supplier quotes.')
    st.warning('Mercury lamps contain hazardous mercury and can break even with protective coatings. Use factory-rated shatter containment, outdoor-rated fixtures, vibration restraints, appropriate ballasts, shielded UVC testing, interlocks, and spill procedures.')

st.download_button('Export inputs, dose and electrical budget to Excel',
                   data=excel_bytes(c,heights,low,nominal,high,parts,emit_elec,ac_load,gen_req),
                   file_name='UVC_Vineyard_Simulation_Results.xlsx',
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with st.expander('Model equations, interpretation, and limitations'):
    st.latex(r'E_i(x,y)=\frac{P_i(m+1)}{2\pi}\frac{z^{m+1}}{[(x-x_i)^2+(y-y_i)^2+z^2]^{(m+3)/2}}')
    st.latex(r'D(y)=\frac{1}{v}\int_{-\infty}^{+\infty}\sum_i E_i(x,y)\,dx')
    st.write('LEDs use a normalized generalized-Lambertian angular profile; 90° means full width at half maximum. The mercury-lamp model uses a vertical line source made of many small emitters with a forward hemispheric (m=0) assumption, plus user-entered coating transmission and forward-directed fraction. These are not measured emission patterns.')
    st.write('The complete-pass integral is translation-invariant along tractor travel. Therefore horizontal column spacing or separation between sequential identical arrays may alter the instantaneous heatmap but not the final dose at a stationary point under these assumptions. Vertical spacing and stagger DO affect the dose across canopy height.')
    st.write('Radiant dose depends on wavelength and canopy geometry; 15 mJ/cm² at one wavelength is not necessarily equivalent to another. The model omits shadows, foliage orientation, dirt, optical aging, thermal derating, reflections, and finite-row interruptions. Validate with a calibrated wavelength-appropriate UVC radiometer.')
    st.write('Electrical sizing is a budget estimate. LED current must be actively regulated; matching string voltages, enclosure ratings, grounding, fault protection and generator startup behavior require an electrical design review.')
