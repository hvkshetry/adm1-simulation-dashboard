import streamlit as st
import pandas as pd
import numpy as np
from chemicals.elements import molecular_weight as get_mw
from qsdsan import sanunits as su, processes as pc, WasteStream, System, set_thermo

# Initialize thermo - this is critical to fix the error
cmps = pc.create_adm1_cmps()
set_thermo(cmps)


def run_simulation(Q, Temp, HRT,
                   concentrations, kinetic_params,
                   simulation_time, t_step, method,
                   use_kinetics=True):
    """
    Run ADM1 with either user-provided kinetic parameters (if use_kinetics=True) 
    or default QSDsan parameters (if use_kinetics=False).
    """
    try:
        if use_kinetics and kinetic_params:
            adm1 = pc.ADM1(**kinetic_params)
        else:
            # Use default kinetics
            adm1 = pc.ADM1()  # no overrides

        # Create streams
        inf = WasteStream('Influent', T=Temp)
        eff = WasteStream('Effluent', T=Temp)
        gas = WasteStream('Biogas')

        C_mw = get_mw({'C': 1})
        N_mw = get_mw({'N': 1})
        default_conc = {
            'S_su': 0.01,
            'S_aa': 1e-3,
            'S_fa': 1e-3,
            'S_va': 1e-3,
            'S_bu': 1e-3,
            'S_pro': 1e-3,
            'S_ac': 1e-3,
            'S_h2': 1e-8,
            'S_ch4': 1e-5,
            'S_IC': 0.04 * C_mw,
            'S_IN': 0.01 * N_mw,
            'S_I': 0.02,
            'X_c': 2.0,
            'X_ch': 5.0,
            'X_pr': 20.0,
            'X_li': 5.0,
            'X_su': 1e-2,
            'X_aa': 1e-2,
            'X_fa': 1e-2,
            'X_c4': 1e-2,
            'X_pro': 1e-2,
            'X_ac': 1e-2,
            'X_h2': 1e-2,
            'X_I': 25,
            'S_cat': 0.04,
            'S_an': 0.02,
        }
        for k,v in concentrations.items():
            if k in default_conc:
                default_conc[k] = v

        inf_kwargs = {
            'concentrations': default_conc,
            'units': ('m3/d', 'kg/m3')
        }
        inf.set_flow_by_concentration(Q, **inf_kwargs)

        # AnaerobicCSTR
        AD = su.AnaerobicCSTR(
            'AD', ins=inf, outs=(gas, eff),
            model=adm1, V_liq=Q*HRT, V_gas=Q*HRT*0.1, T=Temp
        )
        
        # Default init cond
        default_init_conds = {
            'S_su': 0.0124*1e3,
            'S_aa': 0.0055*1e3,
            'S_fa': 0.1074*1e3,
            'S_va': 0.0123*1e3,
            'S_bu': 0.0140*1e3,
            'S_pro': 0.0176*1e3,
            'S_ac': 0.0893*1e3,
            'S_h2': 2.5055e-7*1e3,
            'S_ch4': 0.0555*1e3,
            'S_IC': 0.0951*C_mw*1e3,
            'S_IN': 0.0945*N_mw*1e3,
            'S_I': 0.1309*1e3,
            'X_ch': 0.0205*1e3,
            'X_pr': 0.0842*1e3,
            'X_li': 0.0436*1e3,
            'X_su': 0.3122*1e3,
            'X_aa': 0.9317*1e3,
            'X_fa': 0.3384*1e3,
            'X_c4': 0.3258*1e3,
            'X_pro': 0.1011*1e3,
            'X_ac': 0.6772*1e3,
            'X_h2': 0.2848*1e3,
            'X_I': 17.2162*1e3
        }
        AD.set_init_conc(**default_init_conds)

        sys = System('Anaerobic_Digestion', path=(AD,))
        sys.set_dynamic_tracker(eff, gas)
        
        # Run dynamic
        sys.simulate(
            state_reset_hook='reset_cache',
            t_span=(0, simulation_time),
            t_eval=np.arange(0, simulation_time+t_step, t_step),
            method=method
        )
        return sys, inf, eff, gas
    except Exception as e:
        st.error(f"Error running simulation: {e}")
        return None, None, None, None


def create_influent_stream(Q, Temp, concentrations):
    """Create an Influent stream without running the simulation, for display."""
    try:
        inf = WasteStream('Influent', T=Temp)
        C_mw = get_mw({'C': 1})
        N_mw = get_mw({'N': 1})
        
        default_conc = {
            'S_su': 0.01,
            'S_aa': 1e-3,
            'S_fa': 1e-3,
            'S_va': 1e-3,
            'S_bu': 1e-3,
            'S_pro': 1e-3,
            'S_ac': 1e-3,
            'S_h2': 1e-8,
            'S_ch4': 1e-5,
            'S_IC': 0.04 * C_mw,
            'S_IN': 0.01 * N_mw,
            'S_I': 0.02,
            'X_c': 2.0,
            'X_ch': 5.0,
            'X_pr': 20.0,
            'X_li': 5.0,
            'X_su': 1e-2,
            'X_aa': 1e-2,
            'X_fa': 1e-2,
            'X_c4': 1e-2,
            'X_pro': 1e-2,
            'X_ac': 1e-2,
            'X_h2': 1e-2,
            'X_I': 25,
            'S_cat': 0.04,
            'S_an': 0.02,
        }
        for k, value in concentrations.items():
            if k in default_conc:
                default_conc[k] = value
        
        inf_kwargs = {
            'concentrations': default_conc,
            'units': ('m3/d', 'kg/m3')
        }
        inf.set_flow_by_concentration(Q, **inf_kwargs)
        return inf
    except Exception as e:
        st.error(f"Error creating influent stream: {e}")
        return None


def display_liquid_stream(stream):
    """Display key properties of a liquid stream (influent/effluent)."""
    flow = 0
    try:
        flow = stream.get_total_flow('m3/d')
    except:
        try:
            flow = stream.F_vol / 1000 * 24
        except:
            pass
    
    def safe_composite(stream, param):
        try:
            if hasattr(stream, 'composite'):
                return stream.composite(param)
            return 'N/A'
        except:
            return 'N/A'
    
    data = {
        'Parameter': [
            'Flow', 'pH', 'Alkalinity', 'COD', 'BOD', 
            'TN', 'TP', 'TSS', 'VSS'
        ],
        'Value': [
            f"{flow:,.2f} m³/d",
            f"{getattr(stream, 'pH', 'N/A')}",
            f"{getattr(stream, 'alkalinity', 'N/A')} mg/L",
            f"{safe_composite(stream, 'COD'):,.1f} mg/L" if safe_composite(stream, 'COD') != 'N/A' else "N/A mg/L",
            f"{safe_composite(stream, 'BOD'):,.1f} mg/L" if safe_composite(stream, 'BOD') != 'N/A' else "N/A mg/L",
            f"{safe_composite(stream, 'N'):,.1f} mg/L" if safe_composite(stream, 'N') != 'N/A' else "N/A mg/L",
            f"{safe_composite(stream, 'P'):,.1f} mg/L" if safe_composite(stream, 'P') != 'N/A' else "N/A mg/L",
            f"{safe_composite(stream, 'solids'):,.1f} mg/L" if safe_composite(stream, 'solids') != 'N/A' else "N/A mg/L",
            f"N/A mg/L"
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)


def display_gas_stream(stream):
    """
    Display gas stream properties with proper unit conversions:
    - Flow rate in Nm³/d
    - Methane and CO2 in vol/vol%
    - H2 in ppmv
    """
    MW_CH4 = 16.04
    MW_CO2 = 44.01
    MW_H2 = 2.02
    MW_C = 12.01
    
    DENSITY_CH4 = 0.716
    DENSITY_CO2 = 1.977
    DENSITY_H2 = 0.0899
    
    COD_CH4 = 4.0
    COD_H2 = 8.0
    
    flow_vol_total = 0.0
    methane_flow = 0.0
    co2_flow = 0.0
    h2_flow = 0.0
    
    try:
        if hasattr(stream, 'imass'):
            mass_cod_ch4 = stream.imass['S_ch4'] * 24
            mass_ch4 = mass_cod_ch4 / COD_CH4
            methane_flow = mass_ch4 / DENSITY_CH4

            mass_c = stream.imass['S_IC'] * 24
            mass_co2 = mass_c * (MW_CO2 / MW_C)
            co2_flow = mass_co2 / DENSITY_CO2

            mass_cod_h2 = stream.imass['S_h2'] * 24
            mass_h2 = mass_cod_h2 / COD_H2
            h2_flow = mass_h2 / DENSITY_H2

        flow_vol_total = methane_flow + co2_flow + h2_flow
        methane_pct = (methane_flow / flow_vol_total * 100) if flow_vol_total > 0 else 0
        co2_pct = (co2_flow / flow_vol_total * 100) if flow_vol_total > 0 else 0
        h2_ppmv = (h2_flow / flow_vol_total * 1e6) if flow_vol_total > 0 else 0

    except Exception as e:
        st.error(f"Error calculating gas properties: {str(e)}")
        methane_pct = 0.0
        co2_pct = 0.0
        h2_ppmv = 0.0
        flow_vol_total = 0.0

    data = {
        'Parameter': ['Flow', 'Methane', 'CO2', 'H2'],
        'Value': [
            f"{flow_vol_total:.2f} Nm³/d",
            f"{methane_pct:.2f} vol/vol%",
            f"{co2_pct:.2f} vol/vol%",
            f"{h2_ppmv:.2f} ppmv"
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)
