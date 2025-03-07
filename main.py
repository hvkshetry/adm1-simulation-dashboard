import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Import custom modules
from simulation_helpers import (
    run_simulation, create_influent_stream, 
    display_liquid_stream, display_gas_stream
)
from ui_helpers import (
    set_page_styling, display_branding_header, display_footer
)
from ai_helpers import (
    get_ai_recommendations, parse_ai_recommendations
)

# ======================= MAIN APP =======================
def main():
    st.set_page_config(page_title="Puran Water ADM1 Simulation", layout="wide")
    set_page_styling()
    display_branding_header()
    
    st.title("Anaerobic Digestion Model No. 1 (ADM1) Simulation Dashboard")
    st.markdown("""
    This dashboard allows you to run **three concurrent ADM1 simulations** with different reactor 
    parameters on the **same feedstock**. Use the AI assistant to get recommended feedstock 
    state variables (and optionally kinetic parameters), then compare simulation results.
    """)

    # ------------- Session State Initialization -------------
    # 1) Feedstock composition
    if 'influent_values' not in st.session_state:
        st.session_state.influent_values = {}
    if 'influent_explanations' not in st.session_state:
        st.session_state.influent_explanations = {}
    
    # 2) Kinetic parameters
    if 'kinetic_params' not in st.session_state:
        st.session_state.kinetic_params = {}
    if 'kinetic_explanations' not in st.session_state:
        st.session_state.kinetic_explanations = {}
    
    # 3) Whether or not we are including kinetics
    if 'use_kinetics' not in st.session_state:
        st.session_state.use_kinetics = False  # default

    # 4) Common flow
    if 'Q' not in st.session_state:
        st.session_state.Q = 170.0
    
    # 5) Simulation parameters for each of the three concurrent sims
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = [
            {'Temp': 308.15, 'HRT': 30.0, 'method': 'BDF'},
            {'Temp': 308.15, 'HRT': 45.0, 'method': 'BDF'},
            {'Temp': 308.15, 'HRT': 60.0, 'method': 'BDF'}
        ]
    
    # 6) Store results of all three simulations
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = [None, None, None]
    
    # 7) Simulation time & step
    if 'simulation_time' not in st.session_state:
        st.session_state.simulation_time = 150.0
    if 't_step' not in st.session_state:
        st.session_state.t_step = 0.1
    
    # 8) AI raw response storage
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = None

    # ================ SIDEBAR ================
    with st.sidebar:
        st.header("Feedstock & Simulation Setup")
        
        # --- AI Assistant Mode Selection ---
        st.subheader("AI Assistant Mode")
        mode = st.radio(
            "Choose what the AI should provide:",
            ("Feedstock State Variables Only", "Feedstock + Reaction Kinetics"),
            index=0
        )
        # This bool will decide if we ask for kinetics or not
        st.session_state.use_kinetics = (mode == "Feedstock + Reaction Kinetics")

        # --- AI Assistant for feedstock (and possibly kinetics) ---
        st.subheader("AI Feedstock Assistant")
        feedstock_description = st.text_area(
            "Describe your feedstock in natural language:",
            placeholder="Example: Food waste with ~40% carbs, ~20% proteins, ~10% lipids, etc."
        )
        if st.button("Get AI Recommendations"):
            with st.spinner("Getting AI recommendations..."):
                response = get_ai_recommendations(
                    feedstock_description,
                    include_kinetics=st.session_state.use_kinetics
                )
                if response:
                    st.session_state.ai_recommendations = response
                    # Parse the JSON
                    (fv, fe, kv, ke) = parse_ai_recommendations(
                        response,
                        include_kinetics=st.session_state.use_kinetics
                    )
                    
                    # Update session state
                    if fv:
                        st.session_state.influent_values.update(fv)
                    if fe:
                        st.session_state.influent_explanations.update(fe)
                    
                    # If user only wants feedstock, do not update or store kinetic data
                    if st.session_state.use_kinetics and kv:
                        st.session_state.kinetic_params.update(kv)
                    if st.session_state.use_kinetics and ke:
                        st.session_state.kinetic_explanations.update(ke)
                    
                    st.success("AI recommendations parsed and stored successfully!")

        # --- Common Influent Flow ---
        st.subheader("Common Influent Flow")
        Q_new = st.number_input(
            "Influent Flow Rate (m³/d)",
            min_value=1.0,
            value=float(st.session_state.Q),
            step=1.0
        )
        if Q_new != st.session_state.Q:
            st.session_state.Q = Q_new

        # --- Reactor & Integration Parameters (Three Simulations) ---
        st.subheader("Parameters for Each Simulation")
        for i in range(3):
            st.markdown(f"**Simulation {i+1}**")
            temp_val = st.number_input(
                f"Temperature (K) for Sim {i+1}",
                min_value=273.15,
                value=float(st.session_state.sim_params[i]['Temp']),
                step=0.1,
                key=f"temp_sim_{i}"
            )
            hrt_val = st.number_input(
                f"HRT (days) for Sim {i+1}",
                min_value=1.0,
                value=float(st.session_state.sim_params[i]['HRT']),
                step=1.0,
                key=f"hrt_sim_{i}"
            )
            method_val = st.selectbox(
                f"Integration Method for Sim {i+1}",
                ["BDF","RK45","RK23","DOP853","Radau","LSODA"],
                index=["BDF","RK45","RK23","DOP853","Radau","LSODA"].index(
                    st.session_state.sim_params[i]['method']
                ),
                key=f"method_sim_{i}"
            )
            st.session_state.sim_params[i]['Temp'] = temp_val
            st.session_state.sim_params[i]['HRT'] = hrt_val
            st.session_state.sim_params[i]['method'] = method_val
        
        # --- Simulation time and step
        st.subheader("Simulation Time & Step")
        sim_time = st.slider(
            "Simulation Time (days)",
            10.0, 300.0,
            st.session_state.simulation_time,
            step=5.0
        )
        t_step = st.slider(
            "Time Step (days)",
            0.01, 1.0,
            st.session_state.t_step,
            step=0.01
        )
        st.session_state.simulation_time = sim_time
        st.session_state.t_step = t_step

        # --- Run All Simulations ---
        if st.button("Run All Simulations"):
            with st.spinner("Creating influent stream..."):
                common_inf = create_influent_stream(
                    Q=st.session_state.Q,
                    Temp=st.session_state.sim_params[0]['Temp'],
                    concentrations=st.session_state.influent_values
                )
            
            # Run each sim
            for i in range(3):
                with st.spinner(f"Running Simulation {i+1}..."):
                    sys_i, inf_i, eff_i, gas_i = run_simulation(
                        Q=st.session_state.Q,
                        Temp=st.session_state.sim_params[i]['Temp'],
                        HRT=st.session_state.sim_params[i]['HRT'],
                        concentrations=st.session_state.influent_values,
                        kinetic_params=st.session_state.kinetic_params,
                        simulation_time=st.session_state.simulation_time,
                        t_step=st.session_state.t_step,
                        method=st.session_state.sim_params[i]['method'],
                        use_kinetics=st.session_state.use_kinetics
                    )
                    st.session_state.sim_results[i] = (sys_i, inf_i, eff_i, gas_i)
            st.success("All simulations completed successfully!")

    # ================ MAIN CONTENT ================
    col1, col2 = st.columns([1, 2])
    
    # ---------- Left Column: Feedstock & Kinetics + Streams ----------
    with col1:
        st.header("Model & Stream Data")
        
        # --- AI raw response (optional display) ---
        if st.session_state.ai_recommendations:
            with st.expander("AI Recommendations (Raw JSON)", expanded=False):
                st.markdown(f"```\n{st.session_state.ai_recommendations}\n```")

        # 1) Manual feedstock parameter input
        with st.expander("Manual Feedstock Parameter Input", expanded=False):
            from chemicals.elements import molecular_weight as get_mw
            C_mw = get_mw({'C':1})
            N_mw = get_mw({'N':1})
            
            feedstock_defaults = {
                'S_su': 0.01,
                'S_aa': 1e-3,
                'S_fa': 1e-3,
                'S_va': 1e-3,
                'S_bu': 1e-3,
                'S_pro': 1e-3,
                'S_ac': 1e-3,
                'S_h2': 1e-8,
                'S_ch4': 1e-5,
                'S_IC': 0.04*C_mw,
                'S_IN': 0.01*N_mw,
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
            for k,v in st.session_state.influent_values.items():
                feedstock_defaults[k] = v
            
            st.subheader("Soluble Components (S_)")
            sol_keys = ['S_su','S_aa','S_fa','S_va','S_bu','S_pro','S_ac','S_h2','S_ch4','S_IC','S_IN','S_I','S_cat','S_an']
            cols_s = st.columns(3)
            for i, key in enumerate(sol_keys):
                with cols_s[i % 3]:
                    val = st.number_input(
                        f"{key} [kg/m³]",
                        value=float(feedstock_defaults.get(key, 0.0)),
                        format="%.6f",
                        key=f"feed_{key}"
                    )
                    st.session_state.influent_values[key] = val
            
            st.markdown("---")
            st.subheader("Particulate Components (X_)")
            part_keys = ['X_c','X_ch','X_pr','X_li','X_su','X_aa','X_fa','X_c4','X_pro','X_ac','X_h2','X_I']
            cols_x = st.columns(3)
            for i, key in enumerate(part_keys):
                with cols_x[i % 3]:
                    val = st.number_input(
                        f"{key} [kg/m³]",
                        value=float(feedstock_defaults.get(key, 0.0)),
                        format="%.4f",
                        key=f"feed_{key}"
                    )
                    st.session_state.influent_values[key] = val
            
            if st.button("Update Feedstock Parameters"):
                st.success("Feedstock parameters updated!")

        # 2) Manual Kinetic Parameter Input (only relevant if user selected Kinetics)
        if st.session_state.use_kinetics:
            with st.expander("Manual Kinetic Parameter Input", expanded=False):
                kinetic_defaults = {
                    "q_dis": 0.5,
                    "q_ch_hyd": 10.0,
                    "q_pr_hyd": 10.0,
                    "q_li_hyd": 10.0,
                    "k_su": 30.0,
                    "k_aa": 50.0,
                    "k_fa": 6.0,
                    "k_c4": 20.0,
                    "k_pro": 13.0,
                    "k_ac": 8.0,
                    "k_h2": 35.0,
                    "b_su": 0.02,
                    "b_aa": 0.02,
                    "b_fa": 0.02,
                    "b_c4": 0.02,
                    "b_pro": 0.02,
                    "b_ac": 0.02,
                    "b_h2": 0.02,
                    "K_su": 0.5,
                    "K_aa": 0.3,
                    "K_fa": 0.4,
                    "K_c4": 0.2,
                    "K_pro": 0.1,
                    "K_ac": 0.15,
                    "K_h2": 7e-6,
                    "KI_h2_fa": 5e-6,
                    "KI_h2_c4": 1e-5,
                    "KI_h2_pro": 3.5e-6,
                    "KI_nh3": 1.8e-3,
                    "KS_IN": 1e-4,
                    "Y_su": 0.1,
                    "Y_aa": 0.08,
                    "Y_fa": 0.06,
                    "Y_c4": 0.06,
                    "Y_pro": 0.04,
                    "Y_ac": 0.05,
                    "Y_h2": 0.06,
                    "f_bu_su": 0.13,
                    "f_pro_su": 0.27,
                    "f_ac_su": 0.41,
                    "f_va_aa": 0.23,
                    "f_bu_aa": 0.26,
                    "f_pro_aa": 0.05,
                    "f_ac_aa": 0.40,
                    "f_ac_fa": 0.7,
                    "f_pro_va": 0.54,
                    "f_ac_va": 0.31,
                    "f_ac_bu": 0.8,
                    "f_ac_pro": 0.57
                }
                for k,v in st.session_state.kinetic_params.items():
                    kinetic_defaults[k] = v
                
                param_list = sorted(list(kinetic_defaults.keys()))
                cols_k = st.columns(3)
                for i, p in enumerate(param_list):
                    with cols_k[i % 3]:
                        val = st.number_input(
                            f"{p}",
                            value=float(kinetic_defaults[p]),
                            format="%.6f",
                            key=f"kin_{p}"
                        )
                        st.session_state.kinetic_params[p] = val
                
                if st.button("Update Kinetic Parameters"):
                    st.success("Kinetic parameters updated!")

        # 3) Show stream properties
        with st.expander("Stream Properties", expanded=True):
            tabs_streams = st.tabs([
                "Influent (Common)",
                "Effluent Sim 1", "Biogas Sim 1",
                "Effluent Sim 2", "Biogas Sim 2",
                "Effluent Sim 3", "Biogas Sim 3"
            ])

            with tabs_streams[0]:
                st.markdown("**Influent**")
                temp_inf = create_influent_stream(
                    Q=st.session_state.Q,
                    Temp=308.15,
                    concentrations=st.session_state.influent_values
                )
                if temp_inf:
                    display_liquid_stream(temp_inf)
            
            for i in range(3):
                sys_res = st.session_state.sim_results[i]
                eff_tab = tabs_streams[1 + 2*i]
                gas_tab = tabs_streams[2 + 2*i]

                with eff_tab:
                    st.markdown(f"**Effluent - Simulation {i+1}**")
                    if sys_res and all(sys_res):
                        _, _, eff_i, _ = sys_res
                        display_liquid_stream(eff_i)
                    else:
                        st.info(f"No results for Simulation {i+1} yet.")
                
                with gas_tab:
                    st.markdown(f"**Biogas - Simulation {i+1}**")
                    if sys_res and all(sys_res):
                        _, _, _, gas_i = sys_res
                        display_gas_stream(gas_i)
                    else:
                        st.info(f"No results for Simulation {i+1} yet.")

    # ------------- Right Column: Simulation Results & Plots -------------
    with col2:
        st.header("Simulation Results")

        any_sim_ran = any([res is not None and all(res) for res in st.session_state.sim_results])
        if not any_sim_ran:
            st.info("Run the simulations to see results here.")
            display_footer()
            return

        plot_type = st.selectbox(
            "Select Plot Type",
            [
                "Effluent - Acids",
                "Effluent - Inorganic Carbon",
                "Effluent - Biomass Components",
                "Gas - Hydrogen",
                "Gas - Methane",
                "Total VFAs"
            ]
        )

        fig = go.Figure()

        for i, (sys_i, inf_i, eff_i, gas_i) in enumerate(st.session_state.sim_results):
            if not sys_i or not inf_i or not eff_i or not gas_i:
                continue
            t_stamp_eff = eff_i.scope.time_series
            t_stamp_gas = gas_i.scope.time_series
            cmps_obj = inf_i.components

            if plot_type == "Effluent - Acids":
                acid_list = ['S_su','S_aa','S_fa','S_va','S_bu','S_pro','S_ac']
                for acid in acid_list:
                    idx = cmps_obj.index(acid)
                    data_acid = eff_i.scope.record[:, idx]
                    fig.add_trace(go.Scatter(
                        x=t_stamp_eff,
                        y=data_acid,
                        mode='lines',
                        name=f"{acid} (Sim {i+1})"
                    ))
            elif plot_type == "Effluent - Inorganic Carbon":
                idx = cmps_obj.index('S_IC')
                data_ic = eff_i.scope.record[:, idx]
                fig.add_trace(go.Scatter(
                    x=t_stamp_eff,
                    y=data_ic,
                    mode='lines',
                    name=f"S_IC (Sim {i+1})"
                ))
            elif plot_type == "Effluent - Biomass Components":
                bio_list = ['X_su','X_aa','X_fa','X_c4','X_pro','X_ac','X_h2']
                for bio in bio_list:
                    idx = cmps_obj.index(bio)
                    data_bio = eff_i.scope.record[:, idx]
                    fig.add_trace(go.Scatter(
                        x=t_stamp_eff,
                        y=data_bio,
                        mode='lines',
                        name=f"{bio} (Sim {i+1})"
                    ))
            elif plot_type == "Gas - Hydrogen":
                idx = cmps_obj.index('S_h2')
                data_h2 = gas_i.scope.record[:, idx]
                fig.add_trace(go.Scatter(
                    x=t_stamp_gas,
                    y=data_h2,
                    mode='lines',
                    name=f"S_h2 (Sim {i+1})"
                ))
            elif plot_type == "Gas - Methane":
                idx_ch4 = cmps_obj.index('S_ch4')
                idx_ic = cmps_obj.index('S_IC')
                data_ch4 = gas_i.scope.record[:, idx_ch4]
                data_ic = gas_i.scope.record[:, idx_ic]
                fig.add_trace(go.Scatter(
                    x=t_stamp_gas,
                    y=data_ch4,
                    mode='lines',
                    name=f"S_ch4 (Sim {i+1})"
                ))
                fig.add_trace(go.Scatter(
                    x=t_stamp_gas,
                    y=data_ic,
                    mode='lines',
                    name=f"S_IC (Sim {i+1})"
                ))
            elif plot_type == "Total VFAs":
                idx_vfa = cmps_obj.indices(['S_va','S_bu','S_pro','S_ac'])
                vfa_matrix = eff_i.scope.record[:, idx_vfa]
                total_vfa = np.sum(vfa_matrix, axis=1)
                fig.add_trace(go.Scatter(
                    x=t_stamp_eff,
                    y=total_vfa,
                    mode='lines',
                    name=f"Total VFAs (Sim {i+1})"
                ))

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            height=600
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            title="Time [d]"
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            title="Concentration [mg/L]"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Export/Download options
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Export Plot"):
                fig.write_html("adm1_three_sim_plot.html")
                with open("adm1_three_sim_plot.html", "rb") as file:
                    st.download_button(
                        label="Download Interactive Plot (HTML)",
                        data=file,
                        file_name="adm1_three_sim_plot.html",
                        mime="text/html"
                    )
        with c2:
            if st.button("Export Data"):
                df_list = []
                for i, (sys_i, inf_i, eff_i, gas_i) in enumerate(st.session_state.sim_results):
                    if eff_i and gas_i:
                        t_eff = eff_i.scope.time_series
                        df_sim = pd.DataFrame({"Time_d": t_eff})
                        
                        if plot_type == "Total VFAs":
                            idx_vfa = inf_i.components.indices(['S_va','S_bu','S_pro','S_ac'])
                            vfa_matrix = eff_i.scope.record[:, idx_vfa]
                            total_vfa = np.sum(vfa_matrix, axis=1)
                            df_sim[f"Total_VFA_Sim{i+1}"] = total_vfa
                        elif plot_type == "Effluent - Acids":
                            acid_list = ['S_su','S_aa','S_fa','S_va','S_bu','S_pro','S_ac']
                            for acid in acid_list:
                                idx = inf_i.components.index(acid)
                                df_sim[f"{acid}_Sim{i+1}"] = eff_i.scope.record[:, idx]
                        elif plot_type == "Effluent - Inorganic Carbon":
                            idx = inf_i.components.index('S_IC')
                            df_sim[f"S_IC_Sim{i+1}"] = eff_i.scope.record[:, idx]
                        elif plot_type == "Effluent - Biomass Components":
                            bio_list = ['X_su','X_aa','X_fa','X_c4','X_pro','X_ac','X_h2']
                            for bio in bio_list:
                                idx = inf_i.components.index(bio)
                                df_sim[f"{bio}_Sim{i+1}"] = eff_i.scope.record[:, idx]
                        elif plot_type == "Gas - Hydrogen":
                            idx_h2 = inf_i.components.index('S_h2')
                            df_sim[f"S_h2_Sim{i+1}"] = gas_i.scope.record[:, idx_h2]
                        elif plot_type == "Gas - Methane":
                            idx_ch4 = inf_i.components.index('S_ch4')
                            idx_ic = inf_i.components.index('S_IC')
                            df_sim[f"S_ch4_Sim{i+1}"] = gas_i.scope.record[:, idx_ch4]
                            df_sim[f"S_IC_Sim{i+1}"] = gas_i.scope.record[:, idx_ic]
                        
                        df_list.append(df_sim)
                
                if df_list:
                    df_merged = df_list[0]
                    for d in df_list[1:]:
                        df_merged = pd.merge(df_merged, d, on="Time_d", how="outer")
                    df_merged.sort_values(by="Time_d", inplace=True)
                    csv_data = df_merged.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="adm1_three_sim_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data available to export.")

        display_footer()


if __name__ == "__main__":
    main()
