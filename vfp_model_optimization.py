from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.interpolate import griddata


# References for optimization methods
method_references = {
    "Nelder-Mead": "Nelder, J. A., & Mead, R. (1965). *A simplex method for function minimization*. The Computer Journal, 7(4), 308-313.",
    "Powell": "Powell, M. J. D. (1964). *An efficient method for finding the minimum of a function of several variables without calculating derivatives*. The Computer Journal, 7(2), 155–162.",
    "L-BFGS-B": "Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on Scientific Computing, 16(5), 1190–1208.",
    "trust-constr": "Branch, M. A., Coleman, T. F., & Li, Y. (1999). *A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems*. SIAM Journal on Scientific Computing, 21(1), 1-23.",
    "COBYLA": "Powell, M. J. D. (1994). *A direct search optimization method that models the objective and constraint functions by linear interpolation*. Advances in Optimization and Numerical Analysis, 51–67.",
    "Newton-CG": "Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer Series in Operations Research."
}

# Optimization method comparison summary
optimization_summary = {
    "Method": ["Nelder-Mead", "Powell", "L-BFGS-B", "trust-constr", "COBYLA", "Newton-CG"],
    "Uses Gradient": ["No", "No", "Yes", "Yes", "No", "Yes"],
    "Supports Bounds": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
    "Supports Constraints": ["No", "No", "No", "Yes", "Yes", "Yes"],
    "Derivative-Free": ["Yes", "Yes", "No", "No", "Yes", "No"]
}

summary_df = pd.DataFrame(optimization_summary)
fracture_pressure = 5000
DEFAULT_BHP_LIMIT = 0.85 * fracture_pressure  # 85% of fracture pressure

bhp_max_limit = st.sidebar.number_input(
    "Max BHP Limit (psia)",
    value=DEFAULT_BHP_LIMIT,
    min_value=1000.0,
    max_value=10000.0,
    step=50.0
)

thp_orig = None
flow_orig = None
bhp_values = None

st.title("Upload BHP Data for VFP Optimisation")

uploaded_file = st.sidebar.file_uploader("Upload CSV with THP, FlowRate, and BHP columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Preview data
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Validate required columns
    if not {'THP', 'FlowRate', 'BHP'}.issubset(df.columns):
        st.error("CSV must include 'THP', 'FlowRate', and 'BHP' columns.")
    else:
        # Sort and extract unique levels
        thp_orig = sorted(df['THP'].unique())
        flow_orig = sorted(df['FlowRate'].unique())

        # Create a pivoted matrix of BHP values
        bhp_pivot = df.pivot_table(index='THP', columns='FlowRate', values='BHP')
        
        # Ensure matrix is complete (handle missing values if any)
        bhp_pivot = bhp_pivot.reindex(index=thp_orig, columns=flow_orig).interpolate(axis=0).interpolate(axis=1)
        bhp_values = bhp_pivot.values

        st.success("Data loaded and pivoted successfully!")
        st.write("BHP Grid:")
        st.dataframe(pd.DataFrame(bhp_values, index=thp_orig, columns=flow_orig))



        # Prepare interpolation points
        if thp_orig and flow_orig:
            THP_grid, Flow_grid = np.meshgrid(thp_orig, flow_orig, indexing='ij')
            points = np.column_stack((THP_grid.ravel(), Flow_grid.ravel()))
            values = bhp_values.ravel()

            thp_min, thp_max = min(thp_orig), max(thp_orig)
            thp_input = st.sidebar.slider("THP (psia)", thp_min, thp_max, (thp_min + thp_max) // 2, step=10)

            flow_min, flow_max = min(flow_orig), max(flow_orig)
            bounds = [(thp_min, thp_max), (flow_min, flow_max)]

            flow_rate_input = st.sidebar.slider("Flow Rate (MSCF/day)", flow_min, flow_max, round((flow_min + flow_max)/2, 2), step=0.5)

            initial_guess = [thp_input, flow_rate_input]
            
            # Streamlit App
        st.title("VFP Model: Injection Paramters Optimization")
        st.sidebar.header("User Input")    

        method = st.sidebar.selectbox(
            "Select Optimization Method",
            ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'trust-constr', 'COBYLA', 'Newton-CG']
        )

        def calculate_bhp(thp, flow_rate):
            bhp = griddata(points, values, (thp, flow_rate), method='linear')
            if bhp is None or np.isnan(bhp):
                # Fallback to nearest interpolation if linear fails
                bhp = griddata(points, values, (thp, flow_rate), method='nearest')
            return bhp


        def calculate_drawdown(thp, flow_rate):
            bhp = calculate_bhp(thp, flow_rate)
            return bhp - thp if bhp is not None else None

        def objective(x):
            return calculate_bhp(x[0], x[1])

        def numerical_gradient(func, x, epsilon=1e-6):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] -= epsilon
                x2[i] += epsilon
                try:
                    f1, f2 = func(x1), func(x2)
                    if not np.isfinite(f1) or not np.isfinite(f2):
                        grad[i] = 1e6
                    else:
                        grad[i] = (f2 - f1) / (2 * epsilon)
                except Exception:
                    grad[i] = 1e6
            return grad


        # Display manual input
        bhp_val = calculate_bhp(thp_input, flow_rate_input)
        drawdown_val = calculate_drawdown(thp_input, flow_rate_input)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Manual Input Results")
            if bhp_val is not None:
                st.write(f"**THP:** {thp_input} psia")
                st.write(f"**Flow Rate:** {flow_rate_input} MSCF/day")
                st.write(f"**BHP:** {bhp_val:.2f} psia")
                st.write(f"**Drawdown:** {drawdown_val:.2f} psia")
            else:
                st.error("Selected values are outside interpolation range.")


        constraints = [{'type': 'ineq', 'fun': lambda x: bhp_max_limit - calculate_bhp(x[0], x[1])}]

        if method == 'Nelder-Mead':
            result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
        elif method == 'Powell':
            result = minimize(objective, initial_guess, method='Powell', bounds=bounds)
        elif method == 'L-BFGS-B':
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        elif method == 'trust-constr':
            result = minimize(objective, initial_guess, method='trust-constr',
                              bounds=bounds, jac=lambda x: numerical_gradient(objective, x))
        elif method == 'COBYLA':
            result = minimize(objective, initial_guess, method='COBYLA', constraints=constraints)
        elif method == 'Newton-CG':
            result = minimize(objective, initial_guess, method='Newton-CG',
                              jac=lambda x: numerical_gradient(objective, x), bounds=bounds)

        with col2:
            st.subheader("Optimization Results")
            if result.success:
                thp_opt, flow_opt = result.x
                bhp_opt = calculate_bhp(thp_opt, flow_opt)
                drawdown_opt = calculate_drawdown(thp_opt, flow_opt)

                st.write(f"**Optimized THP:** {thp_opt:.2f} psia")
                st.write(f"**Optimized Flow Rate:** {flow_opt:.2f} MSCF/day")
                st.write(f"**Optimized BHP:** {bhp_opt:.2f} psia")
                st.write(f"**Drawdown:** {drawdown_opt:.2f} psia")
            else:
                st.error("Optimization failed.")

        # Plotting
        st.subheader("BHP Optimization Surface")
        #THP_vals = np.linspace(800, 1250, 50)
        #Flow_vals = np.linspace(10.23, 60.32, 50)
        THP_vals = np.linspace(thp_min, thp_max, 50)
        Flow_vals = np.linspace(flow_min, flow_max, 50)

        THP_grid, Flow_grid = np.meshgrid(THP_vals, Flow_vals)
        BHP_grid = np.zeros_like(THP_grid)

        for i in range(THP_grid.shape[0]):
            for j in range(THP_grid.shape[1]):
                BHP_grid[i, j] = calculate_bhp(THP_grid[i, j], Flow_grid[i, j])

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(THP_grid, Flow_grid, BHP_grid, cmap='viridis', alpha=0.9)

        if bhp_val is not None:
            ax.scatter(thp_input, flow_rate_input, bhp_val, color='blue', label="Manual Input", s=60)
        if result.success:
            ax.scatter(thp_opt, flow_opt, bhp_opt, color='red', label="Optimized", s=100)

        ax.set_xlabel("THP (psia)")
        ax.set_ylabel("Flow Rate (MSCF/day)")
        ax.set_zlabel("BHP (psia)")
        ax.legend()
        st.pyplot(fig)

        # Highlight "Yes" in green and "No" in red
        def highlight_bool(val):
            color = 'green' if val == "Yes" else 'red'
            return f'background-color: {color}; color: white; text-align: center;'

        styled_summary = summary_df.style.applymap(highlight_bool, subset=["Uses Gradient", "Supports Bounds", "Supports Constraints", "Derivative-Free"])


        with st.expander("Optimization Methods Comparison Table"):
            st.dataframe(styled_summary, use_container_width=True)


        with st.expander("Optimization Method Reference"):
            st.markdown(f"<sub><i>{method_references[method]}</i></sub>", unsafe_allow_html=True)

        methods_to_test = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'trust-constr', 'COBYLA', 'Newton-CG']

        results = []

        for method in methods_to_test:
            try:
                if method in ['Nelder-Mead', 'Powell', 'L-BFGS-B']:
                    res = minimize(objective, initial_guess, method=method, bounds=bounds)
                elif method == 'trust-constr':
                    res = minimize(objective, initial_guess, method='trust-constr', bounds=bounds,
                                   jac=lambda x: numerical_gradient(objective, x))
                elif method == 'COBYLA':
                    res = minimize(objective, initial_guess, method='COBYLA',
                                   constraints={'type': 'ineq', 'fun': lambda x: 3500 - calculate_bhp(x[0], x[1])})
                elif method == 'Newton-CG':
                    res = minimize(objective, initial_guess, method='Newton-CG',
                                   jac=lambda x: numerical_gradient(objective, x))

                if res.success:
                    thp_opt, flow_opt = res.x
                    bhp_opt = calculate_bhp(thp_opt, flow_opt)
                    drawdown_opt = calculate_drawdown(thp_opt, flow_opt)
                    results.append({
                    "Method": method,
                    "THP (psia)": round(float(thp_opt), 2),
                    "Flow Rate (MSCF/day)": round(float(flow_opt), 2),
                    "BHP (psia)": round(float(bhp_opt), 2),
                    "Drawdown (psia)": round(float(drawdown_opt), 2),
                    "Success": "Yes"
                    })
                else:
                    results.append({
                        "Method": method,
                        "THP (psia)": None,
                        "Flow Rate (MSCF/day)": None,
                        "BHP (psia)": None,
                        "Drawdown (psia)": None,
                        "Success": "No"
                    })
            except Exception as e:
                results.append({
                    "Method": method,
                    "THP (psia)": None,
                    "Flow Rate (MSCF/day)": None,
                    "BHP (psia)": None,
                    "Drawdown (psia)": None,
                    "Success": f"Error: {str(e)}"
                })

        # Create a DataFrame for manual input
        manual_input_df = pd.DataFrame([{
            "Method": "Manual Input",
            "THP (psia)": round(float(thp_input), 2),
            "Flow Rate (MSCF/day)": round(float(flow_rate_input), 2),
            "BHP (psia)": round(float(bhp_val), 2),
            "Drawdown (psia)": round(float(drawdown_val), 2),
            "Success": "N/A"
        }])


        result_df = pd.DataFrame(results)

        combined_df = pd.concat([manual_input_df, result_df], ignore_index=True)
        st.subheader("Optimization Results & Manual Input Summary")
        st.dataframe(combined_df, use_container_width=True)
        combined_df = combined_df.sort_values(by="Drawdown (psia)")

        # Determine color coding for drawdown
        min_drawdown = combined_df['Drawdown (psia)'].min()
        colors = [
            'green' if val == min_drawdown else 'orange' if val < 1000 else 'red'
            for val in combined_df['Drawdown (psia)']
        ]

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(combined_df['Method'], combined_df['Drawdown (psia)'], color=colors)

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 20, f'{height:.2f}', ha='center', va='bottom')

        # Styling
        ax.set_title('Drawdown Comparison Across Methods')
        ax.set_ylabel('Drawdown (psia)')
        ax.set_ylim(0, combined_df['Drawdown (psia)'].max() + 200)
        ax.tick_params(axis='x', rotation=45)

        # Add legend
        legend_elements = [
            Patch(facecolor='green', label='Best (Minimum Drawdown)'),
            Patch(facecolor='orange', label='Acceptable (<1000 psia)'),
            Patch(facecolor='red', label='High Drawdown')
        ]
        ax.legend(handles=legend_elements)

        # Display in Streamlit
        st.subheader("Drawdown Comparison Chart")
        st.pyplot(fig)

        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="vfp_optimization_results.csv", mime='text/csv')

else:
    #st.warning("Please upload a valid CSV with THP and FlowRate data to proceed.")
        # Load placeholder data
    st.warning("No file uploaded. Using placeholder data.")
    df = pd.read_csv("input.csv")
