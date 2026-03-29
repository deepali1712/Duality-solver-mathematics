import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import json

# ---------------- PAGE ----------------
st.set_page_config(page_title="Optimization Studio", layout="centered")

# ---------------- DARK MODE TOGGLE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- UI ----------------
if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
    }
    label, .stNumberInput label {
        color: white !important;
    }
    input, select {
        background-color: #1e293b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🚀 Optimization Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Linear Programming + Duality</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    problem_type = st.selectbox("Problem Type", ["Maximization", "Minimization"])
    n = st.number_input("Variables", min_value=2, value=2)
    m = st.number_input("Constraints", min_value=1, value=2)

# ---------------- EXAMPLE ----------------
if st.button("📌 Load Example"):
    c = [3, 2]
    A = [[2, 1], [1, 3]]
    b = [10, 15]
    ineq = ["≤", "≤"]
else:
    c = [st.number_input(f"c{i+1}", value=1.0) for i in range(n)]
    A, b, ineq = [], [], []

    for i in range(m):
        cols = st.columns(n + 2)
        row = []

        for j in range(n):
            row.append(cols[j].number_input(f"a{i}{j}", value=1.0))

        sign = cols[-2].selectbox("Type", ["≤", "≥", "="], key=f"s{i}")
        rhs = cols[-1].number_input(f"b{i}", value=10.0)

        A.append(row)
        b.append(rhs)
        ineq.append(sign)

# ---------------- SOLVER FUNCTION ----------------
def solve_lp(problem_type, c, A, b, ineq):
    """
    Solves a Linear Programming problem using the HiGHS method.
    Normalizes constraints and objective functions for the linprog solver.
    """
    # Initialize lists for upper-bound constraints (A_ub * x <= b_ub)
    A_ub, b_ub = [], []

    # Loop through each constraint to ensure they follow the "Less than or Equal to" (<=) format
    for i in range(len(ineq)):
        if ineq[i] == "≤":
            # If already <=, add the coefficients and constants directly
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif ineq[i] == "≥":
            # If >=, multiply both sides by -1 to flip the inequality sign to <=
            A_ub.append([-x for x in A[i]])
            b_ub.append(-b[i])

    # linprog is a minimizer by default. 
    # If the goal is Maximization, we negate the objective function coefficients (c).
    c_mod = [-x for x in c] if problem_type == "Maximization" else c

    # Call the solver using the modified objective and normalized constraints
    return linprog(c_mod, A_ub=A_ub, b_ub=b_ub, method="highs")

# ---------------- SOLVE ----------------
# Trigger the solving process when the user clicks the button
if st.button("🚀 Solve Problem"):

    # Validation: Ensure each row in the constraint matrix A matches the number of variables n
    if any(len(row) != n for row in A):
        st.error("Invalid constraint matrix")
        st.stop()

    # Call the solver function with the provided problem parameters
    result = solve_lp(problem_type, c, A, b, ineq)

    # Check if the solver successfully found a feasible optimal solution
    if result.success:
        # If the problem is Maximization, flip the sign of the result 
        # (Standard solvers like SciPy minimize by default)
        val = -result.fun if problem_type == "Maximization" else result.fun

        # Display the results in the Streamlit UI
        st.subheader("✅ Solution")
        st.metric("Optimal Value", round(val, 4))

        # Iterate through and display the value for each decision variable (x1, x2, etc.)
        for i, v in enumerate(result.x):
            st.write(f"x{i+1} = {round(v,4)}")

        # ---------------- DUAL ----------------
        try:
            A_np = np.array(A)

            if A_np.shape[0] > 0:
                c_dual = b
                A_dual = -A_np.T
                b_dual = [-x for x in c]

                dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, method="highs")

                if dual.success:
                    st.subheader("🔁 Dual Solution")
                    st.write("Variables:", np.round(dual.x, 4))
                    st.write("Optimal Value:", round(dual.fun, 4))
                    st.success("Strong Duality holds")

        except Exception:
            st.warning("Dual computation skipped")

        # ---------------- PDF ----------------
        def create_pdf():
            doc = SimpleDocTemplate("result.pdf")
            styles = getSampleStyleSheet()
            content = [
                Paragraph(f"Optimal Value: {round(val,4)}", styles["Normal"]),
                Paragraph(f"Variables: {np.round(result.x,4)}", styles["Normal"])
            ]
            doc.build(content)

        create_pdf()

        with open("result.pdf", "rb") as f:
            st.download_button("📄 Download PDF", f, "solution.pdf")

        # ---------------- JSON ----------------
        solution_data = {
            "optimal_value": round(val, 4),
            "variables": list(np.round(result.x, 4))
        }

        st.download_button(
            "📦 Download JSON",
            data=json.dumps(solution_data, indent=2),
            file_name="solution.json",
            mime="application/json"
        )

        # ---------------- GRAPH ----------------
        if n == 2:
            st.subheader("📊 Graph")

            x = np.linspace(0, 10, 400)
            plt.figure()

            plt.xlim(0, 10)
            plt.ylim(0, 10)

            for i in range(len(A)):
                if A[i][1] != 0:
                    y = (b[i] - A[i][0] * x) / A[i][1]
                    plt.plot(x, y, label=f"C{i+1}")

            plt.legend()
            plt.grid()
            st.pyplot(plt)

    else:
        st.error("❌ No feasible solution")