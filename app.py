import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.set_page_config(page_title="Duality Solver", layout="centered")

st.title("🔷 Smart Duality Solver")

# ---------------- INPUT ----------------
problem_type = st.selectbox("Problem Type", ["Maximization", "Minimization"])

n = st.number_input("Number of Variables", min_value=2, value=2)
m = st.number_input("Number of Constraints", min_value=1, value=2)

st.subheader("Objective Coefficients")
c = []
for i in range(n):
    c.append(st.number_input(f"c{i+1}", value=1.0))

st.subheader("Constraints")

A = []
b = []
ineq = []

for i in range(m):
    cols = st.columns(n + 2)
    row = []
    for j in range(n):
        val = cols[j].number_input(f"a{i+1}{j+1}", value=1.0, key=f"a{i}{j}")
        row.append(val)

    sign = cols[-2].selectbox("Type", ["≤", "≥", "="], key=f"s{i}")
    rhs = cols[-1].number_input(f"b{i+1}", value=10.0, key=f"b{i}")

    A.append(row)
    b.append(rhs)
    ineq.append(sign)

# ---------------- SOLVE ----------------
if st.button("Solve"):

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for i in range(m):
        if ineq[i] == "≤":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif ineq[i] == "≥":
            A_ub.append([-x for x in A[i]])
            b_ub.append(-b[i])
        else:
            A_eq.append(A[i])
            b_eq.append(b[i])

    c_mod = [-x for x in c] if problem_type == "Maximization" else c

    result = linprog(c_mod,
                     A_ub=A_ub if A_ub else None,
                     b_ub=b_ub if b_ub else None,
                     A_eq=A_eq if A_eq else None,
                     b_eq=b_eq if b_eq else None,
                     method='highs')

    # ---------------- OUTPUT ----------------
    if result.success:
        val = -result.fun if problem_type == "Maximization" else result.fun

        st.subheader("✅ Primal Solution")
        st.write("Variables:", np.round(result.x, 4))
        st.write("Optimal Value:", round(val, 4))
    else:
        st.error("❌ No feasible solution")
        st.write(result.message)

    # ---------------- DUAL ----------------
    try:
        A_np = np.array(A)

        c_dual = b
        A_dual = -A_np.T
        b_dual = [-x for x in c]

        result_dual = linprog(c_dual,
                              A_ub=A_dual,
                              b_ub=b_dual,
                              method='highs')

        if result_dual.success:
            st.subheader("🔁 Dual Solution")
            st.write("Variables:", np.round(result_dual.x, 4))
            st.write("Optimal Value:", round(result_dual.fun, 4))

            st.subheader("📌 Strong Duality")
            st.success(f"Primal = {round(val,4)} | Dual = {round(result_dual.fun,4)}")

        else:
            st.warning("Dual problem could not be solved")

    except:
        st.warning("Dual conversion not applicable")

    # ---------------- GRAPH ----------------
    if n == 2:
        x = np.linspace(0, 10, 100)
        plt.figure()

        for i in range(len(A)):
            if A[i][1] != 0:
                y = (b[i] - A[i][0]*x) / A[i][1]
                plt.plot(x, y, label=f"Constraint {i+1}")

        
        plt.legend()
        st.pyplot(plt)
