import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

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

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
    }

    /* Labels */
    label, .stNumberInput label {
        color: white !important;
        font-weight: 500;
    }

    /* Inputs */
    input, select {
        background-color: #1e293b !important;
        color: white !important;
        border-radius: 6px !important;
    }

    /* Buttons */
    .stButton>button {
        background: #3b82f6;
        color: white;
        border-radius: 8px;
    }

    input::placeholder {
        color: #cbd5f5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fb;
        color: #1e293b;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 8px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🚀 Optimization Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Linear Programming + Duality</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    problem_type = st.selectbox("Problem Type", ["Maximization", "Minimization"])
    n = st.number_input("Variables", min_value=2, value=2)
    m = st.number_input("Constraints", min_value=1, value=2)

# ---------------- EXAMPLE ----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    example_btn = st.button("📌 Load Example")

# ---------------- INPUT ----------------
if example_btn:
    c = [3,2]
    A = [[2,1],[1,3]]
    b = [10,15]
    ineq = ["≤","≤"]
else:
    st.subheader("📥 Input")

    c = [st.number_input(f"c{i+1}", value=1.0) for i in range(n)]

    A, b, ineq = [], [], []

    for i in range(m):
        st.markdown(f"**Constraint {i+1}**")
        cols = st.columns(n+2)

        row = []
        for j in range(n):
            val = cols[j].number_input(f"a{i+1}{j+1}", value=1.0, key=f"a{i}{j}")
            row.append(val)

        sign = cols[-2].selectbox("Type", ["≤","≥","="], key=f"s{i}")
        rhs = cols[-1].number_input(f"b{i+1}", value=10.0, key=f"b{i}")

        A.append(row)
        b.append(rhs)
        ineq.append(sign)

# ---------------- SOLVE ----------------
solve = st.button("🚀 Solve Problem")

if solve:

    A_ub, b_ub = [], []
    for i in range(m):
        if ineq[i] == "≤":
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif ineq[i] == "≥":
            A_ub.append([-x for x in A[i]])
            b_ub.append(-b[i])

    c_mod = [-x for x in c] if problem_type=="Maximization" else c

    result = linprog(c_mod, A_ub=A_ub, b_ub=b_ub, method="highs")

    if result.success:
        val = -result.fun if problem_type=="Maximization" else result.fun

        st.subheader("✅ Solution")
        st.metric("Optimal Value", round(val,4))

        for i,v in enumerate(result.x):
            st.write(f"x{i+1} = {round(v,4)}")
        #Update
        # ---------------- DUAL ----------------
        try:
            A_np = np.array(A)

            c_dual = b
            A_dual = -A_np.T
            b_dual = [-x for x in c]

            dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, method="highs")

            if dual.success:
                st.subheader("🔁 Dual Solution")
                st.write("Variables:", np.round(dual.x,4))
                st.write("Optimal Value:", round(dual.fun,4))
                st.success(f"Strong Duality: {round(val,4)} = {round(dual.fun,4)}")
        except:
            st.warning("Dual not available")

        # ---------------- AI ----------------
        st.subheader("🤖 Explanation")
        st.info(f"Optimal solution is {np.round(result.x,2)} giving value {round(val,2)}.")

        # ---------------- STEPS ----------------
        st.subheader("🧠 Steps")
        st.markdown("""
        1. Convert to standard form  
        2. Handle inequalities  
        3. Apply optimization  
        4. Compute solution  
        """)
       #update
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

        with open("result.pdf","rb") as f:
            st.download_button("📄 Download PDF", f, file_name="solution.pdf", mime="application/pdf")
        #update
        # ---------------- GRAPH ----------------
        if n==2:
            st.subheader("📊 Graph")

            x = np.linspace(0,10,400)
            plt.figure()

            for i in range(len(A)):
                if A[i][1]!=0:
                    y = (b[i] - A[i][0]*x)/A[i][1]
                    plt.plot(x,y)

            plt.grid()
            st.pyplot(plt)

    else:
        st.error("❌ No feasible solution")
