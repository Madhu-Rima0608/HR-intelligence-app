import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io

# -----------------------------
# SESSION STATE (LOGIN SYSTEM)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "password" not in st.session_state:
    st.session_state.password = "123456"

if "username" not in st.session_state:
    st.session_state.username = ""

if "show_change" not in st.session_state:
    st.session_state.show_change = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="HR Intelligence App", layout="wide")
st.title("💼 HR Intelligence & Attrition Prediction System")

# -----------------------------
# LOGIN PAGE (FIRST SCREEN)
# -----------------------------
if not st.session_state.logged_in:

    st.title("🔐 Welcome to the HR Intelligence App!!")
    st.markdown("### Please login to use the app for taking valuable business decisions.")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            if password == st.session_state.password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username} 👋")
                st.rerun()
            else:
                st.error("❌ Incorrect Password")

    with col2:
        if st.button("Change Password"):
            st.session_state.show_change = True

    if st.session_state.show_change:
        st.markdown("### 🔄 Change Password")

        old_pass = st.text_input("Old Password", type="password")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Update Password"):
            if old_pass == st.session_state.password:
                st.session_state.password = new_pass
                st.success("✅ Password updated!")
                st.session_state.show_change = False
            else:
                st.error("❌ Wrong old password")

    st.stop()   # 🔴 THIS STOPS THE APP HERE UNTIL LOGIN

st.sidebar.title("👩‍💼 HR Intelligence App")

page = st.sidebar.radio("Navigation", [
    "Attrition Analysis Dashboard",
    "Departmental Performance",
    "Departmental Productivity",
    "Trend Analysis",
    "Advanced Insights",
    "Attrition Predictor"
])

st.sidebar.success(f"👋 Logged in as: {st.session_state.username}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
uploaded_file = st.sidebar.file_uploader("Upload CSV")

if not uploaded_file:
    st.warning("⬅️ Please Upload Dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# -----------------------------
# COLUMN FINDER
# -----------------------------
def find_col(possible):
    for col in df.columns:
        for name in possible:
            if col.lower() == name.lower():
                return col
    for col in df.columns:
        for name in possible:
            if name.lower() in col.lower():
                return col
    return None

col_map = {
    "Age": find_col(["age"]),
    "MonthlyIncome": find_col(["income","salary"]),
    "YearsAtCompany": find_col(["yearsatcompany","tenure"]),
    "YearsInCurrentRole": find_col(["yearsincurrentrole","currentrole"]),
    "YearsSinceLastPromotion": find_col(["yearssincelastpromotion","promotion"]),
    "DistanceFromHome": find_col(["distance"]),
    "Gender": find_col(["gender"]),
    "PerformanceRating": find_col(["performance"]),
    "JobSatisfaction": find_col(["satisfaction"]),
    "WorkLifeBalance": find_col(["worklife"]),
    "OverTime": find_col(["overtime"]),
    "Attrition": find_col(["attrition","left"]),
    "Department": find_col(["department"]),
    "JobRole": find_col(["role"]),
    "Date": find_col(["date","hire"])
}

required = ["Age","MonthlyIncome","YearsAtCompany",
            "PerformanceRating","JobSatisfaction",
            "WorkLifeBalance","OverTime","Attrition"]

missing = [k for k in required if col_map[k] is None]

if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# Rename safely
reverse_map = {}
for k,v in col_map.items():
    if v and v not in reverse_map:
        reverse_map[v] = k

df = df.rename(columns=reverse_map)
df = df.loc[:, ~df.columns.duplicated()]

# -----------------------------
# PREPROCESSING
# -----------------------------
df['Attrition'] = df['Attrition'].astype(str).str.lower().map({'yes':1,'no':0})

# -----------------------------
# BUSINESS BIN FUNCTION
# -----------------------------
def create_bins(series, bins, labels):
    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    result = df.groupby(binned)['Attrition'].mean().reset_index()
    result.columns = ['Category', 'AttritionRate']
    result['AttritionRate'] = result['AttritionRate'] * 100
    return result

def plot_bar_with_labels(df_chart, title):

    df_chart['AttritionLabel'] = df_chart['AttritionRate'].round(1).astype(str) + '%'

    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Category:N', title=title, sort='-y'),
        y=alt.Y('AttritionRate:Q', title='Attrition (%)',
        scale=alt.Scale(domain=[0, df_chart['AttritionRate'].max()*1.2])
        ),
        color=alt.Color('Category:N', legend=None)
    )

    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-2,
        fontSize=14,
        fontWeight='bold'
    ).encode(
        text='AttritionLabel:N'
    )

    st.altair_chart(chart + text, use_container_width=True)

def plot_normal_bar(df_chart, x_col, y_col, title, value_format=".0f"):
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X(f'{x_col}:N', title=title),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{x_col}:N', legend=None)
    )

    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-2,
        fontSize=14,
        fontWeight='bold'
    ).encode(
        text=alt.Text(f'{y_col}:Q', format=value_format)
    )

    st.altair_chart(chart + text, use_container_width=True)

# -----------------------------
# EXCEL EXPORT
# -----------------------------
def generate_excel():
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        create_bins(df['Age'], [18,25,35,45,60],
                    ["18–25","26–35","36–45","46–60"])\
            .to_excel(writer, sheet_name="Age", index=False)
        
        if "DistanceFromHome" in df.columns:
            create_bins(df['DistanceFromHome'],
                             [0,5,10,20,50],
                             ["0–5 km","6–10 km","11–20 km","20+ km"])\
                .to_excel(writer, sheet_name="DistanceFromHome", index=False)
            
        create_bins(df['MonthlyIncome'],
                    [0,3000,7000,15000,50000],
                    ["Low (<3K)","Medium (3K–7K)","High (7K–15K)","Very High (15K+)"])\
            .to_excel(writer, sheet_name="MonthlyIncome", index=False)
        
        create_bins(df['YearsAtCompany'],
                    [0,2,5,10,40],
                    ["0–2 yrs","3–5 yrs","6–10 yrs","10+ yrs"])\
            .to_excel(writer, sheet_name="Length Of Service", index=False)
        
        if "YearsInCurrentRole" in df.columns:
            create_bins(df['YearsInCurrentRole'],
                             [0,2,5,10,40],
                             ["0–2 yrs","3–5 yrs","6–10 yrs","10+ yrs"])\
            .to_excel(writer, sheet_name="YearsInCurrentRole", index=False)

        if "YearsSinceLastPromotion" in df.columns:
            create_bins(df['YearsSinceLastPromotion'],
                              [0,1,3,6,15],
                              ["0–1 yr","2–3 yrs","4–6 yrs","6+ yrs"])\
            .to_excel(writer, sheet_name="YearsSinceLastPromotion", index=False)

        if "Department" in df.columns:
            df.groupby('Department')['Attrition'].mean()\
              .reset_index().to_excel(writer, sheet_name="Department", index=False)

        if "JobRole" in df.columns:
            df.groupby('JobRole')['Attrition'].mean()\
              .reset_index().to_excel(writer, sheet_name="JobRole", index=False)

    return output.getvalue()

def generate_top_insights(df):

    insights = []

    # 1. Highest Attrition Driver (Department / Role)
    if "Department" in df.columns:
        dept_attr = df.groupby('Department')['Attrition'].mean()
        worst_dept = dept_attr.idxmax()
        worst_val = dept_attr.max() * 100
        insights.append(f"🚨 Highest attrition is in '{worst_dept}' ({worst_val:.1f}%) → Immediate HR intervention required.")

    # 2. Salary Risk Insight
    salary_bins = pd.cut(df['MonthlyIncome'],
                         bins=[0,3000,7000,20000],
                         labels=["Low","Medium","High"])

    salary_attr = df.groupby(salary_bins)['Attrition'].mean()

    if not salary_attr.isna().all():
        risk_group = salary_attr.idxmax()
        risk_val = salary_attr.max() * 100
        insights.append(f"💰 Employees in '{risk_group}' salary group show highest attrition ({risk_val:.1f}%) → Compensation strategy needed.")

    # 3. Performance Insight
    perf_attr = df.groupby('PerformanceRating')['Attrition'].mean()

    if not perf_attr.isna().all():
        low_perf = perf_attr.idxmax()
        perf_val = perf_attr.max() * 100
        insights.append(f"📉 Performance rating '{low_perf}' has highest attrition ({perf_val:.1f}%) → Training & engagement required.")

    return insights[:3]

# -----------------------------
# ATTRITION ANALYSIS DASHBOARD
# -----------------------------
if page == "Attrition Analysis Dashboard":
    st.header("📊 HR Attrition Analysis Dashboard")

    attr_rate = df['Attrition'].mean()

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Employees", len(df))
    col2.metric("Attrition %", f"{attr_rate*100:.2f}%")
    col3.metric("Avg Salary", f"{df['MonthlyIncome'].mean():.0f}")
    col4.metric("Performance", f"{df['PerformanceRating'].mean():.2f}")

    st.download_button(
        "📥 Download Excel Report",
        generate_excel(),
        "HR_Report.xlsx"
    )

    st.markdown("---")
    st.subheader("🧠 Top 3 Key HR Insights")

    insights = generate_top_insights(df)

    for insight in insights:
        st.info(insight)

    st.markdown("---")
    st.subheader("📊 Attrition Drivers (Business View)")

    # AGE
    age_df = create_bins(df['Age'],
                         [18,25,35,45,60],
                         ["18–25","26–35","36–45","46–60"])
    st.write("👶 Age-wise Attrition")
    plot_bar_with_labels(age_df, "Age Group")

    # DISTANCE
    if "DistanceFromHome" in df.columns:
        dist_df = create_bins(df['DistanceFromHome'],
                             [0,5,10,20,50],
                             ["0–5 km","6–10 km","11–20 km","20+ km"])
        st.write("🏠 Distance-wise Attrition")
        plot_bar_with_labels(dist_df, "Distance")

    # GENDER
    if "Gender" in df.columns:
        gender_df = df.groupby('Gender')['Attrition'].mean().reset_index()
        gender_df.columns = ['Category','AttritionRate']
        st.write("👨‍👩‍👧 Gender-wise Attrition")
        plot_bar_with_labels(gender_df, "Gender")

    # SALARY
    salary_df = create_bins(df['MonthlyIncome'],
                           [0,3000,7000,15000,50000],
                           ["Low (<3K)","Medium (3K–7K)","High (7K–15K)","Very High (15K+)"])
    st.write("💰 Salary-wise Attrition")
    plot_bar_with_labels(salary_df, "Salary")

    # TENURE
    tenure_df = create_bins(df['YearsAtCompany'],
                           [0,2,5,10,40],
                           ["0–2 yrs","3–5 yrs","6–10 yrs","10+ yrs"])
    st.write("🏢 Tenure-wise Attrition")
    plot_bar_with_labels(tenure_df, "Tenure")

    # ROLE TENURE
    if "YearsInCurrentRole" in df.columns:
        role_df = create_bins(df['YearsInCurrentRole'],
                             [0,2,5,10,40],
                             ["0–2 yrs","3–5 yrs","6–10 yrs","10+ yrs"])
        st.write("🧑‍💼 Role Tenure Attrition")
        plot_bar_with_labels(role_df, "Role Tenure")

    # PROMOTION GAP
    if "YearsSinceLastPromotion" in df.columns:
        promo_df = create_bins(df['YearsSinceLastPromotion'],
                              [0,1,3,6,15],
                              ["0–1 yr","2–3 yrs","4–6 yrs","6+ yrs"])
        st.write("📈 Promotion Gap Attrition")
        plot_bar_with_labels(promo_df, "Promotion Gap")

    # DEPARTMENT
    if "Department" in df.columns:
        dept_attr = df.groupby('Department')['Attrition'].mean().reset_index()
        dept_attr.columns = ['Category', 'AttritionRate']
        dept_attr['AttritionRate'] = dept_attr['AttritionRate'] * 100
        st.write("Department-wise Attrition")
        plot_bar_with_labels(dept_attr, "Department")

    # JOB ROLE
    if "JobRole" in df.columns:
        role_attr = df.groupby('JobRole')['Attrition'].mean().reset_index()
        role_attr.columns = ['Category', 'AttritionRate']
        role_attr['AttritionRate'] = role_attr['AttritionRate'] * 100
        st.write("JobRole-wise Attrition")
        plot_bar_with_labels(role_attr, "Job Role")

    # PERFORMANCE RATING ATTRITION
    st.write("⭐ Performance Rating vs Attrition")

    perf_attr_df = df.groupby('PerformanceRating')['Attrition'].mean().reset_index()
    perf_attr_df.columns = ['Category', 'AttritionRate']
    perf_attr_df['AttritionRate'] = perf_attr_df['AttritionRate'] * 100

    plot_bar_with_labels(perf_attr_df, "Performance Rating")

    # Insight
    highest_perf_risk = perf_attr_df.loc[perf_attr_df['AttritionRate'].idxmax(), 'Category']
    st.warning(f"⚠️ Highest attrition observed at Performance Rating: {highest_perf_risk}")

    st.info("💡 Insight: Lower performance employees tend to leave more → Training & engagement needed.")

    avg_attr = perf_attr_df['AttritionRate'].mean()

    # Find high-risk groups
    high_risk_groups = perf_attr_df[perf_attr_df['AttritionRate'] > avg_attr]

    if not high_risk_groups.empty:
        st.error("🚨 Performance imbalance detected")

        st.write("👉 High-risk performance groups:")

        for _, row in high_risk_groups.iterrows():
            st.write(
                f"• Performance Rating {row['Category']} → {row['AttritionRate']:.2f}% attrition"
            )

    else:
        st.success("✅ No major performance-based attrition risk detected")

    if not high_risk_groups.empty:
        st.info("💡 Insight: These performance groups are leaving more than average → focus on training, engagement & performance review policies.")
# -----------------------------
# PERFORMANCE
# -----------------------------
elif page == "Departmental Performance":
    st.header("🎯 Performance Analysis")

    if "Department" in df.columns:
        dept = st.selectbox("Department", df['Department'].unique())
        filtered = df[df['Department']==dept]
    else:
        filtered = df

    st.bar_chart(filtered['PerformanceRating'].value_counts())
    st.metric("Avg Performance", f"{filtered['PerformanceRating'].mean():.2f}")

    if "JobRole" in df.columns:
        role_perf = filtered.groupby('JobRole')['PerformanceRating'].mean().reset_index()
        plot_normal_bar(role_perf, "JobRole", "PerformanceRating", "JobRole", ".2f")

# -----------------------------
# PRODUCTIVITY
# -----------------------------
elif page == "Departmental Productivity":
    st.header("🏢 Productivity")

    if "Department" in df.columns:
        prod = df.groupby('Department')['MonthlyIncome'].mean().reset_index()
        plot_normal_bar(prod, "Department", "MonthlyIncome", "Department", ".0f")

# -----------------------------
# TREND
# -----------------------------
elif page == "Trend Analysis":
    st.header("📈 Trend Analysis")

    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        trend = df.groupby('Month')['Attrition'].mean()
        st.line_chart(trend)

# -----------------------------
# ADVANCED INSIGHTS
# -----------------------------
elif page == "Advanced Insights":
    st.header("🧠 Advanced Insights")

    # Attrition vs Performance
    perf_df = df.groupby('PerformanceRating')['Attrition'].mean().reset_index()
    perf_df['Attrition'] *= 100

    st.subheader("📊 Attrition vs Performance")
    chart = alt.Chart(perf_df).mark_bar(color="orange").encode(
        x='PerformanceRating:N',
        y='Attrition:Q',
        tooltip=['PerformanceRating','Attrition']
    )
    st.altair_chart(chart, use_container_width=True)

    st.info("💡 Insight: Lower performance employees tend to have higher attrition risk.")

    # Risk Segments
    df['RiskGroup'] = pd.cut(df['MonthlyIncome'],
                            bins=[0,3000,7000,20000],
                            labels=["Low","Medium","High"])

    risk_df = df.groupby('RiskGroup')['Attrition'].mean().reset_index()
    risk_df['Attrition'] *= 100

    st.subheader("💰 Salary Risk Segments")

    chart2 = alt.Chart(risk_df).mark_bar(color="red").encode(
        x='RiskGroup:N',
        y='Attrition:Q',
        tooltip=['RiskGroup','Attrition']
    )
    st.altair_chart(chart2, use_container_width=True)

    st.info("💡 Insight: Lower salary groups show higher attrition → compensation strategy needed.")

# -----------------------------
# PREDICTOR
# -----------------------------
elif page == "Attrition Predictor":
    st.header("⚠️ Attrition Predictor")

    num = ['Age','MonthlyIncome','YearsAtCompany',
           'PerformanceRating','JobSatisfaction','WorkLifeBalance']
    cat = ['OverTime']

    X = df[num+cat]
    y = df['Attrition']

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5))
    ])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)

    st.metric("Accuracy", f"{accuracy_score(y_test,model.predict(X_test)):.2f}")

    age = st.slider("Age",18,60)
    income = st.number_input("Income",3000)
    years = st.slider("Years",0,40)
    perf = st.slider("Performance",1,4)
    overtime = st.selectbox("OverTime",["Yes","No"])
    js = st.slider("Satisfaction",1,4)
    wl = st.slider("WorkLifeBalance",1,4)

    if st.button("🔍 Predict Attrition Risk"):
        input_df = pd.DataFrame({
            'Age':[age],
            'MonthlyIncome':[income],
            'YearsAtCompany':[years],
            'PerformanceRating':[perf],
            'JobSatisfaction':[js],
            'WorkLifeBalance':[wl],
            'OverTime':[overtime]
        })

        prob = model.predict_proba(input_df)[0][1]
        st.subheader(f"Risk Score: {prob*100:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built by Madhurima Roy | Data Analyst")