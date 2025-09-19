"""
Career Stream Analyzer — Streamlit Web App (Full Version)
File: career_stream_predictor.py
Description:
A complete working Streamlit app that accepts Class-12 marksheet data (CSV/Excel upload)
or single-student manual input, analyses marks, suggests suitable streams/careers,
includes an interest questionnaire, generates downloadable Excel + PDF career report,
and an optional demo ML predictor.

Features added in this full version:
- Improved rule-based logic for suggestions
- Interest & aptitude quick questionnaire to refine suggestions
- PDF career report generation (using reportlab)
- Clean UI layout and helpful tooltips
- Example sample data generator (for testing)

Usage:
1) pip install streamlit pandas scikit-learn matplotlib openpyxl reportlab
2) streamlit run career_stream_predictor.py

Author: Generated for Prince Kumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

import matplotlib.pyplot as plt
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Career Stream Analyzer", layout="wide")
st.title("🎓 Career Stream Analyzer — Class 12 Marksheet")
st.markdown("Upload Class-12 marksheet (CSV/XLSX) or enter marks manually. The app suggests suitable streams and creates a downloadable career report (Excel + PDF).")

# -------------------- Helper functions --------------------
COMMON_SUBJECTS = [
    'math', 'mathematics', 'physics', 'chemistry', 'biology', 'bio',
    'accounts', 'accountancy', 'business', 'business studies', 'economics', 'eco',
    'english', 'computer', 'computer science', 'cs', 'informatics'
]


def normalize_col_name(name):
    return str(name).strip().lower()


def detect_subjects(columns):
    col_map = {}
    for c in columns:
        lname = normalize_col_name(c)
        for subj in COMMON_SUBJECTS:
            if subj in lname:
                # map canonical name to actual column label
                col_map[subj] = c
                break
    return col_map


def compute_group_averages(row, col_map):
    def safe_get(key):
        col = col_map.get(key)
        if col is None:
            return np.nan
        try:
            return float(row[col])
        except:
            return np.nan

    # compute group means where applicable
    pcm_vals = []
    for k in ['math','mathematics','physics','chemistry']:
        v = safe_get(k)
        if not np.isnan(v): pcm_vals.append(v)
    pcb_vals = []
    for k in ['physics','chemistry','biology','bio']:
        v = safe_get(k)
        if not np.isnan(v): pcb_vals.append(v)
    commerce_vals = []
    for k in ['accounts','accountancy','business','business studies','economics','eco']:
        v = safe_get(k)
        if not np.isnan(v): commerce_vals.append(v)
    math_cs_vals = []
    for k in ['math','mathematics','computer','computer science','cs','informatics']:
        v = safe_get(k)
        if not np.isnan(v): math_cs_vals.append(v)

    english_val = safe_get('english')

    pcm = np.mean(pcm_vals) if pcm_vals else np.nan
    pcb = np.mean(pcb_vals) if pcb_vals else np.nan
    commerce = np.mean(commerce_vals) if commerce_vals else np.nan
    math_cs = np.mean(math_cs_vals) if math_cs_vals else np.nan

    return {'PCM_avg': pcm, 'PCB_avg': pcb, 'Commerce_avg': commerce, 'Math_CS_avg': math_cs, 'English_avg': english_val}


def rule_based_suggestion(group_avgs, interests=None):
    suggestions = []
    reason = []
    # Basic scoring
    pcm = group_avgs.get('PCM_avg', np.nan)
    pcb = group_avgs.get('PCB_avg', np.nan)
    comm = group_avgs.get('Commerce_avg', np.nan)
    mcs = group_avgs.get('Math_CS_avg', np.nan)
    eng = group_avgs.get('English_avg', np.nan)

    # Engineering / CS / Data Science
    eng_score = np.nanmean([pcm, mcs]) if not np.isnan(pcm) or not np.isnan(mcs) else np.nan
    if not np.isnan(eng_score):
        if eng_score >= 75:
            suggestions.append('Engineering / Computer Science / Data Science')
            reason.append(f'Strong PCM/Math-CS average: {eng_score:.1f}%')
        elif eng_score >= 60:
            suggestions.append('Engineering (prepare for entrance)')
            reason.append(f'Decent PCM/Math-CS average: {eng_score:.1f}%')

    # Medical
    if not np.isnan(pcb):
        if pcb >= 75:
            suggestions.append('Medical / Pharmacy / Biotech')
            reason.append(f'Strong PCB average: {pcb:.1f}%')
        elif pcb >= 60:
            suggestions.append('Life Sciences / B.Sc. (Biology)')
            reason.append(f'Decent PCB average: {pcb:.1f}%')

    # Commerce
    if not np.isnan(comm):
        if comm >= 70:
            suggestions.append('Commerce (B.Com / CA / Finance)')
            reason.append(f'Strong Commerce average: {comm:.1f}%')
        elif comm >= 55:
            suggestions.append('Commerce / BBA / Economics')
            reason.append(f'Decent Commerce average: {comm:.1f}%')

    # Arts & Humanities
    if not np.isnan(eng) and eng >= 65:
        suggestions.append('Arts / Journalism / English / Civil Services foundation')
        reason.append(f'Strong English: {eng:.1f}%')

    # Data Science specific suggestion if math+cs is strong
    if not np.isnan(mcs) and mcs >= 75:
        suggestions.append('Data Science / AI & ML (B.Sc/B.Tech in CS + specialization)')
        reason.append(f'Strong Math/CS: {mcs:.1f}%')

    # Interests influence (if provided)
    if interests:
        # interests = list of keywords
        if 'coding' in interests and 'Engineering / Computer Science / Data Science' not in suggestions:
            suggestions.insert(0, 'Engineering / Computer Science / Data Science (based on coding interest)')
            reason.append('Interest in coding')
        if 'business' in interests and 'Commerce' not in suggestions:
            suggestions.insert(0, 'Commerce / BBA / Finance (based on business interest)')
            reason.append('Interest in business/entrepreneurship')
        if 'health' in interests and 'Medical' not in suggestions:
            suggestions.insert(0, 'Medical / Pharmacy / Biotech (based on health interest)')
            reason.append('Interest in health/medicine')

    if not suggestions:
        suggestions.append('General Graduation (BA / B.Sc / B.Com) and Career Counselling / Skill Courses')
        reason.append('No strong group averages detected')

    # return unique suggestions preserving order
    uniq = []
    for s in suggestions:
        if s not in uniq:
            uniq.append(s)
    return uniq, reason


def generate_pdf_report(student_info, group_avgs, suggestions, reasons, filename='career_report.pdf'):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    x = margin
    y = height - margin

    c.setFont('Helvetica-Bold', 16)
    c.drawString(x, y, 'Career Report Card')
    y -= 12 * mm

    c.setFont('Helvetica', 11)
    c.drawString(x, y, f"Name: {student_info.get('Name','-')}")
    y -= 6 * mm
    c.drawString(x, y, f"Roll/ID: {student_info.get('Roll','-')}")
    y -= 10 * mm

    c.setFont('Helvetica-Bold', 12)
    c.drawString(x, y, 'Group Averages:')
    y -= 8 * mm
    c.setFont('Helvetica', 10)
    for k, v in group_avgs.items():
        val = f"{v:.1f}%" if not np.isnan(v) else 'N/A'
        c.drawString(x + 5 * mm, y, f"{k}: {val}")
        y -= 6 * mm

    y -= 4 * mm
    c.setFont('Helvetica-Bold', 12)
    c.drawString(x, y, 'Suggested Streams:')
    y -= 8 * mm
    c.setFont('Helvetica', 10)
    for s in suggestions:
        c.drawString(x + 5 * mm, y, f"- {s}")
        y -= 6 * mm
        if y < margin:
            c.showPage()
            y = height - margin

    y -= 4 * mm
    c.setFont('Helvetica-Bold', 12)
    c.drawString(x, y, 'Reasons / Notes:')
    y -= 8 * mm
    c.setFont('Helvetica', 10)
    for r in reasons:
        c.drawString(x + 5 * mm, y, f"* {r}")
        y -= 6 * mm
        if y < margin:
            c.showPage()
            y = height - margin

    c.setFont('Helvetica-Oblique', 8)
    c.drawString(margin, margin, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def to_excel_bytes(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet, d in df_dict.items():
            d.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()


# -------------------- UI and Logic --------------------
st.sidebar.header('Options')
use_ml = st.sidebar.checkbox('Use demo ML predictor (synthetic training)', value=False)
include_pdf = st.sidebar.checkbox('Enable PDF report generation', value=True)

uploaded = st.file_uploader('Upload Class-12 marksheet (CSV or XLSX). Header row required.', type=['csv','xlsx'])

# sample data button
if st.button('Generate sample CSV for testing'):
    sample = pd.DataFrame([
        {'Name':'Aman','Roll':'101','Math':88,'Physics':82,'Chemistry':80,'Biology':40,'Accounts':0,'Business':0,'Economics':0,'English':78,'Computer':85},
        {'Name':'Sneha','Roll':'102','Math':65,'Physics':60,'Chemistry':70,'Biology':72,'Accounts':82,'Business':80,'Economics':78,'English':80,'Computer':60},
        {'Name':'Rohit','Roll':'103','Math':45,'Physics':48,'Chemistry':50,'Biology':55,'Accounts':60,'Business':62,'Economics':60,'English':58,'Computer':40}
    ])
    csv = sample.to_csv(index=False).encode('utf-8')
    st.download_button('Download sample CSV', data=csv, file_name='sample_marks.csv', mime='text/csv')

if uploaded is not None:
    try:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f'Could not read file: {e}')
        st.stop()

    st.success(f'Loaded file: {uploaded.name} ({df.shape[0]} rows)')
    st.write('Detected columns:')
    st.write(list(df.columns))

    col_map = detect_subjects(df.columns)
    st.sidebar.markdown('**Detected subject columns (heuristic)**')
    st.sidebar.write(col_map)

    # Interest questionnaire per student (optional) - simple global interest input for demo
    st.sidebar.markdown('---')
    st.sidebar.markdown('Quick Interests (these influence suggestions)')
    interest_options = st.sidebar.multiselect('Select interests if known (coding/business/health/creative)', ['coding','business','health','creative','data'], default=[])

    results = []
    for idx, row in df.iterrows():
        name = row.get('Name') or row.get('name') or f'Student_{idx+1}'
        roll = row.get('Roll') or row.get('roll') or ''
        group_avgs = compute_group_averages(row, col_map)
        suggestions, reasons = rule_based_suggestion(group_avgs, interests=interest_options)
        res = {'Name': name, 'Roll': roll, **group_avgs, 'Suggestions': '; '.join(suggestions), 'Reasons': '; '.join(reasons)}
        results.append(res)

    res_df = pd.DataFrame(results)

    # Optional ML demo
    if use_ml:
        st.info('Training demo ML model on synthetic data (for demonstration only).')
        rng = np.random.RandomState(42)
        N = 1200
        X = rng.randint(30, 100, size=(N,5))
        y = []
        for x in X:
            pcm, pcb, comm, mcs, eng = x
            if (pcm>=75 and mcs>=70): y.append('Engineering')
            elif (pcb>=75): y.append('Medical')
            elif (comm>=70): y.append('Commerce')
            elif eng>=70: y.append('Arts')
            elif mcs>=70: y.append('DataScience')
            else: y.append('General')
        labels = list(sorted(set(y)))
        y_num = np.array([labels.index(l) for l in y])
        X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)
        model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=80, random_state=42))
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        st.success(f'Demo model accuracy (synthetic): {acc*100:.1f}%')

        feats = res_df[['PCM_avg','PCB_avg','Commerce_avg','Math_CS_avg','English_avg']].fillna(0).values
        probs = model.predict_proba(feats)
        pred = [labels[np.argmax(p)] for p in probs]
        res_df['ML_Prediction'] = pred
        res_df['ML_Top3'] = [', '.join([f"{labels[i]}:{prob:.2f}" for i,prob in sorted(enumerate(row), key=lambda x:-x[1])[:3]]) for row in probs]

    st.subheader('Analyzed Results')
    st.dataframe(res_df, height=350)

    # Download Excel
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_bytes = to_excel_bytes({'analysis': res_df})
    st.download_button('Download analysis as Excel', data=excel_bytes, file_name=f'career_analysis_{now}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Per-student PDF report generation and download
    if include_pdf:
        st.markdown('### Generate PDF report for a student')
        sel_idx = st.number_input('Enter row number (1-based) from uploaded file', min_value=1, max_value=len(res_df), value=1)
        if st.button('Generate PDF for selected student'):
            s = res_df.iloc[sel_idx-1]
            student_info = {'Name': s['Name'], 'Roll': s['Roll']}
            group_avgs = {k: s[k] for k in ['PCM_avg','PCB_avg','Commerce_avg','Math_CS_avg','English_avg']}
            suggestions = str(s['Suggestions']).split(';') if pd.notna(s['Suggestions']) else []
            reasons = str(s['Reasons']).split(';') if pd.notna(s['Reasons']) else []
            pdf_bytes = generate_pdf_report(student_info, group_avgs, suggestions, reasons)
            st.download_button('Download PDF Report', data=pdf_bytes, file_name=f"career_report_{student_info.get('Name','student')}_{now}.pdf", mime='application/pdf')

    # Visual: class-level averages
    st.subheader('Class-level Strengths (Averages)')
    avg_row = res_df[['PCM_avg','PCB_avg','Commerce_avg','Math_CS_avg','English_avg']].mean()
    fig, ax = plt.subplots()
    avg_row.plot(kind='bar', ax=ax)
    ax.set_ylabel('Average Marks')
    ax.set_ylim(0,100)
    st.pyplot(fig)

else:
    st.info('No file uploaded. Use the manual single-student form below or upload a CSV/XLSX.')

# -------------------- Manual single student form --------------------
st.markdown('---')
st.header('Manual Single-Student Prediction (Form)')
with st.form('manual_form'):
    st.write('Enter marks (0-100) for the student. Leave a subject 0 if not applicable.')
    name = st.text_input('Name')
    roll = st.text_input('Roll / ID')
    math = st.number_input('Math', min_value=0.0, max_value=100.0, value=0.0)
    physics = st.number_input('Physics', min_value=0.0, max_value=100.0, value=0.0)
    chemistry = st.number_input('Chemistry', min_value=0.0, max_value=100.0, value=0.0)
    biology = st.number_input('Biology', min_value=0.0, max_value=100.0, value=0.0)
    accounts = st.number_input('Accounts', min_value=0.0, max_value=100.0, value=0.0)
    business = st.number_input('Business Studies', min_value=0.0, max_value=100.0, value=0.0)
    economics = st.number_input('Economics', min_value=0.0, max_value=100.0, value=0.0)
    computer = st.number_input('Computer / CS', min_value=0.0, max_value=100.0, value=0.0)
    english = st.number_input('English', min_value=0.0, max_value=100.0, value=0.0)
    interest = st.multiselect('Select interests (optional)', ['coding','business','health','creative','data'])
    submitted = st.form_submit_button('Predict')

if submitted:
    row = {'Math': math, 'Physics': physics, 'Chemistry': chemistry, 'Biology': biology, 'Accounts': accounts, 'Business': business, 'Economics': economics, 'Computer': computer, 'English': english}
    col_map = {k.lower(): k for k in row.keys()}
    group_avgs = compute_group_averages(row, col_map)
    suggestions, reasons = rule_based_suggestion(group_avgs, interests=interest)

    st.subheader('Suggestions')
    for s in suggestions:
        st.markdown(f"- **{s}**")
    st.write('Group averages:')
    st.json(group_avgs)

    if include_pdf:
        pdf_bytes = generate_pdf_report({'Name': name, 'Roll': roll}, group_avgs, suggestions, reasons)
        st.download_button('Download PDF Report (single student)', data=pdf_bytes, file_name=f'career_report_{name or "student"}_{now}.pdf', mime='application/pdf')

st.markdown('---')
st.caption('Note: The ML predictor included is trained on synthetic examples for demo only. For production-grade ML, collect labelled historical data (marks -> chosen career) and perform proper training/validation.')
