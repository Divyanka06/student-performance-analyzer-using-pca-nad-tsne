import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# STUDENT DATA
# ============================================================

data = {
    "Name": [
        "Aarav",
        "Riya",
        "Rahul",
        "Sneha",
        "Ananya",
        "Kabir",
        "Priya",
        "Aditya"
    ],

    "Python": [85, 78, 65, 90, 88, 72, 95, 70],

    "NumPy": [80, 82, 60, 85, 90, 68, 92, 75],

    "Pandas": [88, 75, 70, 92, 85, 74, 96, 78],

    "Mathematics": [75, 80, 62, 88, 91, 70, 89, 73]
}

df = pd.DataFrame(data)

subjects = [
    "Python",
    "NumPy",
    "Pandas",
    "Mathematics"
]


# ============================================================
# BASIC PERFORMANCE CALCULATIONS
# ============================================================

df["Total"] = df[subjects].sum(axis=1)

df["Average"] = np.round(
    df[subjects].mean(axis=1),
    2
)

df["Result"] = np.where(
    df["Average"] >= 40,
    "Pass",
    "Fail"
)

class_average = round(
    df["Average"].mean(),
    2
)

topper = df.loc[
    df["Average"].idxmax(),
    "Name"
]

pass_rate = round(
    (df["Result"] == "Pass").mean() * 100,
    2
)


# ============================================================
# CSS
# ============================================================

css = """
.title {
    text-align: center;
    margin-bottom: 20px;
}

.kpi {
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    background: #f5f5f5;
}

footer {
    display: none !important;
}
"""


# ============================================================
# STUDENT ANALYSIS
# ============================================================

def student_analysis(student):

    row = df[df["Name"] == student].iloc[0]

    values = row[subjects].values

    best_subject = subjects[
        np.argmax(values)
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        subjects,
        values
    )

    ax.set_title(
        f"{student} - Subject Performance"
    )

    ax.set_xlabel(
        "Subjects"
    )

    ax.set_ylabel(
        "Marks"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    information = f"""
## 👨‍🎓 Student Information

**Student:** {student}

**Total Marks:** {row["Total"]}

**Average:** {row["Average"]}

**Result:** {row["Result"]}

**Best Subject:** {best_subject}
"""

    return information, fig


# ============================================================
# SUBJECT ANALYSIS
# ============================================================

def subject_analysis():

    averages = df[subjects].mean()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        subjects,
        averages
    )

    ax.set_title(
        "Average Marks by Subject"
    )

    ax.set_xlabel(
        "Subjects"
    )

    ax.set_ylabel(
        "Average Marks"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    return fig


# ============================================================
# STUDENT AVERAGE CHART
# ============================================================

def student_average_chart():

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        df["Name"],
        df["Average"]
    )

    ax.set_title(
        "Student Average Performance"
    )

    ax.set_xlabel(
        "Students"
    )

    ax.set_ylabel(
        "Average Marks"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    return fig


# ============================================================
# HEATMAP
# ============================================================

def performance_heatmap():

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    values = df[subjects].values

    ax.imshow(
        values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(subjects))
    )

    ax.set_xticklabels(
        subjects
    )

    ax.set_yticks(
        range(len(df))
    )

    ax.set_yticklabels(
        df["Name"]
    )

    # Display marks inside the heatmap
    for i in range(len(df)):

        for j in range(len(subjects)):

            ax.text(
                j,
                i,
                values[i, j],
                ha="center",
                va="center"
            )

    ax.set_title(
        "Student Performance Heatmap"
    )

    ax.set_xlabel(
        "Subjects"
    )

    ax.set_ylabel(
        "Students"
    )

    plt.tight_layout()

    return fig


# ============================================================
# PCA
# ============================================================

def pca_visualization():

    # Select performance features
    X = df[subjects]

    # Standardize the data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # Create dataframe
    pca_df = pd.DataFrame(
        X_pca,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    pca_df["Name"] = df["Name"]

    # Create plot
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        s=120
    )

    # Add student names
    for i, name in enumerate(
        df["Name"]
    ):

        ax.annotate(
            name,
            (
                pca_df["PC1"].iloc[i],
                pca_df["PC2"].iloc[i]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_title(
        "PCA - Student Performance Visualization"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    # Explained variance
    variance_1 = (
        pca.explained_variance_ratio_[0]
        * 100
    )

    variance_2 = (
        pca.explained_variance_ratio_[1]
        * 100
    )

    total_variance = (
        variance_1 + variance_2
    )

    information = f"""
## 🔵 PCA Analysis

**Original Dimensions:** {len(subjects)}

**Reduced Dimensions:** 2

**Principal Component 1:** {variance_1:.2f}%

**Principal Component 2:** {variance_2:.2f}%

**Total Variance Represented:** {total_variance:.2f}%

### What does PCA do?

PCA transforms the original student
performance features into new variables
called Principal Components.

The first two components are used to
create a 2D visualization while retaining
as much variation in the original data
as possible.
"""

    return information, fig


# ============================================================
# t-SNE
# ============================================================

def tsne_visualization():

    # Select performance features
    X = df[subjects]

    # Standardize data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=3,
        random_state=42
    )

    X_tsne = tsne.fit_transform(
        X_scaled
    )

    # Create dataframe
    tsne_df = pd.DataFrame(
        X_tsne,
        columns=[
            "Dimension 1",
            "Dimension 2"
        ]
    )

    tsne_df["Name"] = df["Name"]

    # Create plot
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        tsne_df["Dimension 1"],
        tsne_df["Dimension 2"],
        s=120
    )

    # Add student names
    for i, name in enumerate(
        df["Name"]
    ):

        ax.annotate(
            name,
            (
                tsne_df["Dimension 1"].iloc[i],
                tsne_df["Dimension 2"].iloc[i]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_title(
        "t-SNE - Student Performance Visualization"
    )

    ax.set_xlabel(
        "t-SNE Dimension 1"
    )

    ax.set_ylabel(
        "t-SNE Dimension 2"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    information = """
## 🟣 t-SNE Analysis

**Original Dimensions:** 4

**Reduced Dimensions:** 2

### What does t-SNE do?

t-SNE is a non-linear dimensionality
reduction technique.

It tries to keep similar observations
close together in the lower-dimensional
space.

Therefore, it is useful for identifying
patterns and possible clusters among
students.

### Important

The two t-SNE axes do not have a direct
meaning such as "marks" or "average".

They are mainly used for visualization.
"""

    return information, fig


# ============================================================
# PCA VS t-SNE
# ============================================================

comparison_text = """
## ⚖️ PCA vs t-SNE

| Feature | PCA | t-SNE |
|---|---|---|
| Type | Linear | Non-linear |
| Main purpose | Reduce dimensions and preserve variance | Visualize local similarities |
| Focus | Overall variation | Local neighborhoods |
| Speed | Faster | Generally slower |
| Best use | Reduction + visualization | Visualization |
| Output | Principal Components | 2D/3D embedding |

### Easy way to remember

**PCA → Preserve variance**

**t-SNE → Preserve neighborhoods**
"""


# ============================================================
# GRADIO DASHBOARD
# ============================================================

with gr.Blocks(
    title="Student Performance Analyzer"
) as demo:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 📚 Student Performance Analyzer

        ### Analyze student performance using Python,
        ### Pandas, NumPy, PCA and t-SNE
        """
    )

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Class Overview"
    )

    with gr.Row():

        with gr.Column():
            gr.Markdown(
                f"""
                <div class="kpi">

                ### 👨‍🎓 Students

                ## {len(df)}

                </div>
                """
            )

        with gr.Column():
            gr.Markdown(
                f"""
                <div class="kpi">

                ### 📈 Class Average

                ## {class_average}

                </div>
                """
            )

        with gr.Column():
            gr.Markdown(
                f"""
                <div class="kpi">

                ### 🏆 Topper

                ## {topper}

                </div>
                """
            )

        with gr.Column():
            gr.Markdown(
                f"""
                <div class="kpi">

                ### ✅ Pass Rate

                ## {pass_rate}%

                </div>
                """
            )

    gr.Markdown("---")

    # --------------------------------------------------------
    # STUDENT ANALYSIS
    # --------------------------------------------------------

    gr.Markdown(
        "## 👨‍🎓 Student Analysis"
    )

    with gr.Row():

        student_dropdown = gr.Dropdown(
            choices=df["Name"].tolist(),
            value=df["Name"].iloc[0],
            label="Select Student"
        )

        student_button = gr.Button(
            "Analyze Student",
            variant="primary"
        )

    student_info = gr.Markdown()

    student_plot = gr.Plot(
        label="Student Performance"
    )

    student_button.click(
        fn=student_analysis,
        inputs=student_dropdown,
        outputs=[
            student_info,
            student_plot
        ]
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # CLASS PERFORMANCE
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Class Performance"
    )

    with gr.Row():

        subject_chart = gr.Plot(
            label="Subject Performance"
        )

        average_chart = gr.Plot(
            label="Student Average"
        )

    gr.Markdown("---")

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    gr.Markdown(
        "## 🔥 Performance Heatmap"
    )

    heatmap_plot = gr.Plot(
        label="Performance Heatmap"
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🧠 Dimensionality Reduction

        ### PCA and t-SNE Visualization

        The student dataset contains multiple
        performance features.

        PCA and t-SNE reduce these features into
        two dimensions for visualization.
        """
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    gr.Markdown(
        "## 🔵 Principal Component Analysis (PCA)"
    )

    pca_button = gr.Button(
        "Run PCA",
        variant="primary"
    )

    pca_info = gr.Markdown()

    pca_plot = gr.Plot(
        label="PCA 2D Visualization"
    )

    pca_button.click(
        fn=pca_visualization,
        inputs=None,
        outputs=[
            pca_info,
            pca_plot
        ]
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # t-SNE
    # --------------------------------------------------------

    gr.Markdown(
        "## 🟣 t-SNE Visualization"
    )

    tsne_button = gr.Button(
        "Run t-SNE",
        variant="primary"
    )

    tsne_info = gr.Markdown()

    tsne_plot = gr.Plot(
        label="t-SNE 2D Visualization"
    )

    tsne_button.click(
        fn=tsne_visualization,
        inputs=None,
        outputs=[
            tsne_info,
            tsne_plot
        ]
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    gr.Markdown(
        comparison_text
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # COMPLETE DATASET
    # --------------------------------------------------------

    gr.Markdown(
        "## 📋 Complete Student Dataset"
    )

    dataset_table = gr.Dataframe(
        value=df,
        interactive=False
    )

    # --------------------------------------------------------
    # LOAD INITIAL CHARTS
    # --------------------------------------------------------

    demo.load(
        fn=subject_analysis,
        inputs=None,
        outputs=subject_chart
    )

    demo.load(
        fn=student_average_chart,
        inputs=None,
        outputs=average_chart
    )

    demo.load(
        fn=performance_heatmap,
        inputs=None,
        outputs=heatmap_plot
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        css=css
    )