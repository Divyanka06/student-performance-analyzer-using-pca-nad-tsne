import gradio as gr
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
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

    "Python": [
        85, 78, 65, 90,
        88, 72, 95, 70
    ],

    "NumPy": [
        80, 82, 60, 85,
        90, 68, 92, 75
    ],

    "Pandas": [
        88, 75, 70, 92,
        85, 74, 96, 78
    ],

    "Mathematics": [
        75, 80, 62, 88,
        91, 70, 89, 73
    ]
}


# Create DataFrame
df = pd.DataFrame(data)

subjects = [
    "Python",
    "NumPy",
    "Pandas",
    "Mathematics"
]


# ============================================================
# CALCULATIONS
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
# STUDENT PERFORMANCE CHART
# ============================================================

def student_performance():

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = np.arange(len(df))

    width = 0.2

    for i, subject in enumerate(subjects):

        ax.bar(
            x + i * width,
            df[subject],
            width,
            label=subject
        )

    ax.set_title(
        "Student Performance by Subject"
    )

    ax.set_xlabel(
        "Students"
    )

    ax.set_ylabel(
        "Marks"
    )

    ax.set_xticks(
        x + width * 1.5
    )

    ax.set_xticklabels(
        df["Name"]
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    return fig


# ============================================================
# SUBJECT AVERAGE CHART
# ============================================================

def subject_average():

    averages = df[subjects].mean()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        averages.index,
        averages.values
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

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    return fig


# ============================================================
# STUDENT AVERAGE CHART
# ============================================================

def student_average():

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

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    return fig


# ============================================================
# PERFORMANCE HEATMAP
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

    # Display marks inside cells
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
# PCA VISUALIZATION
# ============================================================

def pca_visualization():

    # Select subject columns
    X = df[subjects]

    # Standardize data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # Apply PCA
    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # Create DataFrame
    pca_df = pd.DataFrame(
        X_pca,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    pca_df["Name"] = df["Name"]

    # Create graph
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
            textcoords="offset points",
            fontsize=9
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

    return fig


# ============================================================
# t-SNE VISUALIZATION
# ============================================================

def tsne_visualization():

    # Select subject columns
    X = df[subjects]

    # Standardize data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=3,
        random_state=42
    )

    X_tsne = tsne.fit_transform(
        X_scaled
    )

    # Create DataFrame
    tsne_df = pd.DataFrame(
        X_tsne,
        columns=[
            "Dimension 1",
            "Dimension 2"
        ]
    )

    tsne_df["Name"] = df["Name"]

    # Create graph
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
            textcoords="offset points",
            fontsize=9
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

    return fig


# ============================================================
# GRADIO INTERFACE
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
        Pandas, NumPy, Matplotlib, PCA and t-SNE.
        """
    )


    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    gr.Markdown(
        "## 📋 Student Performance Data"
    )

    student_table = gr.Dataframe(
        value=df,
        label="Student Performance",
        interactive=False
    )


    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Performance Summary"
    )

    with gr.Row():

        class_avg_box = gr.Textbox(
            value=str(class_average),
            label="Class Average"
        )

        topper_box = gr.Textbox(
            value=topper,
            label="Topper"
        )

        pass_rate_box = gr.Textbox(
            value=f"{pass_rate}%",
            label="Pass Rate"
        )


    # --------------------------------------------------------
    # PERFORMANCE CHARTS
    # --------------------------------------------------------

    gr.Markdown(
        "---"
    )

    gr.Markdown(
        "## 📈 Performance Charts"
    )

    with gr.Row():

        performance_chart = gr.Plot(
            value=student_performance(),
            label="Student Performance"
        )

        subject_chart = gr.Plot(
            value=subject_average(),
            label="Subject Average"
        )


    with gr.Row():

        average_chart = gr.Plot(
            value=student_average(),
            label="Student Average"
        )

        heatmap_chart = gr.Plot(
            value=performance_heatmap(),
            label="Performance Heatmap"
        )


    # --------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # --------------------------------------------------------

    gr.Markdown(
        "---"
    )

    gr.Markdown(
        """
        # 🧠 Dimensionality Reduction

        ### PCA and t-SNE Visualization

        The student performance data is reduced
        to two dimensions for visualization.
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

    pca_plot = gr.Plot(
        label="PCA 2D Visualization"
    )

    pca_button.click(
        fn=pca_visualization,
        inputs=None,
        outputs=pca_plot
    )


    # --------------------------------------------------------
    # t-SNE
    # --------------------------------------------------------

    gr.Markdown(
        "---"
    )

    gr.Markdown(
        "## 🟣 t-SNE Visualization"
    )

    tsne_button = gr.Button(
        "Run t-SNE",
        variant="primary"
    )

    tsne_plot = gr.Plot(
        label="t-SNE 2D Visualization"
    )

    tsne_button.click(
        fn=tsne_visualization,
        inputs=None,
        outputs=tsne_plot
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch()