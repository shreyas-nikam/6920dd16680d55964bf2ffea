Here's a comprehensive `README.md` file for your Streamlit application lab project, "QuLab: Financial Model Stress Analyzer."

---

# QuLab: Financial Model Stress Analyzer

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

The **QuLab: Financial Model Stress Analyzer** is a Streamlit application designed for financial professionals to conduct scenario-based robustness tests on Machine Learning (ML) models. It provides a structured environment to define stress scenarios, apply them as quantitative input shocks to model features, and observe their impact on model predictions and aggregated portfolio metrics. This tool is invaluable for quantitative analysts, model developers, risk managers, and validators seeking to understand and validate the stability of ML models under adverse financial conditions.

## Table of Contents

1.  [Project Description](#project-description)
    *   [Target Audience](#target-audience)
    *   [Learning Goals](#learning-goals)
    *   [Methodology Overview](#methodology-overview)
    *   [Key Formulae](#key-formulae)
2.  [Features](#features)
3.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
4.  [Usage](#usage)
5.  [Project Structure](#project-structure)
6.  [Technology Stack](#technology-stack)
7.  [Contributing](#contributing)
8.  [License](#license)
9.  [Contact](#contact)

## 1. Project Description

This Streamlit application enables financial professionals to conduct scenario-based robustness tests on Machine Learning (ML) models. It allows users to define stress scenarios, apply them as quantitative input shocks to model features, and observe their impact on model predictions and aggregated portfolio metrics.

### Target Audience

*   Quantitative Analysts
*   Model Developers
*   Risk Managers
*   Validators

### Learning Goals

Upon completion of this application, participants will be able to:
*   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
*   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
*   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
*   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
*   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.

### Methodology Overview

This notebook employs a scenario-based stress testing methodology for Machine Learning (ML) models in finance. The core idea is to subject an ML model to structured, economically meaningful shocks applied directly to its input features, thereby simulating adverse financial environments. The model's response to these shocks is then measured using various risk metrics.

Here's a plain-English outline of the approach:

1.  **Synthetic Data Generation**: Create a realistic, yet controlled, dataset of financial features and a target variable (e.g., Probability of Default).
2.  **Mock ML Model**: Implement a placeholder ML model that provides predictions based on a fixed function of input features, bypassing the need for actual model training.
3.  **Baseline Prediction**: Obtain predictions from the mock model on the original, unstressed dataset to establish a reference point ($\hat{y} = f_\theta(x)$).
4.  **Stress Transformation Library**: Develop functions to apply quantitative shocks to input features, either multiplicatively ($x^{(s)} = x \odot (1 + \delta_s)$) or additively ($x^{(s)} = x + \delta_s$).
5.  **Scenario Parameterization**: Define concrete financial stress scenarios (e.g., "Macro Downturn") and map them to specific quantitative shocks ($\delta_s$) on input features.
6.  **Scenario Application & Stressed Predictions**: Apply defined shocks to baseline features to create stressed datasets ($x^{(s)}$), and then generate stressed predictions ($\hat{y}^{(s)} = f_\theta(x^{(s)})$) using the mock ML model.
7.  **Impact Metrics Calculation**: Quantify the difference between stressed and baseline predictions using metrics like individual prediction change ($\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}$), change in mean PD ($\Delta \overline{PD}$), and change in Expected Loss ($\Delta EL$).
8.  **Portfolio Aggregation**: Aggregate impact metrics by various portfolio segments (e.g., rating band, sector) to identify differential impacts.
9.  **Sensitivity Analysis (Alpha Path)**: Examine model response to gradually increasing shock intensity ($\alpha$) where $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$.
10. **Visualization**: Generate plots and tables to effectively communicate the results of the stress tests, including distribution shifts, sensitivity trajectories, and segmented impact heatmaps.

### Key Formulae

*   **Multiplicative Shock**: This transformation scales a feature $x$ by a factor determined by the shock $\delta_s$. This is particularly useful for modeling percentage changes in financial variables.
    $$x^{(s)} = x \odot (1 + \delta_s)$$
    Where $x^{(s)}$ is the stressed feature, $x$ is the baseline feature, and $\delta_s$ is the scenario-specific shock factor (e.g., 0.10 for a 10% increase, -0.15 for a 15% decrease).

*   **Additive Shock**: This transformation shifts a feature $x$ by a constant value $\delta_s$. This is suitable for modeling absolute changes.
    $$x^{(s)} = x + \delta_s$$
    Where $\delta_s$ is the scenario-specific additive shock value.

*   **Individual Prediction Change**: This metric quantifies the impact of stress on a single prediction by taking the difference between the stressed prediction and the baseline prediction.
    $$\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}$$
    Where $\hat{y}^{(s)}$ is the stressed prediction and $\hat{y}$ is the baseline prediction.

*   **Expected Loss (EL)**: A critical portfolio-level risk metric. For simplicity in this notebook, we define EL as the sum of `PD * exposure` across the portfolio. The change in EL ($\Delta EL^{(s)}$) is then the difference between stressed EL and baseline EL.
    $$EL = \sum (PD \times Exposure)$$
    $$\Delta EL^{(s)} = EL^{(s)} - EL_{baseline}$$
    Where $PD$ is the Probability of Default (our model output), and $Exposure$ is the financial exposure associated with each entity. This formula provides a direct financial impact measure, crucial for capital allocation and risk provisioning.

These formulae and the overall methodology provide a robust framework for assessing ML model stability and identifying potential vulnerabilities under various financial stress conditions, directly supporting prudent risk management and model validation.

## 2. Features

The QuLab: Financial Model Stress Analyzer offers the following key functionalities:

*   **Configurable Synthetic Data Generation**: Generate realistic financial datasets with adjustable sample sizes, simulating credit portfolios for robust testing.
*   **Mock ML Model Integration**: Utilize a pre-defined, interpretable mock ML model to isolate and precisely measure the impact of feature transformations without the complexities of actual model training.
*   **Baseline Prediction Calculation**: Establish a reference point for model output under normal, unstressed conditions.
*   **Pre-defined Stress Scenarios**: Apply common financial stress scenarios (e.g., "Macro Downturn," "Market Stress," "Housing Market Downturn") with pre-configured feature shocks.
*   **Custom Scenario Definition**: Create and apply your own multiplicative or additive shocks to any feature, offering unparalleled flexibility in scenario design.
*   **Adjustable Shock Intensity ($\alpha$)**: Control the severity of applied shocks, enabling sensitivity analysis and understanding of model response across a spectrum of stress levels.
*   **Automated Impact Metrics Calculation**: Instantly compute critical risk metrics like `Delta_PD` (change in Probability of Default) and `Delta_EL` (change in Expected Loss) at both individual and aggregate levels.
*   **Interactive Distribution Shifts**: Visualize changes in the distribution of PD (baseline vs. stressed) and the distribution of `Delta_PD` using Plotly charts.
*   **Sensitivity Trajectories Plot**: Observe how mean stressed PD evolves as shock intensity ($\alpha$) increases, highlighting model sensitivity.
*   **Segmented Portfolio Heatmaps**: Analyze stress impact across various categorical segments (e.g., `rating_band`, `sector`, `region`) using interactive bar charts, identifying vulnerable portfolio segments.
*   **Threshold-based Instance Flagging**: Identify individual entities where the absolute change in PD exceeds a user-defined threshold, signaling high-impact instances.
*   **Clear & Intuitive User Interface**: A Streamlit-powered interface ensures ease of navigation and interaction for users of all technical levels.

## 3. Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have the following installed:

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/quolab-financial-stress-analyzer.git
    cd quolab-financial-stress-analyzer
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    Create a `requirements.txt` file in the root directory of your project with the following content:

    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    ```

    Then install the packages:

    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage

To run the Streamlit application:

1.  **Navigate to the project's root directory** (where `app.py` is located) in your terminal.
2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser (usually at `http://localhost:8501`).

### Application Workflow:

The application is structured into three main pages, accessible via the sidebar navigation:

1.  **Data Generation & Model Setup**:
    *   Use the sidebar slider to select the number of synthetic data samples.
    *   Click "Generate Synthetic Data" to create the credit portfolio.
    *   The page displays an overview of the generated data, descriptive statistics, and the setup of the mock ML model, along with baseline PD predictions.

2.  **Scenario Definition & Stress Test**:
    *   **Select a Stress Scenario**: Choose from pre-defined scenarios or select "Custom Scenario".
    *   **Define Custom Scenario (if selected)**: Use the expander to add multiplicative or additive shocks to specific features.
    *   **Adjust Shock Intensity ($\alpha$)**: Use the slider to control the severity of the stress (0.0 for no stress, 1.0 for full stress).
    *   **Run Stress Test**: Click the button to apply the scenario, calculate stressed PDs, and derive `Delta_PD` and `Delta_EL`.

3.  **Impact Analysis & Visualizations**:
    *   This page automatically loads results from the last conducted stress test.
    *   **Review Aggregate Metrics**: See tables summarizing mean PD, total EL, and their changes.
    *   **Explore Distribution Shifts**: Visualize how PD distributions change under stress.
    *   **Analyze Sensitivity Trajectories**: Understand how mean PD responds to increasing shock intensity.
    *   **Examine Portfolio Heatmaps**: Select an aggregation segment (e.g., `rating_band`, `sector`) to view aggregated `Delta_PD` and `Delta_EL` impacts.
    *   **Identify Flagged Instances**: Set a `Delta PD` threshold to highlight individual entities most impacted by the stress.

## 5. Project Structure

The project is organized into modular Python files for clarity and maintainability:

```
.
├── app.py                      # Main Streamlit application entry point and navigation.
├── requirements.txt            # Lists Python dependencies.
├── README.md                   # Project overview and documentation.
└── application_pages/          # Directory containing individual Streamlit pages.
    ├── __init__.py             # Makes application_pages a Python package.
    ├── page1.py                # Handles synthetic data generation and mock ML model setup.
    ├── page2.py                # Manages scenario definition and stress test execution.
    └── page3.py                # Focuses on impact analysis and result visualizations.
```

## 6. Technology Stack

*   **Programming Language**: Python
*   **Web Framework**: Streamlit
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning Utilities**: Scikit-learn (for `StandardScaler` in data generation)
*   **Interactive Visualizations**: Plotly (including `plotly.express` and `plotly.graph_objects`)

## 7. Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (Note: you'd need to create a `LICENSE` file in your repository with the MIT license text).

## 9. Contact

For any questions or inquiries, please open an issue in the GitHub repository or contact:

*   **Project Maintainer**: [Your Name/QuantUniversity]
*   **Email**: [your.email@example.com / info@quantuniversity.com]
*   **Project Link**: [https://github.com/your-username/quolab-financial-stress-analyzer](https://github.com/your-username/quolab-financial-stress-analyzer) (Replace with your actual GitHub link)

---

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
