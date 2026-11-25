# Security Breach Detection – MCA Research Portal + Live ML Lab

This project contains two sections in one Streamlit application:

1. **Research Portal** – the supplied interactive HTML research portal, with the paper-style explanation, methodology, model explanations, diagrams, metric definitions and viva-oriented content.
2. **Live ML Experiment** – uploads UNSW-NB15 training/testing CSVs, trains six classifiers, calculates metrics from the actual run, displays confusion matrices/ROC curves and exports results.

## Run locally (Windows 11)

1. Install Python 3.11 or 3.12.
2. Extract this folder.
3. Open Command Prompt in this folder.
4. Run `python -m pip install -r requirements.txt`.
5. Run `python -m streamlit run app.py`.
6. Choose **Research Portal** or **Live ML Experiment** in the sidebar.

The supplied `run_project.bat` is also available for local launch.

## Publish online for public access (Streamlit Community Cloud)

You need a GitHub account. The deployed app will have a public URL that can be opened by anyone if you choose public access.

1. Create a new GitHub repository, e.g. `security-breach-detection-mca`.
2. Upload the contents of this project folder to the repository root:
   - `app.py`
   - `research_portal.html`
   - `requirements.txt`
   - `README_FIRST.txt`
3. Do **not** upload the UNSW CSV dataset files unless you intentionally want to store them in the public repository. The app lets visitors upload the two CSVs themselves.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**.
6. Choose your repository, branch (`main`) and main file path (`app.py`).
7. Deploy. Wait for the build to finish.
8. Open the generated `*.streamlit.app` URL and test both sections.

If you update the GitHub files later, Streamlit Community Cloud normally redeploys the app from the repository.

## Important notes before sharing

- The Research Portal currently contains **illustrative/simulation model-performance profiles** in its charts. The portal itself includes a visible data-integrity notice. Do not present those sample chart values as your measured experimental findings.
- Use **Live ML Experiment** to upload the official `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, run the six models and download the actual CSV/TXT results.
- The live experiment may take time or exceed free hosting resources at large sample sizes. Start with 5,000 training and 2,000 testing records. Increase only after a successful run.
- Uploaded datasets and generated session results are not a permanent research database. Download your results after each run.
- This is an offline labelled network-flow classification prototype. It does not capture live network packets, block attacks or provide production-grade network protection.
- The application does not perform a statistical significance test. Do not claim that the null hypothesis has been rejected.
- The project uses Decision Tree, Random Forest, LinearSVC (linear SVM), KNN, MultinomialNB and Logistic Regression.

## Dataset

Official UNSW-NB15 dataset page:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Expected files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`
