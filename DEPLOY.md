# Deploy to Streamlit Cloud

This document provides step-by-step instructions for deploying your Dropout Prediction application to Streamlit Community Cloud.

## Prerequisites

1. A GitHub account
2. Git installed on your computer
3. The code from this repository

## Steps for Deployment

### 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "dropout-prediction-app")
4. Choose whether to make it public or private
5. Click "Create repository"

### 2. Prepare Your Local Repository

1. Make sure your project directory contains all these files:

   - `app.py` (Streamlit application)
   - `model_dropout.pkl` (trained model)
   - `feature_names.json` (feature names)
   - `data.csv` (dataset)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (Streamlit configuration)
   - `.gitignore` (Git ignore file)

2. Open a terminal/command prompt and navigate to your project directory

3. Initialize Git repository and link it to your GitHub repository:

   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your GitHub username and repository name.

### 3. Deploy to Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with your GitHub account

2. Click on "New app" button

3. Select your repository, branch, and main file path:

   - Repository: Select the repository you created
   - Branch: main
   - Main file path: app.py

4. Click "Deploy!"

5. Wait for the deployment to complete. This may take a few minutes.

6. Once deployed, you will get a URL for your application (e.g., https://your-app-name.streamlit.app)

### 4. Update Your README.md

After successful deployment, update your README.md file with the URL to your live application:

```markdown
## Mengakses Aplikasi Online

Aplikasi ini dapat diakses secara online melalui Streamlit Community Cloud:
[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)
```

### 5. Maintaining Your Application

- Any changes pushed to the main branch of your GitHub repository will automatically trigger a redeployment
- You can manage your app settings, view logs, and more from the Streamlit Community Cloud dashboard

## Troubleshooting

If you encounter issues during deployment:

1. Check your requirements.txt file to ensure all dependencies are listed correctly
2. Verify that your app.py runs locally without errors
3. Check if your repository contains all necessary files
4. Review the deployment logs in Streamlit Community Cloud for specific error messages
