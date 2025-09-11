# PROHI Dashboard Example

# [Name of the project]

_You can modify this README file with all the information that your team consider relevant for a technical audience who would like to understand your project or to run it in the future._

_Note that this file is written in **MarkDown** language. A reference is available here: <https://www.markdownguide.org/basic-syntax/>_

Include the name, logo and images refering to your project

![Your dashboard](./assets/example-image.jpg)

## Introduction

[Project ] is an interactive web dashboard to.... 

The problem detected was...

The proposed solution is valuable because...

## System description

### Dependencies

Tested on Python 3.12.7 with the following packages:
  - Jupyter v1.1.1
  - Streamlit v1.46.1
  - Seaborn v0.13.2
  - Plotly v6.2.0
  - Scikit-Learn v1.7.0
  - shap v0.48.0

### Installation

Run the commands below in a terminal to configure the project and install the package dependencies for the first time.

If you are using Mac, you may need to follow install Xcode. Check the official Streamlit documentation [here](https://docs.streamlit.io/get-started/installation/command-line#prerequisites). 

1. Create the environment with `python -m venv env`
2. Activate the virtual environment for Python
   - `source env/bin/activate` [in Linux/Mac]
   - `.\env\Scripts\activate.bat` [in Windows command prompt]
   - `.\env\Scripts\Activate.ps1` [in Windows PowerShell]
3. Make sure that your terminal is in the environment (`env`) not in the global Python installation
4. Install required packages `pip install -r ./requirements.txt`
5. Check that everything is ok running `streamlit hello`

### Execution

To run the dashboard execute the following command:

```
> streamlit run Dashboard.py
# If the command above fails, use:
> python -m streamlit run Dashboard.py
```


### Creating pre-trained models for the web dashboadr 

⚠️ **NOTE:** In the predictive analytics tab, the web dashboard is looking for a pre-trained model in the folder `assets/`. The first time that you execute the application, it will show an error saying that such file does not exist. Therefore, you need to execute the notebook inside the folder `jupyter-notebook/` to create the pre-trained model.

This logic resembles the expected pipeline, where the jupyter notebooks are used to iterate the data modeling part until a satisfactory trained model is created, and the streamlit scripts are only in charge of rendering the user-facing interface to generate the prediction for new data. In practice, the data science pipeline is completely independent from the web dashboard, and both are connected via the pre-trained model. 

## Contributors

_Add the project's authors, contact information, and links to their websites or portfolios._


### test by Xue Wu
### the best coffee master Max! 
### Qilu Wang is useless in this task

change
