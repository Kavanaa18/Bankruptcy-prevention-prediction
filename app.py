import streamlit as st
import pandas as pd
import joblib
from pickle import load
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title = None,
    options=["Home","Project"],
    icons = ["house","book"],
    default_index=0,
    orientation = "horizontal",

)


if selected == "Home":
   st.title("Bankruptcy Prevention prediction")
   
   image_path = "bankruptcy.jpeg"
   custom_html = f"""
   <div style="text-align: left;">
        <div style="margin-top: 10px;">
            <h2 style="font-size: 42px;">What is bankruptcy </h2>
            <p style="text-align:justify">Bankruptcy is a legal process designed to help individuals or businesses that are unable to meet their financial obligations. This process, which can be initiated by either the debtor or creditors, involves the evaluation and liquidation of the debtor's assets to repay outstanding debts. The primary objectives of bankruptcy are to provide a fresh start for the debtor by either discharging debts or creating a structured repayment plan, and to ensure equitable treatment for creditors.</p>
        </div>
    </div>
    """
   st.sidebar.image(image_path,use_column_width=True)
   st.sidebar.markdown(custom_html,unsafe_allow_html=True)

   st.header("Project Overview")
   st.write(""" <ul style="list-style-type: disc;text-align:justify">
            This project focuses on comprehensive data analysis and predictive modeling techniques to forecast financial stability and enhance decision-making. Key highlights include:
            <li><span style="font-weight: bold;">Exploratory Data Analysis (EDA): </span>Thorough exploration of datasets to uncover insights and patterns crucial for decision support.</li>
            <li><span style="font-weight:bold;">Visualization: </span>Creating visual representations that aid in understanding complex data relationships and trends.</li>
            <li><span style="font-weight:bold;">Model Development: </span>Building robust machine learning models tailored to specific business needs for accurate predictions.</li>
            <li><span style="font-weight:bold;">Streamlit Deployment: </span>Leveraging Streamlit to deploy interactive applications that showcase model insights and facilitate user interaction.</li>
            """, unsafe_allow_html=True)
   
   with st.expander("Model used for deployment"):
        st.write("""This project employs the Support Vector Machine (SVM) algorithm , SVM excels in classifying data points into different categories based on their features, making it suitable for predicting whether a company is likely to face financial distress.
                """)

   with st.expander("Project Objective"):
       st.write("""This is a classification project, since the variable to predict is binary (bankruptcy or non-bankruptcy). The goal here is to model the probability that a business goes bankrupt from different features.
            """)
    



elif selected == "Project":

    st.markdown("<h1 style='text-align: center; color: white; font-size: 60px;margin-top: -30px;'>Bankruptcy Prevention Prediction</h1>", unsafe_allow_html=True)
    #load the models
    try:
        svm_model = joblib.load('svm_model.pkl')
        model_accuracy = joblib.load('model_accuracy.pkl')
        cm = joblib.load('cm.pkl')


    except Exception as e:
        st.error(f"Error loading the model: {e}")



    st.sidebar.header('User Inputs')


    def user_input_features():
        industrial_risk = st.sidebar.selectbox('industrial_risk',('0','0.5','1'))
        management_risk = st.sidebar.selectbox('management_risk',('0','0.5','1'))
        financial_flexibility = st.sidebar.selectbox('financial_flexibility',('0','0.5','1'))
        credibility = st.sidebar.selectbox("credibility",('0','0.5','1'))
        competitiveness = st.sidebar.selectbox("competitiveness",('0','0.5','1'))
        operating_risk = st.sidebar.selectbox("operating_risk ",('0','0.5','1'))
        data = {'industrial_risk':industrial_risk,
                'management_risk':management_risk,
                'financial_flexibility':financial_flexibility,
                'credibility':credibility,
                'competitiveness':competitiveness,
                'operating_risk':operating_risk}
        features = pd.DataFrame(data,index = [0])
        return features 

    df = user_input_features()
    loaded_model = load(open('svm_model.pkl', 'rb'))
   

    st.markdown(
        "<h3 style='text-align: left;margin-left: 2px'>Prediction:</h3>",
        unsafe_allow_html=True
    )

    
    if st.sidebar.button('Predict'):
        try:
            probabilities = svm_model.predict_proba(df)[0]
            bankruptcy_prob = probabilities[0]
            non_bankruptcy_prob = probabilities[1]
            
            result = 'Bankruptcy' if bankruptcy_prob > non_bankruptcy_prob else 'Non-Bankruptcy'
            st.write(f"Prediction: {result}")
            st.write(f"Probability of Bankruptcy: {bankruptcy_prob * 100:.2f}%")
            st.write(f"Probability of Non-Bankruptcy: {non_bankruptcy_prob * 100:.2f}%")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    st.markdown(
        "<h3 style='text-align: left;margin-left: 2px'>Model Accuracy</h3>",
        unsafe_allow_html=True
    )

    st.write(f"Overall model accuracy is : {model_accuracy * 100:.2f}%")

    st.title('Confusion Matrix')
    st.table(cm)









