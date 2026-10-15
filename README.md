Project Description
Customer Loan Default Prediction System is a machine learning project developed to predict whether a customer is likely to default on a loan. The system analyzes customer-related financial and demographic information to identify potential default risks and support better lending decisions.
The project uses a Random Forest classification model trained on historical customer loan data. It includes data preprocessing, model training, evaluation, and individual customer predictions. An interactive Streamlit application combines a dashboard with a prediction interface, making it easier to understand loan default patterns and assess customer risk.
Project Objectives
- Predict the likelihood of loan default using machine learning.
- Analyze customer data to identify patterns associated with loan defaults.
- Visualize customer default patterns and compare default rates across employment types.
- Display individual customer predictions with risk classification and default probability.
- Provide an interactive dashboard for understanding model results.
Technologies Used
Python, Pandas, NumPy, Scikit-learn, Random Forest, Streamlit, Matplotlib, Docker.
Model Performance
The Random Forest model achieved approximately 85% accuracy on the test dataset. It also generates individual customer predictions and estimated default probabilities. Additional evaluation metrics are useful for understanding how effectively the model identifies customers who actually default on their loans.
Dashboard Features
- Displays the total number of customer records and loan defaults in the dataset.
- Visualizes default and non-default distributions using a pie chart.
- Compares default rates across employment types using a bar chart.
- Provides an individual customer prediction interface.
- Displays the predicted risk classification and default probability.
Project Structure
The repository contains the Python source code, trained model artifacts, dependency information, and application components required to run the project. The Streamlit application combines dashboard visualizations and individual customer prediction functionality in a single interface.
How to Run
1. Clone or download the repository.
2. Install the required Python packages listed in the requirements file.
3. Ensure that the dataset and trained model files are available at the locations expected by the application.
4. Open the project folder in the terminal and run the Streamlit application using the command below.
streamlit run app.py
Conclusion
This project demonstrates the application of machine learning to loan default prediction and customer risk analysis. By combining a Random Forest classification model with an interactive Streamlit dashboard, the system presents customer risk predictions and dataset patterns in an understandable format. The project also provides practical experience in data preprocessing, classification, model evaluation, data visualization, and application deployment.
