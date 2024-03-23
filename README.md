## Car Price Prediction Project

### Overview
This project involves building a machine learning model to predict the selling price of used cars based on various features such as year of manufacture, present price, kilometers driven, fuel type, seller type, transmission, and ownership history. The dataset used for this project is `car data.csv`.

### Features
- Data preprocessing: Handling missing values, feature engineering, and one-hot encoding categorical variables.
- Exploratory Data Analysis (EDA): Visualizing relationships between features and target variable.
- Feature Importance: Identifying the most important features using an Extra Trees Regressor model.
- Model Building: Training a Random Forest Regressor model to predict car prices.
- Hyperparameter Tuning: Using Randomized Search Cross Validation to optimize the Random Forest Regressor model.
- Evaluation: Evaluating the model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- Model Deployment: Saving the trained model using pickle for future use.

### File Description
- **car data.csv**: Dataset containing car details.
- **car_price_prediction.ipynb**: Jupyter Notebook containing the Python code for data preprocessing, model building, and evaluation.
- **random_forest_regression_model.pkl**: Trained Random Forest Regressor model saved using pickle.

### Usage
1. **Dataset Preparation**: Ensure that the `car data.csv` file is in the same directory as the Jupyter Notebook.
2. **Environment Setup**: Install the required libraries mentioned in the Jupyter Notebook.
3. **Execution**: Run the cells in the Jupyter Notebook sequentially to preprocess the data, build the model, and evaluate its performance.
4. **Model Deployment**: The trained model will be saved as `random_forest_regression_model.pkl` for future use.

### Dependencies
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

### Acknowledgments
- This project is for educational purposes and is based on a tutorial or personal experimentation.
- The dataset used in this project is sourced from [source link], and credit goes to the data provider.

### Author
- SHREY GOYAL
- shreygoyal73@gmail.com

### Contact
- For any inquiries or issues regarding this project, please contact shreygoyal73@gmail.com.

### Support
- For additional support or custom implementations, please contact shreygoyal73@gmail.com.
