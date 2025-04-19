# House Price Prediction using Advance Regression

## Overview
This project implements a machine learning model to predict house prices based on various features. Using the Kaggle competition dataset "House Prices - Advanced Regression Techniques," we developed a model that achieves an R-squared score of 87.16%, demonstrating strong predictive performance.

## Competition Information
- **Dataset**: House Prices - Advanced Regression Techniques (Kaggle)
- **Model Performance**: 87.16% (R-squared score)

## Technologies Used
- **Python Libraries**:
  - NumPy: For numerical operations
  - Pandas: For data manipulation and analysis
  - Matplotlib & Seaborn: For data visualization
  - Scikit-learn: For machine learning algorithms and preprocessing
  - XGBoost: For gradient boosting implementation

## Methodology

### Data Exploration & Analysis
- Loaded training and test datasets from CSV files
- Performed comprehensive exploratory data analysis
- Utilized visualization techniques (histograms, box plots, heatmaps) to understand:
  - Feature distributions
  - Correlations between features
  - Missing value patterns

### Data Preprocessing
- **Missing Value Treatment**: Implemented imputation strategies for incomplete data
- **Categorical Encoding**: Applied one-hot encoding to categorical variables
- **Feature Scaling**: Standardized numerical features to improve model performance

### Model Development
- **Model Comparison**: Evaluated multiple regression algorithms:
  - Linear Regression
  - Support Vector Regression (SVR)
  - Stochastic Gradient Descent (SGD) Regressor
  - K-Nearest Neighbors Regressor
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Multi-layer Perceptron Regressor
- **Evaluation Method**: Used cross-validation with R-squared scoring
- **Model Selection**: Chose GradientBoostingRegressor based on superior performance metrics

### Prediction & Deployment
- Trained the final model on the complete training dataset
- Generated predictions for the test dataset
- Saved predictions to `submission.csv` for Kaggle submission
- Preserved the trained model as `gbr.pkl` for future use or deployment

## Future Improvements
- Experiment with more advanced feature engineering techniques
- Test ensemble methods for potential performance gains
- Implement hyperparameter tuning for model optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
For any queries, reach out at rcrathod13@gmail.com or open an issue on GitHub!
