# hotel_booking_cancellationprediction_ml
Hotel Booking Cancellation Prediction using Machine Learning and XGBoost with an interactive Streamlit web application.
# Hotel Booking Cancellation Prediction

This project predicts whether a hotel booking is likely to be cancelled using Machine Learning techniques. The model is trained on historical hotel booking data and deployed through an interactive Streamlit web application.

## Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Multiple Machine Learning models comparison
- XGBoost Classifier as the final prediction model
- Interactive Streamlit GUI
- Real-time booking cancellation prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Dataset
The project uses the Hotel Booking Demand dataset containing booking information from resort and city hotels.

## Model Performance
Several machine learning algorithms were evaluated, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM
- Voting Classifier

XGBoost was selected as the final model based on its predictive performance.

## Application
The Streamlit application allows users to enter booking details such as:
- Lead Time
- Weekend Nights
- Week Nights
- Adults
- Children
- Babies
- Previous Cancellations
- Average Daily Rate (ADR)

and instantly predicts whether the booking is likely to be cancelled or confirmed.

## Future Enhancements
- Cloud deployment
- Advanced feature engineering
- Real-time hotel management integration
- Enhanced dashboard and analytics
