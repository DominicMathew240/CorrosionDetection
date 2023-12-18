import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from data import class_names

# Function to display distribution of classes
def display_class_distribution(predictions):
    if len(predictions) == 0:
        st.warning('Note: No data to display. Please lower the confidence threshold!')
        return
    else:
        st.subheader('Class Distribution')
    
        cls_list = predictions.tolist()
        
        # Map class indices to class names
        class_labels = [class_names[i] for i in cls_list]
    
        # Create a pandas Series for class counts
        class_counts = pd.Series(class_labels).value_counts()
    
        st.bar_chart(class_counts)

# Function to display confidence distribution
def display_confidence_distribution(predictions):
    st.subheader('Confidence Distribution')
    conf_list = predictions.tolist()
    confidence_values = conf_list
    
    # Create the histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(confidence_values, bins=20)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

# Function to display a heatmap of class vs confidence
def display_confidence_heatmap(conf, number):
    st.subheader('Class vs Confidence Heatmap')

    confidences = conf.tolist()
    cls = number.tolist() 
    
    # Map class indices to class names
    class_labels = [class_names[i] for i in cls]

    # Create a DataFrame
    df_confidence_heatmap = pd.DataFrame({'class': class_labels, 'confidence': confidences})

    # Pivot the DataFrame
    df_confidence_heatmap = df_confidence_heatmap.pivot_table(index='class', values='confidence', aggfunc='mean')
    
    # Create heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df_confidence_heatmap, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Mean Confidence'}, ax=ax)
    
    # Display the plot
    st.pyplot(fig)

def display_prediction_summary(number, confidence):

    cls = number.tolist() 

    # Map class indices to class names
    class_labels = [class_names[i] for i in cls]

    # Count the number of predictions
    num_predictions = len(number)
    st.write("Number of Predictions: ", num_predictions)
    
    col1, col2 = st.columns(2)
    confidences = confidence.tolist()

    # Calculate total confidence
    total_confidence = sum(confidences)

    # Calculate average confidence
    if len(confidences) > 0:
        average_confidence = total_confidence / len(confidences)
    else:
        average_confidence = 0
    # Display the class name and average confidence
    class_name = class_names[int(number[0])]

    # Use st.metric to display the class name
    col1.metric(label="Class", value=class_name)  
    col2.metric(label="Avg Confidence", value=average_confidence)   

    col1, col2 = st.columns(2) 

    # Display the prediction in a table
    df = pd.DataFrame({'class': class_name, 'confidence': confidences})
    col1.write("Confidence Analysis Summary:")
    col1.write(df['confidence'].describe())  # Display summary statistics for confidence
    
    # Display the prediction in a bar chart
    col2.bar_chart(pd.DataFrame({'class': class_name, 'confidence': confidences}))
