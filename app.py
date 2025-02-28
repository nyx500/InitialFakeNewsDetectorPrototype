# Import required libraries
import streamlit as st # Streamlit app functionality
# Data processing libraries
import pandas as pd
import numpy as np
import re
# Model-loading library
import joblib
# LIME-explainability library
import lime.lime_text
# For visualizations
import matplotlib.pyplot as plt

# Load the WELFake-trained Passive-Aggressive classifier model using st.cache_resource
# Reference: "Decorator to cache functions that return global resources (e.g. database connections, ML models).""
# https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
@st.cache_resource
def loadModel():
    return joblib.load("models/my_model2.joblib")


def generateLIMEExplanation(pipeline, text, num_features=30):
    """
        Generates LIME-based importance scores for individual word features for a single news text. Displays a chart showing the top important features
        pushing the classifier towards a specific prediction.

        Input Parameters:
            pipeline (sklearn.pipeline.Pipeline): a pre-trained Pipeline containing TfidfVectorizer, PassiveAggressiveClassifier and
                CalibratedClassifierCV (to convert raw PassiveAggressiveClassifier scores into probabilities for LIME)
            text (str): the preprocessed news sample to analyze using the LIME explainability perturbation algorithm
            num_features (int): the maximum number of LIME top important features to extract and show to users

        Output:
            fig (matplotlib.pyplot.figure): the feature importance plot to show in the application
            top_features (list): a list of tuples containing (word, float representing LIME score)
    """

    # Creates the LIME explainer and maps the 0 label to "real" and 1 to "fake"
    explainer = lime.lime_text.LimeTextExplainer(class_names=["real", "fake"])

    # Explains the text sample using the LIME explainer instance: perturbs the inputted, preprocessed text and calculates importance based on
    # effect of changing random words on probability score
    explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=num_features)

    # Store the extracted most important features as a list of (word, importance) tuples
    top_features = explanation.as_list()

    # Create a pandas DataFrame with the word features and their corresponding LIME scores in two columns
    df = pd.DataFrame(top_features, columns=["Word", "Importance Score"])

    # Sort the DataFrame by the absolute value ("Absolute Score") of the importance scores: first create a new column storing these absolute scores
    df["Absolute Score"] = df["Importance Score"].apply(lambda x: abs(x))
    # Use the pandas .sort_values function to sort the DataFrame's rows in descending (ascending=False) order using this new absolute score column
    df_sorted = df.sort_values(by="Absolute Score", ascending=False)

    # Now extract the sorted word features and their scores for plotting
    sorted_words = df_sorted["Word"].tolist()
    sorted_scores = df_sorted["Importance Score"].tolist()

    # Plot the sorted columns as a bar chart using matplotlib: importance scores pushing classifier towards Real prediction in blue, towards Fake in red
    fig, ax = plt.subplots(figsize=(10, 6))
    # Separate the pushing-towards-positive class (1 = Fake) and pushing-towards-negative class (0 = Real) importancescores into red and blue colors
    colors = ["red" if score > 0 else "blue" for score in sorted_scores]
    # Create a range of integers from 0 to the number of most important words for plotting the y-axis
    y_pos = np.arange(len(sorted_words))
    # Create horizontal bar-chart with words on the y-axis and the importance (scores) constituting the width. Also map each word to the prediction color
    ax.barh(y_pos, sorted_scores, color=colors)
    # Arrange the word positions on the y-axis
    ax.set_yticks(y_pos)
    # Label the words on the y-axis
    ax.set_yticklabels(sorted_words)
    # Label the x-axis showing word importance score
    ax.set_xlabel("Word Score (Red = Fake, Blue = Real)")
    # Add title
    ax.set_title("LIME Explanation (Sorted by Importance)")
    # Get the current axes (matplotlib plot) and invert the y-axis to show the most important words at the top
    # Reference: https://stackoverflow.com/questions/2051744/how-to-invert-the-x-or-y-axis
    plt.gca().invert_yaxis()
    # Return the matplotlib figure object and the list of LIME importance (word, score) tuples
    return fig, top_features


def convertToNaturalLanguageExplanation(lime_features, prob):
  """
    Converts the LIME importance word scores to a natural language explanations.

    Input Parameters:
        lime_features (list): the list of tuples storing (word, score)
        prob (float): the confidence i.e. probability of news sample being in the positive (Fake News) class

    Output:
        A string containing the natural language explanation.
  """

  # Convert the probability score for the text being fake news into a percentage to the nearest integer and round the percentage to an integer
  perc = round(prob * 100)

  # If the probability is over 0.5, then show overall prediction as fake
  if prob > 0.5:
    category = "fake"
  # If probability is less than 0.5, then show overall prediction as real
  elif prob < 0.5:
    category = "real"
  # If probability is 0, set category to "neither fake nor real"
  else:
    category = "neither fake nor real"

  # Write out a natural language explanation describing what the probability score means
  probability_explanation = f"""This news article has been classed as {perc}% fake news,
                             meaning that its binary label would be: '{category.upper()}'. """

  # Words to explain this particular prediction in natural language will stored in this list
  included_features= []

  # Iterate over the most important features to generate the natural language explanation
  for feature_tuple in lime_features:
    # Add word features based on certain conditions: only add the word feature for explanation if its importance score is actually pushing
    # towards the class that was predicted overall
    if (
        (feature_tuple[1] > 0 and prob > 0.5) # If the word importance is POSITIVE (pushing towards Fake News) AND the final prediction was Fake Newws
        or (feature_tuple[1] < 0 and prob < 0.5) # If the word importance iS NEGATIVE (pushing towards Real News) AND the final prediction was Real News
        or (prob == 0.5) # If the prediction is neutral, just add all the top five features, whether positive or engative
    ):
        included_features.append(feature_tuple[0])
    # Only include the top 5 features for the natural language explanation
    if included_features == 5:
        break

  # Write the generate the natural language explanation about the word features and what their importance means
  feature_explanation = f"""
    The following words in the news text, which have been ranked in terms of their importance, had the most impact 
    on the classifier predicting that this news text is {category} news:
    \n1. {included_features[0]}\n2. {included_features[1]}\n3. {included_features[2]}\n4. {included_features[3]}\n5. {included_features[4]}
    """

  # Concatenate the natural language descriptions
  return probability_explanation + feature_explanation

    
######################################################### Streamlit Application ####################################################################################

# Add the title to the top of the application
st.title("Explainable Fake News Detection")

# "Insert containers separated into tabs"
# Reference: https://docs.streamlit.io/develop/api-reference/layout/st.tabs
tab1, tab2 = st.tabs(["Analyze Text", "About"])

# First tab: the news text content analyzer
with tab1:
    # Creates the text input area for the news text to copy and paste into
    news_text = st.text_area("Enter news text to analyze:", height=200)
    
    # Create a dropdown for selecting analysis options: either LIME-based charts or natural-language based text explanation of the important featurs
    explanation_type = st.selectbox(
        "Choose explanation type:",
        ["LIME Explainer", "Natural Language"]
    )

    # When user clicks on the "Analyze Text" button
    if st.button("Analyze Text"):
        # If the news_text has been entered properly, proceed
        if news_text:
            try:
                # Loads the pre-trained model for making the prediction and show the spinner
                with st.spinner("Loading detection model..."):
                    pipeline = loadModel()
                
                # Analyzes the user's inputted text with the LIME explainer and show spinner to user
                with st.spinner("Analyzing the text..."):
                    # Get the probability of the news being fake news by extracting the probability array (at index 0 as there is only one sample)
                    prediction = pipeline.predict_proba([news_text])[0]
                
                # Displays prediction results
                st.subheader("Results:")
                # Converts the fake news (index 1) probability to percentage
                confidence = prediction[1] * 100
                # Convert probability value to text label --> fake if over 0.5 probability
                pred_label = "Fake" if prediction[1] > 0.5 else "Real"
                # Displays the label and probability scores using the Streamlit Markdown function to use bold text
                st.markdown(f"**Prediction label:** {pred_label} News") # Show final label
                st.markdown(f"**Probability that this is fake news**: {confidence:.2f}%") # Show confidence level (probability score)
                
                # Applies the LIME feature importance explanation function to get the importance features for the prediction as plot and list of score tuples
                with st.spinner("Analyzing important text features and generating explanation..."):
                    figure, features = generateLIMEExplanation(pipeline, news_text)

                # Generates and displays the LIME explanation as a matplotlib bar chart if user selected "Lime Explainer" instead of "Natural Language"
                if explanation_type == "LIME Explainer":
                    # Inform users of chart-generating progress
                    with st.spinner('Generating chart...'):
                        st.subheader("LIME Explanation")
                    # Display the matplotlib figure bar chart showing color-coded importance scores
                    # Reference: https://docs.streamlit.io/develop/api-reference/charts/st.pyplot
                    st.pyplot(figure)      

                    # Display the top features as a bulletted list using the Streamlit Markdown function
                    st.markdown("## Top 10 Ranked Words Influencing Prediction:")

                    ranked_words = ""
                    for feature, score in features[:10]:
                        if score > 0:
                            news_label =  "fake"
                        if score < 0:
                            news_label = "real"
                        if score == 0:
                            news_label = "neutral"
                        ranked_words += f"    - **{feature}**: {score:.3f} -----> (*{news_label} news*)\n" # Show importance score to 3 decimal points

                    st.markdown(ranked_words)

                # Generates and displays the Natural Language explanation if user selects the "Natural Langugae" option instead of the "LIME Explainer" option
                else:  # Natural Language
                    st.subheader("Text Explanation")
                    explanation = convertToNaturalLanguageExplanation(features, prediction[1])
                    st.write(explanation)

            # Catch error if something goes wrong with news text processing               
            except Exception as e:
                st.error(f"Error - invalid news text: {str(e)}")  

        else:
            st.write("Please enter a valid news text") # No news text was entered

# In the second tab, add explanation how the LIME tool works (but this needs to be better explained in layman's terms in the future)
with tab2:
    st.markdown("## About this app:")
    st.markdown(" - This app uses machine learning to detect fake news and explain its decisions.")
    st.markdown(" - This model has been trained on the WELFake dataset and explanations have been generated using the LIME algorithm "
                "(Local Interpretable Model-agnostic Explanations).")
    st.markdown(" - LIME treats the underlying model as a black box, but perturbs (randomly changes) the text we "
                "want to explain and then 'learns' the impact of changing each feature (word) on the final outputted prediction.")
    st.markdown(" - You can find the (WELFake) dataset used for training [here](https://zenodo.org/records/4561253)")
    st.markdown(" - Please check out [this paper](https://arxiv.org/pdf/1602.04938) for a more detailed explanation of how "
                "LIME works.")