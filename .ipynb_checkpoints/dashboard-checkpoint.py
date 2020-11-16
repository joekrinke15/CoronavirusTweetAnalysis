
import plotly
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import os
import chart_studio
import itertools
from collections import Counter
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import nltk
from nltk import bigrams
import itertools
import pandas as pd
from IPython.display import HTML

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

def matrix_to_df(diagnosis_data, n):
    """
    Creates a co-occurence matrix and converts it to a pandas dataframe.
    Inputs:
        diagnosis_data: A co-occurrence matrix.
        n: The number of diseases you want to analyze.
    Outputs:
        pandas_matrix: a dataframe of the co-occurrence matrix.
    """
    # Generate corpus
    corpus = topn_diagnoses(diagnosis_data, n)
    # Create one list using many lists
    data = list(itertools.chain.from_iterable(corpus))
    # Create co-occurrence matrix
    matrix, vocab_index = gen_matrix(data)
    # Put Matrix into pandas df
    pandas_matrix = pd.DataFrame(matrix, index=vocab_index,
                             columns=vocab_index)
    return(pandas_matrix)

def gen_matrix(corpus):
    """
    Creates a co-occurrence matrix from a corpus/list.
    
    Inputs:
        Corpus: A List of list of words.
    Outputs:
        matrix: A co-occurrence matrix. 
        vocab_index: A list of all the words in the vocabulary.
    """
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 
    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))
 
    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
 
    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    matrix = np.zeros((len(vocab), len(vocab)))
 
    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        matrix[pos_current][pos_previous] = count
    matrix = np.matrix(matrix)
 
    # return the matrix and the index
    return matrix, vocab_index

def filter_data(diagnosis, disease):
    """
    Creates a dataframe of people who have a given disease as one of their diagnoses.
    Inputs: 
        Diagnosis dataset.
    Outputs: 
        Filtered data based on chosen disease.
    """
    filtered_data = diagnosis.loc[diagnosis['subject_id'][diagnosis['short_title'] == disease]]
    return(filtered_data)

@st.cache(show_spinner=False)
def get_diagnosis_data():
    """
    Reads in diagnosis data, calculating the length of stay and recategorizing ethnicity.
    Outputs:
        Diagnosis: Diagnosis data with new staylength and ethnicity categories.
    """
    diagnosis = pd.read_csv("https://mimicdatasets.s3.amazonaws.com/diagnosis.csv")
    # add time stayed in the hospital
    diagnosis['staylength']= pd.to_datetime(diagnosis['dischtime']) - pd.to_datetime(diagnosis['admittime'])
    diagnosis['staylength'] = pd.to_timedelta(diagnosis.staylength).dt.total_seconds()
    # Convert seconds to days
    diagnosis['staylength'] = diagnosis['staylength']/(24*60*60)
    # regroup ethnicity
    diagnosis.loc[diagnosis.ethnicity.str.contains('^ASIAN'), 'ethnicity'] = 'Asian'
    diagnosis.loc[diagnosis.ethnicity.str.contains('^HISPANIC/LATINO'), 'ethnicity'] = 'Hispanic or Latino'
    diagnosis.loc[diagnosis.ethnicity.str.contains('^BLACK'), 'ethnicity'] = 'Black/African-American'
    diagnosis.loc[diagnosis.ethnicity.str.contains('^WHITE'), 'ethnicity'] = 'White'
    diagnosis.loc[diagnosis.ethnicity == 'UNABLE TO OBTAIN', 'ethnicity'] = 'Unknown/Not Specified'
    main_cats = ['White', 'Black/African-American', 'Unknown/Not Specified', 
             'Hispanic or Latino', 'Asian']
    diagnosis.loc[~diagnosis.ethnicity.isin(main_cats), 'ethnicity'] = 'Other'
    # Capitalize disease names 
    diagnosis['short_title'] = diagnosis['short_title'].str.title()
    return(diagnosis)
diagnosis = get_diagnosis_data()

def topn_diagnoses(diagnosis, n):
    """
    Creates a list of lists containing the top n most frequent diagnoses.
    Inputs:
        diagnosis_matrix: A column containing diagnosis data for patients.
        n: Number of diseases to examine.
    Outputs:
        corpus: A list of lists containing all the diagnoses. One list for each patient.
    """
    diagnosis_subset= diagnosis[['subject_id', 'short_title', 'seq_num']]
    top_n= diagnosis_subset['short_title'].value_counts()[:n].index.tolist()
    subset_matrix= diagnosis_subset[diagnosis_subset['short_title'].isin(top_n)]
    subset_matrix = subset_matrix.groupby('subject_id')['short_title'].apply(','.join).reset_index()
    diagnoses = subset_matrix['short_title'].to_list()
    corpus = []
    for i in diagnoses:
        corpus.append(i.split(','))
    return (corpus)

#path_to_download_folder = str(os.path.join(Path.home(), "Downloads"))


def disease_freq(diagnosis, disease):
    """
    Calculate relative disease frequency by race.
        Inputs: 
        Diagnosis dataset, chosen disease.
        Outputs: 
        Relative disease frequency by race.
    """
    ethnic_total = diagnosis.groupby('ethnicity').count()[['row_id']]
    subset_total = filter_data(diagnosis, disease).groupby('ethnicity').count()[['row_id']]
    scaled_data = (subset_total/ethnic_total)
    scaled_data['row_id'][pd.isna(scaled_data["row_id"])] = 0
    scaled_data = scaled_data.reset_index()
    scaled_data = scaled_data.sort_values(by='row_id', ascending=False)
    scaled_data['row_id'] = scaled_data['row_id'] * 100
    return(scaled_data)

# calculate relative disease frequency by admittyp
def admit_freq(diagnosis):
    """
    Calculate where patients are admitted for a given disease. 
    Inputs:
        diagnosis: The disease you're interested in.
    Outputs:
        admit_total: The relative frequency of admission location for that disease. 
    """
    diagnosis['admission_location'] = diagnosis['admission_location'].str.title()
    admit_total = diagnosis.groupby('admission_location').count()[['row_id']]
    admit_total = admit_total.reset_index()
    admit_total = admit_total.sort_values(by='row_id', ascending=False)
    return(admit_total)


@st.cache(show_spinner=False)
def get_patient_data():
    return pd.read_csv("https://mimicdatasets.s3.amazonaws.com/Patient.csv")
patients = get_patient_data()

@st.cache(show_spinner=False)
def get_admit_data():
    """
    Loads in admissions table.
    """
    admit = pd.read_csv("https://mimicdatasets.s3.amazonaws.com/Admit.csv")
    admit['admission_location'] = admit['admission_location'].str.title()
admit = get_admit_data()

def get_merged_data(diagnosis):
    """
    Merges diagnosis table and admissions table. 
    """
    return pd.merge(diagnosis, patients, on='subject_id', how='left')
merged_data = get_merged_data(diagnosis)

@st.cache(show_spinner=False)
def get_top_diseases():
    """
    Returns the top 30 most common diseases from the merged dataset. 
    """
    data = pd.DataFrame(merged_data.short_title.value_counts())[:30]
    data.columns = ['count']
    return data
top_diseases = get_top_diseases()

@st.cache(show_spinner=False)
def get_top_5_admin_locations(merged_data):
    """
    Gets the top 5 admissions locations from the merged dataset. 
    """
    merged_data['admission_location'] = merged_data['admission_location'].str.title()
    return pd.DataFrame(merged_data.admission_location.value_counts())[:5]
locations = get_top_5_admin_locations(merged_data)

@st.cache(show_spinner=False)
def get_ethnicity():
    """
    Gets the frequency of each ethnic group in the merged dataset. 
    """
    return pd.DataFrame(merged_data.ethnicity.value_counts())
ethnicity = get_ethnicity()

@st.cache(show_spinner=False)
def add_age(merged_data):
    """
    Calculates an age column from the admitdate and dob. Group ages into categories.
    Inputs: 
        Merged data
    Outputs:
        Merged data with age categories added.
    """
    merged_data['admittime'] = pd.to_datetime(merged_data['admittime']).dt.date
    merged_data['dob'] = pd.to_datetime(merged_data['dob']).dt.date
    merged_data['age_num'] = merged_data.apply(lambda e: (e['admittime'] - e['dob']).days/365, axis=1)
    # transform age into categorical variable
    # create a list of our conditions
    conditions = [
    (merged_data['age_num'] <= 25),
    (merged_data['age_num'] > 25) & (merged_data['age_num'] <= 50),
    (merged_data['age_num'] > 50) & (merged_data['age_num'] <= 75),
    (merged_data['age_num'] > 75)
    ]    
    gender = pd.DataFrame(merged_data.gender.value_counts())
    gender.index = ['Male', 'Female']
    fig3 = px.pie(gender, names=gender.index, values='gender')
    # create a list of the values we want to assign for each condition
    values = ['25 and Under', '25 to 50', '50 to 75', 'Above 75']
    # create a new column and use np.select to assign values to it using our lists as arguments
    merged_data['age'] = np.select(conditions, values)
    return merged_data
merged_data_age = add_age(merged_data)

def demo_disease(ethnicity, gender, age):
    """
    Return the top 10 diseases and staylength for selected demographic information
    Inputs: 
        ethnicity, gender, age
    Outputs: 
        top 10 diagnosis names, staylength distribution
    """
    condition = (merged_data_age['age'] == age) & (merged_data_age['ethnicity'] == ethnicity) & (merged_data_age['gender'] == gender)
    top10diag = pd.DataFrame(merged_data_age[condition]['short_title'].value_counts()[:10])
    top10diag.columns = ['count']
    staylength = pd.DataFrame(merged_data_age[condition]['staylength'])
    staylength.columns = ['staylength']
    return top10diag, staylength


# Change background color
st.markdown("""
<style>
body {
    color: #000;
    background-color: ##FFFFFF;
}
</style>
    """, unsafe_allow_html=True)

@st.cache(show_spinner=False)
def get_association_rules_data():
    """
    Loads market basket analysis dataframe.
    """
    #load data
    data = pd.read_csv('https://mimicdatasets.s3.amazonaws.com/Association_Rules.csv')
    #drop first, unnamed column
    data = data.drop(columns='Unnamed: 0')
    #drop initial label
    return(data)

#sidebar options
topic = st.sidebar.radio('Choose Topic to Explore', ('General Trends', 'Disease to Demographics',

                                   'Demographics to Disease', 'Market Basket Analysis', 'Co-occurrence Analysis'))



# General Trends Dashboard
if topic == 'General Trends':

    # About the dashboard
    expander_bar = st.beta_expander('About')
    expander_bar.markdown("""
    **Data Source**: MIMIC-III Critical Care Database developed by the MIT Lab for Computational Physiology. \n 
    **Data**: Health-related data of over 60,000 patients who stayed in critical care units in Beth Israel Israel Deaconess Medical Center (Boston, MA) between 2001 and 2012. \n
    **Python Libraries**: Streamlit, Pandas, Plotly, Networkx, Nltk \n
    **References**: [Streamlit Documentation](https://docs.streamlit.io/en/stable/api.html), [Mimic Dataset](https://mimic.physionet.org/about/mimic/). \n
    **GitHub Repository Link**: [GitHub](https://github.com/joekrinke15/MIMIC-Analysis)
    """)

    # Display title
    st.markdown("<h1 style='text-align: center; color: black;'>MIMIC Dataset General Trends</h1>", unsafe_allow_html=True)
    
    #plot Top Most Frequent Diseases plot
    n = st.slider('N of Diseases', 1, 30, 5)
    # plot bar chart
    fig1 = go.Figure(go.Bar(
        x=top_diseases[:n].index,
        y=top_diseases[:n]['count'], marker_color='#FE6692'
    ))
    fig1.update_layout(template='ggplot2', title = f'Top {n} Most Frequent Diseases', yaxis_title = 'Frequency', xaxis_title = 'Disease')

    st.plotly_chart(fig1, use_container_width=True)
    
    # Create first 2 columns to hold graphs
    col1, col2 = st.beta_columns(2)


    # Plot age distribution
    age_data_bar= merged_data_age.groupby(by='age').agg(
    Frequency=pd.NamedAgg(column="subject_id", aggfunc="count")
)
    age_bar = px.bar(age_data_bar, labels = {'age':'Age Group', 'value': 'Frequency'}, title = 'Age Distribution')
    age_bar.update_layout(title_x = .50,showlegend=False, xaxis_title = 'Age Group')
    col1.plotly_chart(age_bar)

    # Plot admission location frequency
    fig2 = px.pie(locations, names=locations.index, values='admission_location', height = 550, width = 800)

    fig2.update_layout(title='Top 5 Admission Locations')
    fig2.update_layout(title_x=.50)
    col2.plotly_chart(fig2, use_container_width=True)


    # Create second 2 columns to hold graphs
    # plot Patients by Gender pie
    col3, col4 = st.beta_columns(2)
    gender = pd.DataFrame(merged_data.gender.value_counts())
    gender.index = ['Male', 'Female']
    fig3 = px.pie(gender, names=gender.index, values='gender')
    fig3.update_layout(title='Patients by Gender', title_x = .50)
    col3.plotly_chart(fig3, use_container_width=True)

    #plot Ethnicity
    fig4 = go.Figure(go.Bar(
        x=ethnicity.index,
        y=ethnicity['ethnicity'], marker_color='#19D3F3'
    ))
    fig4.update_layout(template='ggplot2', title='Patients by Ethnicity', yaxis_title = 'Frequency', xaxis_title = 'Ethnicity')
    col4.plotly_chart(fig4, use_container_width=True)
    

# Disease to demographics
elif topic == 'Disease to Demographics':
    # Display title
    st.markdown("<h1 style='text-align: center; color: black;'>Disease to Demographics</h1>", unsafe_allow_html=True)
   # Pick a disease and use it to filter the data
    chosen_disease = st.selectbox('Choose a disease:',
                                  sorted(list(set(diagnosis['short_title'].value_counts()[:100].index))))
    filtered_data = filter_data(diagnosis, chosen_disease)
    data_matrix = matrix_to_df(filtered_data, 25)
    age_data_filtered = filter_data(merged_data_age, chosen_disease)
    
    # Group the data by age to create a bar chart
    age_data_filtered = age_data_filtered.groupby(by='age').agg(
    Frequency=pd.NamedAgg(column="subject_id", aggfunc="count")
)
    age_bar = px.bar(age_data_filtered, labels = {'age':'Age Group', 'value': 'Frequency'}, title = 'Age Distribution')
    age_bar.update_layout(title_x = .50,showlegend=False)

    #Create first 2 columns to hold graphs
    col1, col2 = st.beta_columns(2)

    # Plot age bar chart
    col1.plotly_chart(age_bar, use_container_width=True)

    # Plot admission location frequency
    admit_loc = px.pie(admit_freq(filtered_data), names='admission_location', values='row_id', title = 'Admission Locations',height = 450, width = 800)
    admit_loc.update_layout(title_x=.50)
    col2.plotly_chart(admit_loc, use_container_width=True)

    # create columns to store stuff. Each new set of columns is a row.
    col3, col4 = st.beta_columns(2)
    # Plot gender frequency
    gender = pd.DataFrame(merged_data[merged_data['short_title'] == chosen_disease].gender.value_counts())
    gender.index = ['Male', 'Female']
    fig3 = px.pie(gender, names=gender.index, values='gender', title='Gender Distribution')
    fig3.update_layout(title_x = .50)

    #col3.plotly_chart(stay,height=512,width=512)
    col3.plotly_chart(fig3, use_container_width=True)
    # Plot relative disease frequency by ethnicity
    disease_freq = px.bar(disease_freq(diagnosis, chosen_disease), x='ethnicity', y='row_id', title = 'Percentage of Ethnic Group with ' + chosen_disease, labels = {'ethnicity': 'Ethnicity', 'row_id':'Percentage of Population'},width = 800, height = 450)
    disease_freq.update_layout(title_x=.50)
    col4.plotly_chart(disease_freq,use_container_width=True)
    
# Demographics to Disease Dashboard
elif topic == 'Demographics to Disease':
    # Display title
    st.markdown("<h1 style='text-align: center; color: black;'>Demographics to Disease</h1>",
                unsafe_allow_html=True)
   

    # Select demographic information
    ethnicity = st.selectbox('Choose an ethnicity:', 
                             sorted(list(merged_data_age.ethnicity.value_counts().index)))
    gender = st.selectbox('Choose a gender:', 
                        ['F', 'M'])
    age = st.selectbox('Choose an age:', 
                       sorted(list(merged_data_age.age.value_counts().index)))
    
    # the dataframe for plotting
    top10diag, staylength = demo_disease(ethnicity, gender, age)
    
     # Create first 2 columns to hold graphs
    col1, col2 = st.beta_columns(2)
    
    # plot top 10 Most Frequent Diseases for given ethicity, gender and age
    diag_fig = go.Figure(go.Bar(
        x=top10diag.index,
        y=top10diag['count'], marker_color='light blue'
    ))
    diag_fig.update_layout(template='ggplot2', title = 'Top 10 Most Frequent Diseases', yaxis_title = 'Frequency', xaxis_title = 'Disease')

    col1.plotly_chart(diag_fig, use_container_width=True)
    
    # plot staylength distribution for given eth, gender and age
    stay_fig = px.histogram(staylength, x="staylength", title = 'Distribution of Days in the Hospital', labels = {"staylength":"Days in Hospital", "count": "Frequency"}, width = 800, height = 450)
    stay_fig.update_layout(title_x = .50, yaxis_title = 'Frequency')
    col2.plotly_chart(stay_fig, use_container_width=True)

# Market basket analysis
elif topic == 'Market Basket Analysis':
    # Display title
    st.markdown("<h1 style='text-align: center; color: black;'>Market Basket Analysis of Diseases</h1>",
                unsafe_allow_html=True)
    # plot Top rules 
    n = st.slider('N of Rules', 1, 30, 5)
    arules = get_association_rules_data()
    arules = arules.sort_values(by='lift', ascending=False)
    arules.columns = arules.columns.str.capitalize()
    arules = arules[1:n+1]
    arules['Disease Combination'] = arules['A'] + ' and ' + arules['B']
    arules['Disease Combination'] = arules['Disease Combination'].str.title()
    
    fig =  px.bar(arules, x='Disease Combination', y='Lift', title = 'Disease Combinations with the Highest Lift')
    fig.update_layout(title_x = .50)
    st.plotly_chart(fig, use_container_width = True )
    
elif topic == 'Co-occurrence Analysis':
    # Display title
    st.markdown("<h1 style='text-align: center; color: black;'>Co-occurrence Analysis of Diseases</h1>",
                unsafe_allow_html=True)
    chosen_disease = st.selectbox('Choose a disease:',sorted(list(set(diagnosis['short_title'].value_counts()[:100].index))))
    filtered_data = filter_data(diagnosis, chosen_disease)
    data_matrix = matrix_to_df(filtered_data, 25)
    # Create graph object and prepare for graphing
    g = nx.from_numpy_matrix(data_matrix.to_numpy())
    pos = nx.fruchterman_reingold_layout(g)
    N = list(data_matrix.columns)
    Node = g.nodes()
    E = g.edges()
    labels = N
    for n, p in pos.items():
        g.nodes[n]['pos'] = p
    # Get sizes of nodes
    sizes = data_matrix.sum(axis=0).to_list()
    sizeref = 2. * max(sizes) / (10 ** 2)

    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = g.nodes[edge[0]]['pos']
        x1, y1 = g.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_adjacencies = []
    node_text = []
    i = 0
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('Disease:' + str(labels[i]) + ', Number of Connections: ' + str(int(sizes[i])))
        i += 1
    # Add edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=.4, color='#888'),
        hoverinfo='none',
        mode='lines')
    # Get node x and y values
    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = g.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
    
    # Add nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'greys' | 'YlgnBu' | 'greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlgnBu',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace.marker.color = sizes
    node_trace.marker.size = [s for s in sizes]
    node_trace.marker.sizeref = sizeref
    node_trace.text = node_text

    data = [edge_trace, node_trace]
    
    # Update graph display
    layout = go.Layout(
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=450,
        width=800)
    cooccurrence = go.Figure(data=data,
                             layout=layout)
    cooccurrence.update_layout(title='Diseases Co-occurring with ' + str(chosen_disease), title_x=.50)
    
    # Plot co-occurrence
    st.plotly_chart(cooccurrence, use_container_width = True)
    
    
    
    
    
