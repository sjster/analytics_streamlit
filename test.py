import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from stackapi import StackAPI
import pandas as pd
import plotly.express as px

RUN_STACKOVERFLOW = 0
st.set_page_config(layout='wide')

SITE = StackAPI('stackoverflow')
query = 'mlflow'
questions_votes = SITE.fetch('questions', min=20, tagged=query, sort='votes')
questions_activity = SITE.fetch('questions', tagged=query, sort='activity')

def process_df(questions, query, type):

  print(" --------------------- %s %s -------------------"%(query, type))
  df_original = pd.json_normalize(questions['items'])
  df = df_original[['title','score','view_count','answer_count','is_answered','tags','question_id']]
  corr_matrix = df[['score', 'view_count','answer_count']].corr()
  #df_original[['creation_date','last_activity_date']].hist()
  #df[['score','view_count','answer_count']].hist()
  return(df, df_original[['creation_date','last_activity_date']], corr_matrix)

if(RUN_STACKOVERFLOW == 1):
    #df_votes_query, df_votes_query_dates = process_df(questions_votes, query, 'votes')
    df_activity_query, df_activity_query_dates, corr_matrix = process_df(questions_activity, query, 'activity')
    df_activity_unans = df_activity_query[df_activity_query['is_answered'] == False]
    questions = pd.json_normalize(questions_activity['items'])
    questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
    questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
    questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
    questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')

else:
    questions = pd.read_json('mlflow_questions.json')
    questions.to_json(query + '_questions.json')
    questions['creation_date'] = questions['creation_date'] / 1000
    questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
    questions['last_activity_date'] = questions['last_activity_date'] / 1000
    questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
    questions['last_edit_date'] = questions['last_edit_date'] / 1000
    questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
    questions['closed_date'] = questions['closed_date'] / 1000
    questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')

    corr_matrix = questions[['score', 'view_count','answer_count']].corr()

st.title("Stackoverflow queries")
age = st.slider('How old are you?', 0, 130, 25)
st.write("Search term : ", query)
st.write("Here's our first attempt at using data to create a table:")
st.write(questions.describe(include='all'))
st.write(corr_matrix)
st.write(questions, width=20, height=100)

col1, col2 = st.beta_columns([1,1])

fig = px.histogram(questions['view_count'], nbins=200, title="Distribution of views for posts")
with col1:
    st.plotly_chart(fig)
fig = px.histogram(questions['score'], nbins=200, title="Distribution of scores for posts")
with col2:
    st.plotly_chart(fig)

col3, col4 = st.beta_columns([1,1])
fig = px.scatter(x=questions['view_count'], y=questions['score'],  title="Relationship between views (x) and score (y)")
with col3:
    st.plotly_chart(fig)
fig = px.histogram(questions['answer_count'], nbins=10, title="Distribution of answer counts for posts")
with col4:
    st.plotly_chart(fig)
questions['creation_date_only'] = pd.to_datetime(questions['creation_date'].dt.date)
questions_by_week = questions.set_index('creation_date_only').resample('M').count()['question_id']
answers_and_views_by_week = questions.set_index('creation_date_only').resample('M').sum()[['view_count', 'answer_count']]

col5, col6 = st.beta_columns([1,1])
fig = px.bar(questions_by_week, title="Number of questions by month")
with col5:
    st.plotly_chart(fig)
fig = px.bar(answers_and_views_by_week, title="Number of answers and views by month")
with col6:
    st.plotly_chart(fig)


if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
