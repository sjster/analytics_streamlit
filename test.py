import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
#from pymc3 import traceplot
import pymc3 as pm
import arviz as az
from darts import TimeSeries
from darts.models import ExponentialSmoothing, NaiveDrift
from stackapi import StackAPI
import pandas as pd
import plotly.express as px

RUN_STACKOVERFLOW = 0
st.set_page_config(layout='wide')


def get_time_series_trend(df, columns, fig):
    dart_df = TimeSeries.from_dataframe(df.reset_index(), columns[0], columns[1])
    model = NaiveDrift()
    model.fit(dart_df)
    trend = model.predict(10)
    fig2 = px.line(trend.pd_dataframe(),title='Forecast')
    #fig2.update_traces(marker_color='green')
    fig.add_trace(fig2['data'][0])
    fig.data[1].line.color = 'green'
    fig.data[1].line.width = 8
    print(fig)
    return(fig)


def process_df(questions, query, type):
  print(" --------------------- %s %s -------------------"%(query, type))
  df_original = pd.json_normalize(questions['items'])
  df = df_original[['title','score','view_count','answer_count','is_answered','tags','question_id']]
  corr_matrix = df[['score', 'view_count','answer_count']].corr()
  #df_original[['creation_date','last_activity_date']].hist()
  #df[['score','view_count','answer_count']].hist()
  return(df, df_original[['creation_date','last_activity_date']], corr_matrix)


def run_analytics(query):
    if(RUN_STACKOVERFLOW == 1):
        SITE = StackAPI('stackoverflow')
        questions_votes = SITE.fetch('questions', min=20, tagged=query, sort='votes')
        questions_activity = SITE.fetch('questions', tagged=query, sort='activity')
        #df_votes_query, df_votes_query_dates = process_df(questions_votes, query, 'votes')
        df_activity_query, df_activity_query_dates, corr_matrix = process_df(questions_activity, query, 'activity')
        df_activity_unans = df_activity_query[df_activity_query['is_answered'] == False]
        questions = pd.json_normalize(questions_activity['items'])
        questions.to_json(query + '_questions.json')
        questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
        questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
        questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
        questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')
        data = questions[questions['score'] >= 0]
    else:
        filename = query + '_questions.json'
        print("Reading file ",filename)
        questions = pd.read_json(filename)
        questions['creation_date'] = questions['creation_date'] / 1000
        questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
        questions['last_activity_date'] = questions['last_activity_date'] / 1000
        questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
        questions['last_edit_date'] = questions['last_edit_date'] / 1000
        questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
        questions['closed_date'] = questions['closed_date'] / 1000
        questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')
        data = questions[questions['score'] >= 0]
        corr_matrix = questions[['score', 'view_count','answer_count']].corr()

    # -------------- Header info ----------------#
    st.title("Stackoverflow queries for " + query)
    age = st.slider('How old are you?', 0, 130, 25)
    st.subheader("Data")
    headercol1, headercol2, headercol3, headercol4 = st.columns([1,1,1,1])
    num_questions = len(questions)
    headercol1.metric("Number of questions",num_questions)
    closed = len(questions[questions['is_answered'] == True])
    headercol2.metric("Closed",closed)
    headercol3.metric("Closed fraction", closed/num_questions)
    headercol4.metric("Questions with negative score", len(questions) - len(data))
    st.write(questions, width=20, height=100)
    st.subheader("Summarize the data")
    if st.checkbox('Show dataframe'):
        st.write(questions.describe(include='all'))
    st.subheader("Correlation matrix")
    st.write(corr_matrix)

    # --------------- EDA and feature extraction --------------- #
    questions['creation_date_only'] = pd.to_datetime(questions['creation_date'].dt.date)
    # resample by month
    questions_by_week = questions.set_index('creation_date_only').resample('M').count()['question_id']
    answers_and_views_by_week = questions.set_index('creation_date_only').resample('M').sum()[['view_count', 'answer_count']]

    print(data.mean())

    # ----------------- Views ----------------#
    st.subheader("Views")
    col1, col2 = st.columns([1,1])
    fig = px.histogram(questions['view_count'], nbins=200, title="Distribution of views for posts")
    with col1:
        st.plotly_chart(fig)

    with col2:
        st.text('Bayesian inference of the parameters of the views distribution \n\n')
        with st.spinner('Inferring the parameters of the view distribution (Poisson)'):
            with pm.Model() as model:
              mu = pm.Uniform('mu', lower=0, upper=1000)
              y = pm.Poisson('y', mu=mu, observed=questions['view_count'])
              trace_t = pm.sample(1000)
        st.success('Complete')
        summary = az.summary(trace_t)
        st.metric('Mean estimate for views', summary['mean'])
        st.write(summary)

    # -------------------- Scores --------------- #
    st.subheader("Scores")
    col3, col4 = st.columns([1,1])
    fig = px.histogram(questions['score'], nbins=200, title="Distribution of scores for posts")
    with col3:
        st.plotly_chart(fig)

    with col4:
        st.text('Bayesian inference of the parameters of the scores distribution \n\n')
        with st.spinner('Inferring the parameters of the score distribution (exponential)'):
            with pm.Model() as model:
              lam = pm.Uniform('lam', lower=0, upper=20)
              y = pm.Exponential('y', lam=lam, observed=data['score'])
              trace_t = pm.sample(1000)
        st.success('Complete')
        summary = az.summary(trace_t)
        st.metric('Mean estimate for score (ignoring the negative scores)', summary['mean'])
        st.write(summary)

    col5, col6 = st.columns([1,1])
    fig = px.scatter(x=questions['view_count'], y=questions['score'],  title="Relationship between views (x) and score (y)")
    with col5:
        st.plotly_chart(fig)
    fig = px.histogram(questions['answer_count'], nbins=10, title="Distribution of answer counts for posts")
    with col6:
        st.plotly_chart(fig)

    # ---------------------- Question timeseries ------------------ #
    st.subheader("Time evolution of questions, answers and views")
    col7, col8 = st.columns([1,1])
    fig = px.bar(questions_by_week, title="Number of questions by month, green trend line indicates forecast for next 10 months")
    fig = get_time_series_trend(questions_by_week, ('creation_date_only', 'question_id'), fig)
    with col7:
        st.plotly_chart(fig)
    fig = px.bar(answers_and_views_by_week, title="Number of answers and views by month")
    with col8:
        st.plotly_chart(fig)

    # -------------------- Time series forecast based on the trend ------------ #


    #dart_df.plot(label='Past questions')
    #trend.plot(label='Trend forecast for questions')

if __name__ == '__main__':

    query = 'mlflow'
    run_analytics(query)
