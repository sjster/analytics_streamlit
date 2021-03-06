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
import spacy
from stackapi import StackAPI
import pandas as pd
import plotly.express as px

RUN_STACKOVERFLOW = 0
count = 0
st.set_page_config(layout='wide')

#nlp = spacy.load("en_core_web_sm")

def apply_spacy(x):
  doc = nlp(x)
  elem = list(doc.ents)
  elem.extend(doc.noun_chunks)
  #elem = { "Entities" : list(doc.ents), "Nouns": list(doc.noun_chunks) }
  return(elem)


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


def plot_tags(df):
    flat_list = [item for sublist in df['tags'].values for item in sublist]
    tags_bar = pd.Series(flat_list).value_counts(ascending=False)
    fig = px.bar(tags_bar)
    fig.update_layout(height=500, width=1200)
    fig.update_yaxes(type="log")
    st.plotly_chart(fig)


def run_analytics():

    print("Rerunning")
    # -------------------- Time series forecast based on the trend ------------ #


def Poisson_inference(col):
    with pm.Model() as model:
      mu = pm.Uniform('mu', lower=0, upper=1000)
      y = pm.Poisson('y', mu=mu, observed=col)
      trace_t = pm.sample(1000)
    return(trace_t)


def Exponential_inference(col):
    with pm.Model() as model:
      lam = pm.Uniform('lam', lower=0, upper=20)
      y = pm.Exponential('y', lam=lam, observed=col)
      trace_t = pm.sample(1000)
    return(trace_t)


def get_stats(df):
    q = df.quantile([0.25,0.50,0.75])
    mean = df.mean()
    stats = pd.DataFrame({'0.25': q[0.25], '0.50': q[0.50], '0.75': q[0.75], \
            'mean': mean, 'mode': df.mode()[0]}, index=['stats'])
    st.metric('Median value',q[0.50])
    st.text('Quantile information')
    st.dataframe(stats)



if __name__ == '__main__':

    st.title("Stackoverflow queries")
    if(RUN_STACKOVERFLOW == 0):
        st.subheader('Reading cached data...')
    else:
        st.subheader('Loading from Stackoverflow...')
    query = 'pytorch'
    res = st.radio("Choose a framework", ('mlflow','netflix-metaflow','pytorch','tensorflow'), key='framework_selector', on_change=run_analytics)
    run_analytics()

    query = st.session_state.framework_selector

    if(RUN_STACKOVERFLOW == 1):
        SITE = StackAPI('stackoverflow')
        questions_votes = SITE.fetch('questions', min=20, tagged=query, sort='votes')
        questions_activity = SITE.fetch('questions', tagged=query, sort='activity')
        #df_votes_query, df_votes_query_dates = process_df(questions_votes, query, 'votes')
        df_activity_query, df_activity_query_dates, corr_matrix = process_df(questions_activity, query, 'activity')
        df_activity_unans = df_activity_query[df_activity_query['is_answered'] == False]
        questions = pd.json_normalize(questions_activity['items'])
        print(questions)
        questions.to_json(query + '_questions.json')
        columns = questions.columns
        try:
            questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
        except:
            raise Exception('creation_date does not exist')
        try:
            questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
        except:
            raise Exception('last_activity_date does not exist')
        try:
            questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
        except:
            raise Exception('last_edit_date does not exist')
        try:
            questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')
        except:
            print('closed_date does not exist')
        data = questions[questions['score'] >= 0]
    else:
        filename = query + '_questions.json'
        print("Reading file ",filename)
        questions = pd.read_json(filename)
        print(questions)
        print(questions.columns)
        try:
            questions['creation_date'] = questions['creation_date'] / 1
            questions['creation_date'] = questions['creation_date'].astype('datetime64[s]')
        except:
            raise Exception('creation_date does not exist')
        try:
            questions['last_activity_date'] = questions['last_activity_date'] / 1
            questions['last_activity_date'] = questions['last_activity_date'].astype('datetime64[s]')
        except:
            raise Exception('last_activity_date does not exits')
        try:
            questions['last_edit_date'] = questions['last_edit_date'] / 1
            questions['last_edit_date'] = questions['last_edit_date'].astype('datetime64[s]')
        except:
            raise Exception('last_edit_date does not exist')
        try:
            questions['closed_date'] = questions['closed_date'] / 1
            questions['closed_date'] = questions['closed_date'].astype('datetime64[s]')
        except:
            print('closed_date does not exist')
        data = questions[questions['score'] >= 0]
        corr_matrix = questions[['score', 'view_count','answer_count']].corr()

    # -------------- Header info ----------------#
    st.subheader("Data")
    headercol1, headercol2, headercol3, headercol4 = st.columns([1,1,1,1])
    num_questions = len(questions)
    headercol1.metric("Number of questions",num_questions)
    closed = len(questions[questions['is_answered'] == True])
    headercol2.metric("Closed",closed)
    headercol3.metric("Closed fraction", closed/num_questions)
    headercol4.metric("Questions with negative score", len(questions) - len(data))
    st.write(questions, width=20, height=100)

    #age = st.slider('Top n terms', key='slider', 0, 130, 25)

    st.subheader("Summarize the data")
    if st.checkbox(label='Show dataframe', key='dataframe_show'):
        st.write(questions.describe(include='all'))
    st.subheader("Correlation matrix")
    st.write(corr_matrix)

    # ---------------- Tags --------------- #
    st.subheader("Distribution of tags in questions")
    #questions['entities'] = questions['title'].apply(lambda x: apply_spacy(x))
    plot_tags(questions)

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
        get_stats(questions['view_count'])
        st.text('Bayesian inference of the parameters of the views distribution \n\n')
        with st.spinner('Inferring the parameters of the view distribution (Poisson)'):
            trace_t = Poisson_inference(questions['view_count'])
        st.success('Complete')
        summary = az.summary(trace_t)
        st.write(summary)



    # -------------------- Scores --------------- #
    st.subheader("Scores")
    col3, col4 = st.columns([1,1])
    fig = px.histogram(questions['score'], nbins=200, title="Distribution of scores for posts")
    with col3:
        st.plotly_chart(fig)

    with col4:
        get_stats(data['score'])
        st.text('Bayesian inference of the parameters of the scores distribution (ignoring the negative scores) \n\n')
        with st.spinner('Inferring the parameters of the score distribution (exponential)'):
            trace_t = Exponential_inference(data['score'])
        st.success('Complete')
        summary = az.summary(trace_t)
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
    print(questions_by_week)
    fig = px.bar(questions_by_week, title="Number of questions by month, green trend line indicates forecast for next 10 months")
    fig = get_time_series_trend(questions_by_week, ('creation_date_only', 'question_id'), fig)
    with col7:
        st.plotly_chart(fig)
    fig = px.bar(answers_and_views_by_week, title="Number of answers and views by month")
    with col8:
        st.plotly_chart(fig)
