# ========== (c) JP Hwang 2020-04-02  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
from copy import deepcopy
import plotly.express as px

network_df = pd.read_csv('outputs/network_df.csv', index_col=0)
network_df['citations'] = network_df['citations'].fillna('')
network_df['topic_id'] = network_df['topic_id'].astype(str)

journal_ser = network_df.groupby('journal')['0'].count().sort_values(ascending=False)
top_journals = list(journal_ser.index[:5])

conn_list = list()
conn_nodes = set()
for i, row in network_df.iterrows():
    citations = row['citations']
    if len(citations) == 0:
        citations_list = []
    else:
        citations_list = citations.split(',')
        for cit in citations_list:
            tgt_topic = network_df.iloc[int(cit)]['topic_id']
            temp_dict = {
                'data': {'source': str(i), 'target': cit}, 
                'classes': tgt_topic,
                'tgt_topic': tgt_topic,
                'src_topic': i,
                'locked': True
            }
        conn_list.append(temp_dict)


def tsne_to_cyto(tsne_val):
    scale_factor = 40
    return int(scale_factor * (float(tsne_val)))


network_df = network_df.assign(highlight=1)
node_list = [
    {
        'data': {
            'id': str(i),
            'label': str(i),
            'title': network_df.iloc[i]['title'],
            'journal': network_df.iloc[i]['journal'],
            'pub_date': network_df.iloc[i]['pub_date'],
            'authors': network_df.iloc[i]['authors'],
            'cited_by': network_df.iloc[i]['cited_by'],
            'n_cites': network_df.iloc[i]['n_cites'],
            'node_size': int(np.sqrt(1+network_df.iloc[i]['n_cites']) * 15),
            'highlight': network_df.iloc[i]['highlight'],
        },
        'position': {'x': tsne_to_cyto(network_df.iloc[i]['y']), 'y': tsne_to_cyto(network_df.iloc[i]['x'])},
        'classes': network_df.iloc[i]['topic_id'],
        'selectable': True,
        'locked': True
    } for i in range(len(network_df))]


def update_node_data(node_bools):

    node_list_in = deepcopy(node_list)
    for i in range(len(network_df)):
        tempbool = node_bools[i]
        node_list_in[i]['data']['highlight'] = tempbool
        node_list_in[i]['selectable'] = False if tempbool == 0 else True

    return node_list_in


def filter_node_data(min_conns=5, journals=[], date_filter=None):
    # TODO - explore making this faster
    node_bools = np.array([1] * len(network_df))

    if min_conns is not None:
        highlight_bools = network_df.n_cites.apply(lambda x: 1 * (x >= min_conns)).values
        node_bools *= highlight_bools

    if len(journals) != 0:
        journals_bools = network_df.journal.apply(lambda x: 1 * (x in journals)).values
        node_bools *= journals_bools

    return node_bools


elm_list = node_list
# elm_list += conn_list

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

col_swatch = px.colors.qualitative.Safe
def_stylesheet = [
    {
        'selector': '.' + str(i),
        # 'style': {'background-color': col_swatch[i], 'line-color': col_swatch[i]},
        'style': {'background-color': col_swatch[i], 'line-color': col_swatch[i]}
    } for i in range(len(network_df['topic_id'].unique()))
]
def_stylesheet += [
    {
        'selector': 'node', 'style': {'width': 'data(node_size)', 'height': 'data(node_size)'}
    },
    {
        'selector': 'node[highlight = 0]',
        'style': {'background-opacity': 0.15}
    },
]

body_layout = dbc.Container([
    dbc.Row([
        dcc.Markdown(
            """
            ### CORD-19 Data Explorer Dashboard
            
            This is a demo dashboard - built using CORD-19 data TO BE COMPLETED.  
            """
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                cyto.Cytoscape(
                    id='core_19_cytoscape',
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '400px'},
                    elements=elm_list,
                    stylesheet=def_stylesheet,
                    minZoom=0.06,
                )]
            ),
            dbc.Row([
                dcc.Markdown(id='node-data', children='Click on a node to see its details here')
            ]),
        ], sm=12, md=8),
        dbc.Col([
            dcc.Markdown('Filter node data'),
            dbc.Badge("Minimum citation(s):", color="info", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='n_cites_dropdown',
                    options=[{'label': k, 'value': k} for k in range(21)],
                    value=1,
                    style={'width': '50px'}
                )],
            ),
            dbc.Badge("Journal(s) published:", color="info", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='journals_dropdown',
                    options=[{'label': i + ' (' + str(v) + ' publication(s))', 'value': i} for i, v in journal_ser.items()],
                    value=top_journals,
                    multi=True,
                    style={'width': '100%'}
                )],
            ),
        ], sm=12, md=4),
    ]),
])

app.layout = html.Div([body_layout])


@app.callback(
    Output('core_19_cytoscape', 'elements'),
    [Input('n_cites_dropdown', 'value'), Input('journals_dropdown', 'value')]
)
def filter_nodes(usr_min_cites, usr_journals_list):

    node_bools = filter_node_data(min_conns=usr_min_cites, journals=usr_journals_list, date_filter=None)
    node_list = update_node_data(node_bools)

    return node_list


@app.callback(Output('node-data', 'children'),
              [Input('core_19_cytoscape', 'selectedNodeData')])
def display_nodedata(datalist):

    print(datalist)
    txt = 'Click on a node to see its details here'
    if datalist is not None:
        print(len(datalist))
        if len(datalist) > 0:
            data = datalist[-1]
            txt = '##### Title: ' + data['title'].title()
            txt += '\n\nJournal: ' + data['journal'].title()
            txt += ', Published: ' + data['pub_date']
            txt += '\n\nAuthor(s): ' + str(data['authors'])
            txt += '\n\nCitations: ' + str(data['n_cites'])

    return txt


if __name__ == '__main__':
    app.run_server(debug=True)

