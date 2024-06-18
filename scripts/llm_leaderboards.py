"""
@title

@description

"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

pd.options.display.float_format = '{:.2f}'.format

NUM_SAMPLES = 50
BOOTSTRAP_ROUNDS = 100

ELO_K = 4
ELO_SCALE = 400
ELO_BASE = 10
ELO_INIT_RATING = 1000


def visualize_battle_count(battles, title, show_num_models=30):
    ptbl = pd.pivot_table(battles, index='model_a', columns='model_b', aggfunc='size', fill_value=0)
    battle_counts = ptbl + ptbl.T
    ordering = battle_counts.sum().sort_values(ascending=False).index
    ordering = ordering[:show_num_models]
    fig = px.imshow(battle_counts.loc[ordering, ordering], title=title, text_auto=True)
    fig.update_layout(
        xaxis_title='Model B',
        yaxis_title='Model A',
        xaxis_side='top', height=800, width=800,
        title_y=0.07, title_x=0.5,
        font=dict(size=10)
    )
    fig.update_traces(hovertemplate='Model A: %{y}<br>Model B: %{x}<br>Count: %{z}<extra></extra>')
    return fig


def compute_pairwise_win_fraction(battles, max_num_models=30):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles['winner'] == 'model_a'],
        index='model_a', columns='model_b', aggfunc='size', fill_value=0
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles['winner'] == 'model_b'],
        index='model_a', columns='model_b', aggfunc='size', fill_value=0
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles,
        index='model_a', columns='model_b', aggfunc='size', fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B against all other models
    row_beats_col_freq = (
            (a_win_ptbl + b_win_ptbl.T) /
            (num_battles_ptbl + num_battles_ptbl.T)
    )

    # Arrange ordering according to proportion of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def visualize_pairwise_win_fraction(battles, title, max_num_models=30):
    row_beats_col = compute_pairwise_win_fraction(battles, max_num_models)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu', text_auto='.2f', title=title)
    fig.update_layout(
        xaxis_title=' Model B: Loser',
        yaxis_title='Model A: Winner',
        xaxis_side='top', height=900, width=900,
        title_y=0.07, title_x=0.5
    )
    fig.update_traces(hovertemplate='Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>')

    return fig


def compute_online_elo(battles, k=ELO_K, scale=ELO_SCALE, base=ELO_BASE, init_rating=ELO_INIT_RATING):
    rating = defaultdict(lambda: init_rating)

    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + base ** ((rb - ra) / scale))
        eb = 1 / (1 + base ** ((ra - rb) / scale))
        if winner == 'model_a':
            sa = 1
        elif winner == 'model_b':
            sa = 0
        elif winner == 'tie' or winner == 'tie (bothbad)':
            sa = 0.5
        else:
            raise Exception(f'unexpected vote {winner}')
        rating[model_a] += k * (sa - ea)
        rating[model_b] += k * (1 - sa - eb)

    # calibrate llama-13b to 800
    delta = (800 - rating['llama-13b'])
    for model in battles['model_a'].unique():
        rating[model] += delta

    return rating


def pretty_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=['Model', 'Elo rating']).sort_values('Elo rating', ascending=False).reset_index(drop=True)
    # df['Elo rating'] = (df['Elo rating'] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def pretty_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=['Model', column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def compute_mle_elo(df, scale=ELO_SCALE, base=ELO_BASE, init_rating=ELO_INIT_RATING, sample_weight=None):
    ptbl_a_win = pd.pivot_table(
        df[df['winner'] == 'model_a'],
        index='model_a',
        columns='model_b',
        aggfunc='size',
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df['winner'].isin(['tie', 'tie (bothbad)'])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df['winner'].isin(['tie', 'tie (bothbad)'])],
            index='model_a',
            columns='model_b',
            aggfunc='size',
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df['winner'] == 'model_b'],
        index='model_a',
        columns='model_b',
        aggfunc='size',
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(base)
            X[cur_row, models[m_b]] = -math.log(base)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(base)
            X[cur_row + 1, models[m_b]] = -math.log(base)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating
    if 'mixtral-8x7b-instruct-v0.1' in models.index:
        elo_scores += 1114 - elo_scores[models['mixtral-8x7b-instruct-v0.1']]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc='bootstrap'):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))).reset_index(names='model').sort_values('rating', ascending=False)
    bars['error_y'] = bars['upper'] - bars['rating']
    bars['error_y_minus'] = bars['rating'] - bars['lower']
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x='model', y='rating', error_y='error_y',
                     error_y_minus='error_y_minus', text='rating_rounded',
                     title=title)
    fig.update_layout(xaxis_title='Model', yaxis_title='Rating',
                      height=600)
    return fig


def predict_win_rate(elo_ratings, scale=ELO_SCALE, base=ELO_BASE, init_rating=ELO_INIT_RATING):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + base ** ((elo_ratings[b] - elo_ratings[a]) / scale))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = 'model_a'
    df.columns.name = 'model_b'
    return df.T


def sample_groups(grp, n_per_battle):
    return grp.sample(n_per_battle, replace=True)


def sample_battle_even(battles, n_per_battle):
    groups = battles.groupby(['model_a', 'model_b'], as_index=False)
    resampled = (groups.apply(lambda grp: sample_groups(grp, n_per_battle), include_groups=False).reset_index(drop=True))
    return resampled


# Sampling Battles Evenly
def get_bootstrap_even_sample(battles, n_per_battle, func_compute_elo, num_round=BOOTSTRAP_ROUNDS):
    rows = []
    for n in tqdm(range(num_round), desc='sampling battles evenly'):
        resampled = sample_battle_even(battles, n_per_battle)
        rows.append(func_compute_elo(resampled))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def visualize_battles(battles):
    fig = px.bar(battles['winner'].value_counts(), title='Counts of Battle Outcomes', text_auto=True, height=400)
    fig.update_layout(xaxis_title='Battle Outcome', yaxis_title='Count', showlegend=False)
    fig.show()

    battles_no_ties = battles[~battles['winner'].str.contains('tie')]
    fig = px.bar(pd.concat([battles['model_a'], battles['model_b']]).value_counts(), title='Battle Count for Each Model', text_auto=True)
    fig.update_layout(xaxis_title='model', yaxis_title='Battle Count', height=400, showlegend=False)
    fig.show()

    fig = visualize_battle_count(battles, title='Battle Count of Each Combination of Models', show_num_models=30)
    fig.show()

    fig = visualize_battle_count(battles_no_ties, 'Battle Count for Each Combination of Models (without Ties)')
    fig.show()

    fig = visualize_battle_count(battles[battles['winner'].str.contains('tie')], 'Tie Count for Each Combination of Models')
    fig.show()

    lang_count = battles['language'].value_counts()
    lang_count = lang_count.drop(index=('unknown'))

    topk = 15
    fig = px.bar(lang_count.head(topk), title=f'Battle Counts for the Top {topk} Languages', text_auto=True, height=400, log_y=True)
    fig.update_layout(xaxis_title='Language', yaxis_title='Count', showlegend=False)
    fig.show()

    fig = px.histogram(battles['turn'], title=f'Number of Conversation Turns', text_auto=True, height=400, log_y=True)
    fig.update_layout(xaxis_title='Turns', yaxis_title='Count', showlegend=False)
    fig.show()

    fig = visualize_pairwise_win_fraction(battles_no_ties, title='Fraction of Model A Wins for All Non-tied A vs. B Battles')
    fig.show()

    row_beats_col_freq = compute_pairwise_win_fraction(battles_no_ties)
    fig = px.bar(
        row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
        title='Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)',
        text_auto='.2f'
    )
    fig.update_layout(yaxis_title='Average Win Rate', xaxis_title='Model', showlegend=False)
    fig.show()

    online_elo_ratings = compute_online_elo(battles)
    pretty_print_model_ratings(online_elo_ratings)

    elo_mle_ratings_reverse = compute_online_elo(battles.iloc[::-1])
    pretty_print_two_ratings(online_elo_ratings, elo_mle_ratings_reverse, column_names=['Elo rating', 'Elo rating with reverse order'])

    elo_mle_ratings = compute_mle_elo(battles)
    pretty_print_model_ratings(elo_mle_ratings)
    return


def bootstrap_battles(battles):
    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, BOOTSTRAP_ROUNDS)

    fig = visualize_bootstrap_scores(bootstrap_elo_lu, 'Bootstrap of MLE Elo Rating Estimates')
    fig.show()

    np.random.seed(42)
    bootstrap_online_elo = get_bootstrap_result(battles, compute_online_elo, BOOTSTRAP_ROUNDS)
    pretty_print_two_ratings(bootstrap_elo_lu.quantile(.5), bootstrap_online_elo.quantile(.5), column_names=['Bootstrap Median of MLE Elo', 'Bootstrap Median of Online Elo'])
    fig = visualize_bootstrap_scores(bootstrap_online_elo, 'Bootstrap of Online Elo Rating Estimates')
    fig.show()

    win_rate = predict_win_rate(dict(bootstrap_elo_lu.quantile(0.5)))
    ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
    ordered_models = ordered_models[:30]
    fig = px.imshow(
        win_rate.loc[ordered_models, ordered_models],
        color_continuous_scale='RdBu', text_auto='.2f',
        title='Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle'
    )
    fig.update_layout(
        xaxis_title='Model B',
        yaxis_title='Model A',
        xaxis_side='top', height=900, width=900,
        title_y=0.07, title_x=0.5
    )
    fig.update_traces(hovertemplate='Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>')
    fig.show()

    battles_even = sample_battle_even(battles, NUM_SAMPLES)
    print(pd.pivot_table(battles_even, index='model_a', columns='model_b', aggfunc='size', fill_value=0).head())

    print('number of samples per battle pair:', NUM_SAMPLES)
    bootstrap_even_lu = get_bootstrap_even_sample(battles, NUM_SAMPLES, compute_mle_elo, num_round=100)
    fig = visualize_bootstrap_scores(bootstrap_even_lu, f'Bootstrap of MLE Elo Estimates - Even sample')
    fig.show()
    return


def language_battles(battles):
    english_only_battles = battles[battles['language'] == 'English']
    elo_ratings = compute_mle_elo(english_only_battles)
    print(pd.DataFrame(elo_ratings).head())

    non_english_battles = battles[battles['language'] != 'English']
    elo_ratings = compute_mle_elo(non_english_battles)
    print(pd.DataFrame(elo_ratings).head())
    return


def main(main_args):
    local_model_name = 'local_file_name.json'
    if not (Path(local_model_name).exists() and Path(local_model_name).is_file()):
        url = 'https://storage.googleapis.com/arena_external_data/public/clean_battle_20240606_public.json'
        response = requests.get(url)
        battles = pd.DataFrame(response.json())

        with open(local_model_name, 'wb') as model_file:
            model_file.write(response.content)
    else:
        with open(local_model_name, 'r', encoding='utf8') as file:
            battles = pd.read_json(file)

    battles = battles.sort_values(ascending=True, by=['tstamp'])
    print(battles.head())

    # we use anony battles only for leaderboard
    battles = battles[battles['anony'] == True]

    # we de-duplicate top 0.1% redundant prompts
    # see https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication
    initial_size = len(battles)
    print(f'Before dedup: {initial_size}')
    battles = battles[battles['dedup_tag'].apply(lambda x: x.get('sampled', False))]
    final_size = len(battles)
    print(f'After dedup: {final_size}')

    # visualize_battles(battles)
    bootstrap_battles(battles)
    # language_battles(battles)

    user_in = input('Press anything to close...')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
