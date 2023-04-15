from db.session import session
from db.model import SelectPair, Kinetics
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

from collections import Counter


class analysis:
    def __init__(self, exp_name, round_num=(0,20), max_iters=10):
        current_round = session.query(func.max(SelectPair.round_num)).scalar()
        self.result = session.query(
            SelectPair.first,
            SelectPair.second,
            SelectPair.first_selection,
            SelectPair.sec_selection,
            SelectPair.none_selection
        ).filter(
            SelectPair.exp_name == exp_name,
            SelectPair.none_selection == 0,
            SelectPair.round_num >= 2,
            SelectPair.round_num < 22
        ).all()
        
        self.exp_name = exp_name
        self.data = self.win_time()
        self.ranks = self.bradley_terry_analysis(self.data, max_iters)
        self.final_scores = self.rank2score()
        self.final_scores = self.risk_level()


    def win_time(self):
        pair = [r for r in self.result]
        pair = sorted(pair, key=lambda x: x[1])
        pair = sorted(pair, key=lambda x: x[0])
        wins_a = [p[2] if p[2] == 0 else 1 for p in pair]
        wins_b = [p[3] if p[3] == 0 else 1 for p in pair]
        pair = [p[:2] for p in pair]

        df = pd.DataFrame(pair, columns=['Action A','Action B'])
        df['Wins A'] = wins_a
        df['Wins B'] = wins_b
        return df
        print("Columns: Excerpt A vs. Exceprt B")
        print("Wins A --> Number of times A preferred over B")
        print("Wins B ---> Number of times B preferred over A")


    def bradley_terry_analysis(text_data, max_iters=1000, error_tol=1e-3, epsilon = 1e-5):
        # Do some aggregations for convenience
        # Total wins per excerpt
        winsA = text_data.groupby('Action A').agg(sum)['Wins A'].reset_index()
        winsA.columns = ['Action', 'Wins']
        winsB = text_data.groupby('Action B').agg(sum)['Wins B'].reset_index()
        winsB.columns = ['Action', 'Wins']
        wins = pd.concat([winsA, winsB]).groupby('Action').agg(sum)['Wins']

        # Total games played between pairs
        num_games = Counter()
        for index, row in text_data.iterrows():
            key = tuple(sorted([row['Action A'], row['Action B']]))
            total = sum([row['Wins A'], row['Wins B']])
            num_games[key] += total

        # Iteratively update 'ranks' scores
        actions = sorted(list(set(text_data['Action A']) | set(text_data['Action B'])))
        ranks = pd.Series(np.ones(len(actions)) / len(actions), index=actions)

        for iters in range(max_iters):
            oldranks = ranks.copy()
            for excerpt in ranks.index:
                denom = np.sum(num_games[tuple(sorted([excerpt, p]))]
                            / (ranks[p] + ranks[excerpt])
                            for p in ranks.index if p != excerpt)
                ranks[excerpt] = 1.0 * wins[excerpt] / denom

            ranks /= sum(ranks)

            if np.sum((ranks - oldranks).abs()) < error_tol:
                break

            if ranks.min() <= 0:
                ranks = ranks + epsilon  

        if np.sum((ranks - oldranks).abs()) < error_tol:
            logging.info(" * Converged after %d iterations.", iters)
        else:
            logging.info(" * Max iterations reached (%d iters).", max_iters)


        # Note we can control scaling here. For this competiton we have -'ve and positive values on the scale
        # To reproduce the results from example; I choose to multiply the rank with x100
        ranks = ranks.sort_values(ascending=False) \
                        .apply(lambda x : x*100).round(2)

        return ranks


    def rank2score(self):
        final_scores = pd.DataFrame(self.ranks, columns=['readability']).reset_index()
        final_scores.sort_values(by='Risk score', inplace=True)
        final_scores.columns= ['Action', 'Risk score']
        return final_scores


    def risk_level(self):
        scores = self.final_scores
        # Assume that your risk score data is in a Pandas DataFrame called `data`
        quantiles = scores['Risk score'].quantile(0.9)
        risk_scores = scores[scores["Risk score"] < quantiles]['Risk score']

        # Get summary statistics
        summary_stats = risk_scores.describe()

        # Define risk ranges based on standard deviations from the mean
        high_risk_range = (summary_stats['mean'] + 2 * summary_stats['std'], float('inf'))
        middle_risk_range = (summary_stats['mean'] + summary_stats['std'], summary_stats['mean'] + 2 * summary_stats['std'])
        low_risk_range = (summary_stats['mean'] - summary_stats['std'], summary_stats['mean'] + summary_stats['std'])

        # Define the conditions for each risk level
        conditions = [    
            scores['Risk score'] < low_risk_range[0],
            scores['Risk score'].between(low_risk_range[0], low_risk_range[1]),
            scores['Risk score'].between(middle_risk_range[0], middle_risk_range[1]),
            scores['Risk score'].between(high_risk_range[0], high_risk_range[1]),
            scores['Risk score'] > high_risk_range[1]
        ]

        # Define the labels for each condition
        labels = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

        # Create a new column with the labels based on the conditions
        scores['Risk Level'] = np.select(conditions, labels)

        # Get the data for each risk level
        very_low_risk_data = scores[scores['Risk Level'] == 'Very Low']
        low_risk_data = scores[scores['Risk Level'] == 'Low']
        middle_risk_data = scores[scores['Risk Level'] == 'Middle']
        high_risk_data = scores[scores['Risk Level'] == 'High']
        very_high_risk_data = scores[scores['Risk Level'] == 'Very High']
        print(f'high_risk: {high_risk_range}')
        print(f'middle_ris: {middle_risk_range}')
        print(f'low_risk: {low_risk_range}')
        scores.to_csv(f'static/{self.exp_name}_scores.csv', sep='\t')
        return scores


    def score_distribution(self):
        fig = px.bar(scores, x='Risk score', y='Action', color='Risk Level', color_discrete_sequence=[ 'green', 'blue', 'red'],
                    title='Distribution of risk level on std-based approach', 
                    text='Risk score', width=600)
        # fig.show()
        pio.write_image(fig, f'static/{self.exp_name}_score_distribution.png', engine='kaleido')


    def class_estimate(self, level):
        score = self.final_scores
        score = score[score['Risk Level'] == level]
        actions = score['Action']
        result = session.query(Kinetics).filter(Kinetics.name.in_(actions)).all()
        labels = [r.label for r in result]
        label_class = list(set(labels))
        estimate = [(label, labels.count(label)) for label in label_class]
        estimate = pd.DataFrame(estimate, columns=['class','count'])
        estimate.sort_values(by='count', inplace=True)
        fig = px.bar(estimate, x='count', y='class',
                    title=f'{level} risk label', 
                    text='count', width=600)
        # fig.show()
        pio.write_image(fig, f'static/{self.exp_name}_{level}.png', engine='kaleido')


    def gen_chart(self):
        self.score_distribution()
        self.class_estimate('High')
        self.class_estimate('Middle')
        self.class_estimate('Low')
        self.class_estimate('Very Low')
