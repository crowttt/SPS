import imp
from utils import load_arg
from db.session import session
from db.model import SelectPair
from sqlalchemy import func
import logging
import numpy as np
import pandas as pd
from collections import Counter
from config import high_risk_class, low_risk_class


class convert:
    def __init__(self, argv=None) -> None:
        self.config = load_arg(argv.config)

    def convert(self):
        df = self.preprocess()
        ranks = self.bradley_terry_analysis(df, max_iters=1)
        scores = self.final_score(ranks)
        self.get_level(scores)


    def preprocess(self):
        result = session.query(
            SelectPair.first,
            SelectPair.second,
            SelectPair.first_selection,
            SelectPair.sec_selection,
            SelectPair.none_selection
        ).filter(
            SelectPair.exp_name == self.config.process['exp_name'],
            SelectPair.none_selection == 0,
            SelectPair.round_num <= self.config.process['round'],
        ).all()

        pair = [r for r in result]
        pair = sorted(pair, key=lambda x: x[1])
        pair = sorted(pair, key=lambda x: x[0])
        wins_a = [p[2] if p[2] == 0 else 1 for p in pair]
        wins_b = [p[3] if p[3] == 0 else 1 for p in pair]
        pair = [p[:2] for p in pair]

        df = pd.DataFrame(pair, columns=['Action A','Action B'])
        df['Wins A'] = wins_a
        df['Wins B'] = wins_b
        return df


    def bradley_terry_analysis(self, text_data, max_iters=5, error_tol=1e-3, epsilon = 1e-5):
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
                if denom <= 0:
                    denom = epsilon  
                ranks[excerpt] = 1.0 * wins[excerpt] / denom

            ranks /= sum(ranks)

            if np.sum((ranks - oldranks).abs()) < error_tol:
                break

        if np.sum((ranks - oldranks).abs()) < error_tol:
            logging.info(" * Converged after %d iterations.", iters)
        else:
            logging.info(" * Max iterations reached (%d iters).", max_iters)

        ranks = ranks.sort_values(ascending=False) \
                        .apply(lambda x : x*100).round(2)

        return ranks


    def final_score(self, ranks):
        final_scores = pd.DataFrame(ranks, columns=['readability']).reset_index()
        final_scores.columns= ['Action', 'Risk score']
        final_scores.sort_values(by='Risk score', inplace=True)
        # return final_scores
        min_value = min(final_scores['Risk score'])
        max_value = max(final_scores['Risk score'])
        normalized_data = [((x - min_value) / (max_value - min_value)) * 10 for x in final_scores['Risk score']]

        final_scores['Risk score'] = normalized_data
        return final_scores


    def get_level(self, scores):
        quantiles = scores['Risk score'].quantile(0.9)
        risk_scores = scores[scores["Risk score"] < quantiles]['Risk score']
        summary_stats = risk_scores.describe()
        high_risk_range = (summary_stats['mean'], float('inf'))
        low_risk_range = (-float('inf'), summary_stats['mean'])
        conditions = [    
            scores['Risk score'].between(low_risk_range[0], low_risk_range[1]),
            scores['Risk score'].between(high_risk_range[0], high_risk_range[1]),
        ]
        labels = ['Low','High']
        scores['Risk Level'] = np.select(conditions, labels)
        scores.to_csv(f"{self.config.process['exp_name']}_scores.csv", sep='\t')
        # return scores

