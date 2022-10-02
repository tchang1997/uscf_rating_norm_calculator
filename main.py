from argparse import ArgumentParser
from functools import partial
import operator as op
import warnings

import numpy as np
import pandas as pd
from rich import print
from rich.table import Table

USCF_FLOOR = 100
USCF_BONUS_MULTIPLIER = 14
NORMS = {
    1200: "4th (1200)",
    1400: "3rd (1400)",
    1600: "2nd (1600)",
    1800: "1st (1800)",
    2000: "C (2000)",
    2200: "M (2200)",
    2400: "S (2400)",
}

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--curr-rating", type=float, required=True, help="Your current rating. For unrated players, set this to zero.") # TODO: handle provisional
    psr.add_argument("--prev-games", type=int, help="# of games previously played.", default=25)
    psr.add_argument("--opp-ratings", type=float, required=True, nargs='+', help="List of opponeng ratings.")
    psr.add_argument("--score", type=float, required=True, help="Your tournament score. Must be a multiple of 0.5.")
    return psr.parse_args()

def validate(args):
    if args.score < 0:
        raise ValueError("Score must be non-negative.")
    if args.score * 2 != int(args.score * 2):
        raise ValueError("Score must be a multiple of 0.5 (loss=0, draw=0.5, win=1).")
    if args.prev_games > 0 and args.curr_rating < USCF_FLOOR:
        raise ValueError("Your rating must be above the USCF floor of 100.")
    if any(map(partial(op.gt, USCF_FLOOR), args.opp_ratings)):
        raise ValueError("All opponent ratings must be above the USCF floor of 100.")
    if args.prev_games < 0:
        raise ValueError("Must have played a positive number of games.")
    if args.prev_games < 8:
        warnings.warn("Rating calculation may be inaccurate for provisional ratings under 8 games.", RuntimeWarning)
    if args.curr_rating == 0:
        if args.prev_games > 0:
            raise ValueError("Cannot have a null rating with games played.")
        else:
            warnings.warn("Unrated players are assumed to have an implicit prior rating of 1300, assuming adult USCF membership and no prior US Chess/FIDE/CFC ratings. If you are not unrated, enter your actual rating in the `--curr-rating` argument.")
            args.curr_rating = 1300

def calculate_effective_games(rating, prev_games):
    if rating <= 2355:
        return min(prev_games, 50 / np.sqrt(0.662 + 7.39e-6 * (2569 - rating) ** 2) - 1)
    else:
        return min(prev_games, 50)

def compute_k_numerator(rating):
    if rating <= 2200:
        return 800
    elif rating < 2500:
        return 800 * (6.5 - 0.0025 * rating)
    else:
        return 200

def get_expected_wins(rating, opps):
    return 1 / (1 + np.power(10, -(rating - np.array(opps)) / 400))


def find_performance_rating(init, opps, score, tol=1e-3):
    """
        Solution to optimization problem

        minimize_r 1/2 (score - win_expectancy)^2
    """
    if score == 0:
        return min(opps) - 400
    if score == len(opps):
        return max(opps) + 400
    curr_guess = init
    diff = float('inf')
    inv_lr = 1.
    while np.abs(diff) > tol:
        we = get_expected_wins(curr_guess, opps)
        grad = -(score - we.sum()) * np.dot(we, 1-we)
        diff = 100 / inv_lr * grad
        curr_guess -= diff
    return curr_guess

def compute_special_rating(rating, opps, score, prev_games):
    pwe = np.clip((rating - opps) / 800 + 0.5, 0, 1)
    effective_games = calculate_effective_games(rating, prev_games)
    k = compute_k_numerator(rating) / (effective_games + len(opps))
    #  TODO: finish this


def compute_new_rating(rating, opps, score, prev_games):
    winning_expectancies = get_expected_wins(rating, opps)
    n_games = len(opps)
    effective_games = calculate_effective_games(rating, prev_games)
    k = compute_k_numerator(rating) / (effective_games + len(opps))
    rating_delta = k * (score - sum(winning_expectancies))
    if n_games < 3:
        new_rating = rating + rating_delta
    else:
        bonus = max(0, rating_delta - USCF_BONUS_MULTIPLIER * np.sqrt(max(n_games, 4)))
        new_rating = rating + rating_delta + bonus
    # apply floors
    new_rating = max(USCF_FLOOR, new_rating)

    # if player has an established floor
    if prev_games >= 25:
        established_floor = (rating - 200) // 100 * 100
        new_rating = max(established_floor, new_rating)
    return new_rating

def get_norms(rating, opps, score):
    expectancy_array = np.array([get_expected_wins(level, opps) for level in NORMS]).T
    norm_df = pd.DataFrame(expectancy_array, columns=NORMS.values())
    norm_df.index = norm_df.index + 1 # arrays start at 0; chess tournaments start at 1 lmao
    expected_scores = norm_df.sum(axis=0)
    expected_scores.name = "Expected Score"
    required_scores = expected_scores + 1
    required_scores[required_scores > len(opps)] = "N/A"
    required_scores.name = "Required Score"
    earned = (expected_scores + 1 <= score).map({True: "YES", False: "NO"})
    earned.name = "Norm Earned"
    info_df = pd.concat([expected_scores, required_scores, earned], axis=1).T
    norm_df = pd.concat([norm_df, info_df], axis=0)
    norm_df.index.rename('Round #', inplace=True)
    norm_df = norm_df.reindex(columns=["Opp. Ratings"] + list(norm_df.columns))
    norm_df.loc[:len(opps), "Opp. Ratings"] = opps
    norm_df["Opp. Ratings"] = norm_df["Opp. Ratings"].fillna("N/A")
    return norm_df

def df_to_table(df, n_games):
    """
        Adapted from: https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938
    """
    rich_table = Table(
            title="Expected and required results/scores by norm level",
            header_style="bold cyan",
            row_styles=[""] * n_games + ["cyan", "cyan", "magenta"],
    )
    rich_table.add_column(df.index.name, style="bold", no_wrap=True)

    for column in df.columns:
        rich_table.add_column(str(column))

    for i, (index, value_list) in enumerate(zip(df.index, df.values.tolist())):
        row = [str(index)]
        row += [str(x) for x in value_list]
        rich_table.add_row(*row, end_section=(i + 1 == n_games) or (i + 1 == n_games + 2))

    return rich_table

def print_results(performance, score, we, curr_rating, new_rating, norms_table, opps):
    rating_delta = new_rating - curr_rating
    sign_color = "red" if rating_delta < 0 else "green"
    print("[bold]Rating update:[/bold]", f"{curr_rating:.4f} -> {new_rating:.4f}", f"[bold {sign_color}]({new_rating - curr_rating:+.4f})[/bold {sign_color}]")

    print("[bold]Performance:[/bold]", f"{performance:.4f}")
    print("[bold]Performance above/below expected:[/bold]", f"[bold {sign_color}]{score - we:+.4f}[/bold {sign_color}] (Expected: {we:.4f})")
    earned = norms_table.loc["Norm Earned"]
    norm_idx = earned.where(earned=="YES").last_valid_index()
    print("Norm:", norm_idx)
    norms_table = norms_table.applymap(lambda x: np.round(x, 4) if isinstance(x, (int, float)) else x)
    print(df_to_table(norms_table, len(opps)))

if __name__ == '__main__':
    args = get_args()
    validate(args)
    new_rating = compute_new_rating(args.curr_rating, args.opp_ratings, args.score, args.prev_games)
    we = get_expected_wins(args.curr_rating, args.opp_ratings).sum()
    tpr = find_performance_rating(new_rating, args.opp_ratings, args.score)
    norms_table = get_norms(args.curr_rating, args.opp_ratings, args.score)
    print_results(tpr, args.score, we, args.curr_rating, new_rating, norms_table, args.opp_ratings)
