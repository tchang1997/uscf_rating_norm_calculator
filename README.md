# USCF Norm and Rating Calculator

A command-line utility that lets you calculate USCF rating information and norms. Can compute expected results, expected wins, and other info. This isn't thoroughly tested or affiliated with the USCF in any way -- just another informal tool.

I wrote this because I wanted to know what my rating would be faster. Maybe, one day I'll write a front-end for this.

Example usage:
```
python main.py --curr-rating 1470 --opp-ratings 974 951 2471 1585 --score 3 --prev-games 25
```

Example output:
```
Rating update: 1470.0000 -> 1505.4203 (+35.4203)
Performance: 2033.3468
Performance above/below expected: +0.7590 (Expected: 2.2410)
Norm: 4th (1200)
                                  Expected and required results/scores by norm level                                  
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Round #        ┃ Opp. Ratings ┃ 4th (1200) ┃ 3rd (1400) ┃ 2nd (1600) ┃ 1st (1800) ┃ C (2000) ┃ M (2200) ┃ S (2400) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ 1              │ 974.0        │ 0.786      │ 0.9207     │ 0.9735     │ 0.9915     │ 0.9973   │ 0.9991   │ 0.9997   │
│ 2              │ 951.0        │ 0.8074     │ 0.9299     │ 0.9767     │ 0.9925     │ 0.9976   │ 0.9992   │ 0.9998   │
│ 3              │ 2471.0       │ 0.0007     │ 0.0021     │ 0.0066     │ 0.0206     │ 0.0623   │ 0.1736   │ 0.3992   │
│ 4              │ 1585.0       │ 0.0983     │ 0.2564     │ 0.5216     │ 0.7752     │ 0.916    │ 0.9718   │ 0.9909   │
├────────────────┼──────────────┼────────────┼────────────┼────────────┼────────────┼──────────┼──────────┼──────────┤
│ Expected Score │ N/A          │ 1.6924     │ 2.1091     │ 2.4784     │ 2.7797     │ 2.9732   │ 3.1438   │ 3.3896   │
│ Required Score │ N/A          │ 2.6924     │ 3.1091     │ 3.4784     │ 3.7797     │ 3.9732   │ N/A      │ N/A      │
├────────────────┼──────────────┼────────────┼────────────┼────────────┼────────────┼──────────┼──────────┼──────────┤
│ Norm Earned    │ N/A          │ YES        │ NO         │ NO         │ NO         │ NO       │ NO       │ NO       │
└────────────────┴──────────────┴────────────┴────────────┴────────────┴────────────┴──────────┴──────────┴──────────┘
```

## How does it work?

### Computing new ratings

This is approximately based on the [official rating calculation document](https://new.uschess.org/sites/default/files/media/documents/the-us-chess-rating-system-revised-september-2020.pdf) provided by the USCF for their Glicko rating system. In summary (and excluding special cases for very high ratings/players with few games played) the rating system works as follows:

1. Calculate your expected score (loss=0, draw=0.5, win=1 -- totaled through the entire tournament). This is expressed as a "win probability" based on the function
$$\text{Expected score} = \sum_{i=1}^{\text{num. rounds}} \frac{1}{1 + 10^{-(\text{Your rating} - \text{Opponent } i\text{'s rating})/400}}.$$
This function has a distinctive S-shape, and similar functions are often used to model probabilities in other predictive models. Intuitively, if you're a lot lower rated than your opponent, then $10^{\text{stuff}}$ term in the denominator is very large, and so the fraction is close to 0 (you have very little chance of winning). If you're a lot higher rated than your opponent, then the $10^{\text{stuff}}$ is very close to zero, so the whole fraction is close to 1 (you have an almost 100% chance of winning). 

2. Compute the $k$-factor. The $k$-factor is a measure of how variable/flexible your rating is. In vague terms, the more games you've played (up to a certain point), or the longer the tournament, the lower the $k$-factor. Being high-rated (>2200) also lowers the $k$-factor. For more details on how this is actually calculated, check out the [official calculation document](https://new.uschess.org/sites/default/files/media/documents/the-us-chess-rating-system-revised-september-2020.pdf).

3. How much your rating changes is simply $k * (\text{Expected score} - \text{Actual score})$. That is, "how many more/fewer points did you earn than expected" times the $k$-factor. 

### Calculating performance ratings

To get the performance rating, ideally, we'd like to find a rating such that a player with that particular rating would, on average, score the same number of points as you did. This is slightly more involved, and involves solving the minimization problem

$$\underset{r}{\text{minimize}} \quad \frac{1}{2}(\text{Your score} - \text{Expected score of a player with rating } r)^2.$$

We solve this problem using gradient descent (lol), yielding the rating value that has expected score matching your score in the tournament.

### Norm calculations

This is much simpler than it looks -- we simply calculate the expected results in every round for players with ratings 1200, 1400, ...., 2400 (i.e., the USCF norm levels) using the method described earlier, sum them up, and see if you scored at least one point above those results. If so, you got a norm at that level. If not, git gud, I guess. 

## Limitations

May not work with provisional ratings as well, since a special rating formula is used for computing those.  
