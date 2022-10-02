# USCF Norm and Rating Calculator

A command-line utility that lets you calculate USCF rating information and norms. Can compute expected results, expected wins, and other info. This isn't thoroughly tested or affiliated with the USCF in any way -- just another informal tool.

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


## Limitations

May not work with provisional ratings as well, since a special rating formula is used for computing those.  
