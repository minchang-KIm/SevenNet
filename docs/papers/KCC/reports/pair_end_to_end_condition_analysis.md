# SevenNet Baseline vs Proposal-Only Condition Analysis

- datasets: 31
- median speedup: 1.010481
- geometric mean speedup: 0.857063
- wins: 18
- losses: 13

## Spearman correlations

- num_edges vs speedup: 0.593548
- avg_neighbors_directed vs speedup: 0.628629
- natoms vs speedup: 0.680878

## Simple threshold view

- num_edges >= 3000: count=20, win_rate=0.900000, median=1.013172
- num_edges < 3000: count=11, win_rate=0.000000, median=0.613472
- avg_neighbors_directed >= 40: count=17, win_rate=0.941176, median=1.013118
- avg_neighbors_directed < 40: count=14, win_rate=0.142857, median=0.614574

## Bucket summary

| density_bucket | count | mean | median | std | wins | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| large_dense | 17 | 1.0159254043080441 | 1.0131175539848845 | 0.01729061008783272 | 16 | 0.9411764705882353 |
| large_sparse | 3 | 1.0386016777821527 | 1.0545677215530402 | 0.03714224719542433 | 2 | 0.6666666666666666 |
| small_sparse | 11 | 0.6263427988085115 | 0.6134721080268288 | 0.03370053154991138 | 0 | 0.0 |

## Top wins

| dataset | baseline_mean_ms | baseline_std_ms | proposal_mean_ms | proposal_std_ms | speedup_baseline_over_proposal | natoms | num_edges | avg_neighbors_directed | density_bucket |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md22_buckyball_catcher | 104.06654751859604 | 0.1325198716404491 | 97.54667624365538 | 0.3206959747411525 | 1.066838476983625 | 148 | 6872 | 46.43243243243244 | large_dense |
| oc20_s2ef_val_ood_ads | 102.85658675711602 | 0.184249529304664 | 96.57067931257188 | 1.0102131978993734 | 1.065091262578763 | 225 | 6686 | 29.71555555555556 | large_sparse |
| oc20_s2ef_train_20m | 125.48037192318588 | 0.1176295167293528 | 118.98749540559947 | 0.2448926375646956 | 1.0545677215530402 | 225 | 8260 | 36.71111111111111 | large_sparse |
| omol25_train_neutral | 150.31542009674013 | 0.1372316907172892 | 146.151217748411 | 0.2303673656739482 | 1.0284924232071573 | 181 | 9926 | 54.83977900552486 | large_dense |
| salex_train_official | 152.02908711507916 | 0.2427471032542216 | 147.97975244000554 | 0.1531070855000776 | 1.027364113051313 | 132 | 9932 | 75.24242424242425 | large_dense |
| oc20_s2ef_val_ood_cat | 156.07421009335667 | 0.1220179870517102 | 152.63956461567432 | 0.2764868540560833 | 1.022501672396212 | 225 | 10270 | 45.644444444444446 | large_dense |
| oc20_s2ef_train_2m | 163.39146797545254 | 0.1850424872509146 | 160.5273093096912 | 0.20257005033564 | 1.0178421894572205 | 225 | 10738 | 47.72444444444445 | large_dense |
| oc20_s2ef_val_ood_both | 174.43758137524128 | 0.1487939845079797 | 171.83799822814763 | 0.1915386794660906 | 1.0151281042254825 | 223 | 11486 | 51.506726457399104 | large_dense |
| oc20_s2ef_val_id | 230.7244974654168 | 0.248853869371045 | 227.48780974652615 | 0.2721751819115173 | 1.0142279611487623 | 225 | 15230 | 67.68888888888888 | large_dense |
| salex_val_official | 200.9452628204599 | 0.0991317152068966 | 198.32205441780388 | 0.2951863918337808 | 1.0132270130538772 | 128 | 13088 | 102.25 | large_dense |

## Top losses

| dataset | baseline_mean_ms | baseline_std_ms | proposal_mean_ms | proposal_std_ms | speedup_baseline_over_proposal | natoms | num_edges | avg_neighbors_directed | density_bucket |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md22_dha | 29.379466292448345 | 0.2878755598836471 | 49.15021285414696 | 1.853752748278968 | 0.5977485057823001 | 56 | 1406 | 25.107142857142858 | small_sparse |
| md22_ac_ala3_nhme | 29.28515672683716 | 0.1294756616799604 | 48.875236068852246 | 0.3722444004612053 | 0.5991818982844838 | 42 | 1142 | 27.19047619047619 | small_sparse |
| md22_at_at | 29.31396292988211 | 0.1888826768155194 | 48.27208698261529 | 0.3103374515286291 | 0.6072652906106843 | 60 | 1312 | 21.866666666666667 | small_sparse |
| qm9_hf | 29.40622386522591 | 0.1721031691923248 | 48.07258027140051 | 0.2064122362357504 | 0.6117047119004002 | 29 | 570 | 19.655172413793103 | small_sparse |
| rmd17 | 29.476173082366586 | 0.1687289725569441 | 48.04931699763984 | 0.2657231853133969 | 0.6134566508783974 | 24 | 366 | 15.25 | small_sparse |
| iso17 | 29.38942515756935 | 0.1811283745969218 | 47.90670149959624 | 0.2432164469467641 | 0.6134721080268288 | 19 | 340 | 17.894736842105264 | small_sparse |
| spice_2023 | 30.61731008347124 | 0.2299908997658893 | 49.85837347339839 | 0.3449410194614801 | 0.6140856179314976 | 89 | 728 | 8.179775280898877 | small_sparse |
| phonondb_pbe | 29.36873459257185 | 0.1841774772784941 | 47.74914663285017 | 0.2894805130228878 | 0.6150630254901127 | 8 | 208 | 26.0 | small_sparse |
| ani1x | 31.5002522431314 | 0.3156348405716656 | 48.59277354553342 | 0.4293138233881927 | 0.6482497281949624 | 63 | 2008 | 31.873015873015877 | small_sparse |
| ani1ccx | 31.833992525935173 | 0.4398851714892866 | 48.29108465928584 | 0.205224307405854 | 0.6592105509854984 | 55 | 2042 | 37.127272727272725 | small_sparse |

## Interpretation

The proposal should be claimed as beneficial under the workload conditions that actually show positive speedup here.
In this result set, the strongest practical separator is graph size: graphs with num_edges < 3000 never win, whereas graphs with num_edges >= 3000 win in most cases.
High-neighbor workloads also align strongly with improvement, but large graph size is the more defensible first condition.
FlashTP synergy remains future work; this report is strictly about SevenNet baseline versus proposal-only.
