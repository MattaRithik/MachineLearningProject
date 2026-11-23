# Data (not included)

Place these three files in this folder:

| File | Market | Rows |
|---|---|---|
| `USA.parquet` | United States | 1,604,094 |
| `GBR.parquet` | United Kingdom | 534,295 |
| `JPN.parquet` | Japan | 1,175,157 |

Source: Jensen–Kelly–Pedersen Global Factor data, WRDS table `contrib.global_factor`, January 2000 – December 2025, with
`common = 1`, `exch_main = 1`, `primary_sec = 1` and `obs_main = 1`. Each file has 165 columns: 153 JKP characteristics
and `id`, `eom`, `excntry`, `gvkey`, `permno`, `size_grp`, `me`, `ret_exc_lead1m` and the four screens.

The data are licensed through WRDS and must not be redistributed. Obtain them with your own WRDS access. Git ignores
every file in this folder except this one.
