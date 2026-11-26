# Numbers reported in the thesis

Runs [1, 2, 3, 4, 5, 6], warm-up [1]. Runtime and memory are the median (and, where it differs, the mean) over runs [2, 3, 4, 5, 6].

Consistency: every deterministic measure identical in every run.

## Results: efficiency

- **fastest family**
  - thesis now: of the 96 groups of three runs, EW fastest in 34, CL in 34, EF in 28
  - new: median: of 96 groups: EW 46, CL 19, EF 31  ||  mean: of 96 groups: EW 49, CL 14, EF 33
- **spread between families at the same m**
  - thesis now: at most 25%, median 8%
  - new: at most 6%, median 0%
- **spread among deterministic and the three m=1 runs**
  - thesis now: at most 29%, median 15%
  - new: at most 9%, median 4%
- **rank correlation runtime vs classes (per dataset and model, AC/ACC/SchemEX)**
  - thesis now: median -0.05
  - new: median: median 0.40  ||  mean: median 0.55
- **memory, binned against deterministic**
  - thesis now: within 2% on four datasets, 3.6% on DyLDO; BSBM 1000 larger but at most 0.15 GB
  - new: BSBM 1000 0.0% (0.00 GB); BSBM 100M 0.0% (0.00 GB); WatDiv 10M 0.0% (0.00 GB); DBpedia 3.9 0.0% (0.00 GB); DyLDO 0.0% (0.00 GB); BTC 150M 0.0% (0.00 GB)
- **SchemEX memory / AC memory**
  - thesis now: 1.2 to 1.7 times
  - new: 1.2 to 1.6
- **phase 1, EF at most**
  - thesis now: EF at most 3.8 s, CL at most 18.7 s, both on BSBM 100M
  - new: 1.1 s on BSBM 100M
- **phase 1, CL at most**
  - thesis now: EF at most 3.8 s, CL at most 18.7 s, both on BSBM 100M
  - new: 0.8 s on BSBM 100M
- **phase 1, CL / EF (five larger datasets)**
  - thesis now: 5.5 to 6.2
  - new: 0.7 to 1.1
- **phase 1, CL against the fastest phase-2 run**
  - thesis now: at most 1.8% except BSBM 1000 (0.2 s against about 4 s)
  - new: median: BSBM 1000 2.5% (0.1 s of 4.6 s); BSBM 100M 0.0% (0.8 s of 1653.5 s); WatDiv 10M 0.1% (0.1 s of 158.7 s); DBpedia 3.9 0.1% (0.3 s of 487.3 s); DyLDO 0.1% (0.3 s of 496.4 s); BTC 150M 0.1% (0.1 s of 133.9 s)  ||  mean: BSBM 1000 2.5% (0.1 s of 4.6 s); BSBM 100M 0.0% (0.8 s of 1652.2 s); WatDiv 10M 0.1% (0.1 s of 158.8 s); DBpedia 3.9 0.1% (0.3 s of 484.5 s); DyLDO 0.1% (0.3 s of 497.5 s); BTC 150M 0.1% (0.1 s of 134.3 s)
- **phase 1, collecting the mass values (new)**
  - thesis now: (not in the thesis yet)
  - new: at most 7.9 s
- **fastest family, run by run**
  - thesis now: (single run)
  - new: run 2: of 96 groups: EW 44, CL 24, EF 28 | run 3: of 96 groups: EW 49, CL 18, EF 29 | run 4: of 96 groups: EW 46, CL 21, EF 29 | run 5: of 96 groups: EW 44, CL 18, EF 34 | run 6: of 96 groups: EW 45, CL 19, EF 32

## Results: efficiency / Threats

- **SchemEX slower in every run**
  - thesis now: on five of the six datasets
  - new: on 6 of 6: BSBM 1000, BSBM 100M, WatDiv 10M, DBpedia 3.9, DyLDO, BTC 150M

## Discussion: efficiency

- **largest memory**
  - thesis now: up to 228 GB on BSBM 100M
  - new: 221 GB on BSBM 100M
- **phase 2 per mass-carrying triple**
  - thesis now: 13 to 35 us
  - new: 15 to 34 us
- **collect memory per mass-carrying triple**
  - thesis now: 1.6 to 3.4 KB
  - new: 1.6 to 3.2 KB

## Results: consistency / Threats

- **m=1 reproduces the deterministic class count**
  - thesis now: in all 54 runs of AC, ACC and SchemEX
  - new: 54 of 54

## Appendix: CC

- **additional CC classes under b0**
  - thesis now: WatDiv +9, DBpedia +169, DyLDO +4,797, BTC 150M +2, BSBM +0
  - new: BSBM 1000 +0; BSBM 100M +0; WatDiv 10M +9; DBpedia 3.9 +169; DyLDO +4,797; BTC 150M +2

## Results: compactness / Discussion / Conclusion

- **classes at m=25 (largest family) / deterministic**
  - thesis now: AC 1.3 to 7.5, ACC 1.3 to 3.1, SchemEX 1.1 to 2.2; overall 1.1 to 7.5
  - new: AC 1.3 to 7.5; ACC 1.3 to 3.1; SchemEX 1.1 to 2.2; overall 1.1 to 7.5

## Results: compactness

- **least and most growth, AC**
  - thesis now: BTC 150M least, about 1.3 for every model; BSBM 100M most under AC, 50 to 376
  - new: least BTC 150M (1.30), most BSBM 100M (50 to 376, 7.52)
- **least and most growth, ACC**
  - thesis now: BTC 150M least, about 1.3 for every model; BSBM 100M most under AC, 50 to 376
  - new: least BTC 150M (1.30), most WatDiv 10M (23,870 to 72,933, 3.06)
- **least and most growth, SchemEX**
  - thesis now: BTC 150M least, about 1.3 for every model; BSBM 100M most under AC, 50 to 376
  - new: least BSBM 1000 (1.13), most BSBM 100M (52,539 to 115,594, 2.20)
- **EW or CL gives the most classes at m=25**
  - thesis now: 17 of the 18 cases
  - new: 18 of 18
- **class count not monotone in m**
  - thesis now: CL on BSBM 100M: ACC 49,723 -> 49,011, SchemEX 71,464 -> 70,752
  - new: none

## Results: compactness / Discussion

- **classes at m=25 as a share of the vertices**
  - thesis now: largest about a fifth (SchemEX on DBpedia), most under 5%
  - new: largest 20.4% (SchemEX on DBpedia 3.9); 11 of 18 under 5%
- **EF at m=5 reproduces the deterministic partition on BSBM**
  - thesis now: yes, on both BSBM instances
  - new: BSBM 1000 AC: same; BSBM 1000 ACC: same; BSBM 1000 SchemEX: same; BSBM 100M AC: same; BSBM 100M ACC: same; BSBM 100M SchemEX: same

## Results: binning behaviour

- **non-empty bins over mu, m=5/10/25**
  - thesis now: counted over pi: EW 4/6/9 (BSBM 2/3/6); EF on BSBM 3/4/7 and 3/4/6; CL on BSBM 5/8/15 and 5/9/16; elsewhere EF 17-25 and CL 22-25 at m=25
  - new: BSBM 1000: EW 4/8/17, EF 3/4/7, CL 5/10/21 | BSBM 100M: EW 4/8/20, EF 3/4/5, CL 5/10/23 | WatDiv 10M: EW 5/10/25, EF 5/10/18, CL 5/10/25 | DBpedia 3.9: EW 5/10/25, EF 5/10/18, CL 5/10/25 | DyLDO: EW 5/10/25, EF 5/8/12, CL 5/10/25 | BTC 150M: EW 5/10/25, EF 4/6/11, CL 5/10/25
- **non-empty bins over pi, m=5/10/25**
  - thesis now: counted over pi: EW 4/6/9 (BSBM 2/3/6); EF on BSBM 3/4/7 and 3/4/6; CL on BSBM 5/8/15 and 5/9/16; elsewhere EF 17-25 and CL 22-25 at m=25
  - new: BSBM 1000: EW 2/3/6, EF 3/4/6, CL 3/7/11 | BSBM 100M: EW 2/3/6, EF 3/4/5, CL 3/6/11 | WatDiv 10M: EW 4/6/9, EF 5/9/16, CL 5/10/20 | DBpedia 3.9: EW 4/6/9, EF 5/10/17, CL 5/10/20 | DyLDO: EW 4/6/9, EF 5/7/11, CL 4/9/16 | BTC 150M: EW 4/6/9, EF 4/6/11, CL 5/10/16
- **quantisation error not monotone in m**
  - thesis now: 4 of 36 series rise from m=10 to m=25, three under CL
  - new: of 36: m=10->25 in 0 (-); m=5->10 in 0 (-)

## Results: binning behaviour / Limitations

- **quantisation error on mu above that on pi**
  - thesis now: 42 of 54; all 18 for CL (factor 1.3 to 24); 34 of 36 for the data-dependent families
  - new: 37 of 54; CL 15 of 18 (factor 0.7 to 6.3); data-dependent 28 of 36

## Results: dispersion

- **deterministic mean dispersion (x 1e-3)**
  - thesis now: 0.004 to 0.042
  - new: 0.130 to 15.845
- **share of cells deviating from the class mean, deterministic**
  - thesis now: (not in the thesis yet)
  - new: 3% to 84%
- **share of cells deviating from the class mean, EW m=25**
  - thesis now: (not in the thesis yet)
  - new: 1% to 22%
- **share of cells deviating from the class mean, CL m=25**
  - thesis now: (not in the thesis yet)
  - new: 1% to 19%
- **share of cells deviating from the class mean, EF m=25**
  - thesis now: (not in the thesis yet)
  - new: 2% to 24%
- **binned configurations above the deterministic one**
  - thesis now: none
  - new: 0
- **deterministic maximum**
  - thesis now: up to 0.83
  - new: 0.13 to 0.83
- **deterministic dispersion on WatDiv and DBpedia (x 1e-3)**
  - thesis now: SchemEX 0.020 and 0.015 against 0.029 and 0.019 for AC
  - new: WatDiv 10M: AC 13.715, ACC 12.810, SchemEX 6.550; DBpedia 3.9: AC 15.845, ACC 15.177, SchemEX 10.060
- **dispersion not monotone in m**
  - thesis now: 7 of 54 series rise from m=5 to m=10 (CL on BSBM 100M and BTC 150M, EF for SchemEX on DBpedia)
  - new: none

## Results: dispersion / Abstract / Introduction / Conclusion

- **EW m=25 reduces the dispersion by a factor of**
  - thesis now: 4 to 45 (up to 45)
  - new: 5 to 225

## Results: dispersion / Discussion / Conclusion

- **EW the lowest family at m=5**
  - thesis now: every dataset and model at m=10 and 25; at m=5 12 of 18 (exceptions CL on BSBM 100M, EF on DyLDO)
  - new: 3 of 18 (lowest: EW 3, EF 6, CL 9); exceptions BSBM 100M AC (CL), BSBM 100M ACC (CL), BSBM 100M SchemEX (CL), WatDiv 10M AC (CL), WatDiv 10M ACC (CL), WatDiv 10M SchemEX (CL), DBpedia 3.9 AC (EF), DBpedia 3.9 ACC (EF), DBpedia 3.9 SchemEX (EF), DyLDO AC (EF), DyLDO ACC (EF), DyLDO SchemEX (EF), BTC 150M AC (CL), BTC 150M ACC (CL), BTC 150M SchemEX (CL)
- **EW the lowest family at m=10**
  - thesis now: every dataset and model at m=10 and 25; at m=5 12 of 18 (exceptions CL on BSBM 100M, EF on DyLDO)
  - new: 9 of 18 (lowest: EW 9, EF 3, CL 6); exceptions DBpedia 3.9 AC (EF), DBpedia 3.9 ACC (EF), DBpedia 3.9 SchemEX (EF), DyLDO AC (CL), DyLDO ACC (CL), DyLDO SchemEX (CL), BTC 150M AC (CL), BTC 150M ACC (CL), BTC 150M SchemEX (CL)
- **EW the lowest family at m=25**
  - thesis now: every dataset and model at m=10 and 25; at m=5 12 of 18 (exceptions CL on BSBM 100M, EF on DyLDO)
  - new: 10 of 18 (lowest: EW 10, CL 8); exceptions WatDiv 10M AC (CL), WatDiv 10M ACC (CL), WatDiv 10M SchemEX (CL), DyLDO AC (CL), DyLDO ACC (CL), BTC 150M AC (CL), BTC 150M ACC (CL), BTC 150M SchemEX (CL)

## Results: dispersion / Discussion

- **EW maximum at m=5/10/25**
  - thesis now: at most 0.19, 0.09 and 0.04 (below 1/m)
  - new: 0.19, 0.09, 0.04
- **EF and CL maxima at m=25**
  - thesis now: 0.09 to 0.29 and 0.03 to 0.27 (up to 0.29)
  - new: EF 0.13 to 0.27, CL 0.03 to 0.12
- **the three models at EW m=25 (x 1e-3)**
  - thesis now: within 0.0003 on every dataset
  - new: within 0.283

## Limitations / Results

- **representative error / dispersion**
  - thesis now: 1.4 to 10 times (the case for the class-mean payload)
  - new: 1.1 to 1.8

## Appendix: summary edges

- **edges at m=25 / deterministic (any family)**
  - thesis now: 1.1 to 11.2; largest AC on BSBM 100M, smallest SchemEX on BSBM 1000
  - new: 1.0 (SchemEX EF on BSBM 1000) to 11.1 (AC EW on BSBM 100M)
- **edges per class (deterministic)**
  - thesis now: about 5 to 16, except WatDiv above 100
  - new: BSBM 1000 5 to 11; BSBM 100M 10 to 15; WatDiv 10M 106 to 208; DBpedia 3.9 8 to 11; DyLDO 5 to 8; BTC 150M 15 to 16
- **edges not monotone in m**
  - thesis now: CL on BSBM 100M for ACC and SchemEX
  - new: none
- **EF at m=5 leaves the BSBM edges unchanged**
  - thesis now: yes
  - new: BSBM 1000 AC: same; BSBM 1000 ACC: same; BSBM 1000 SchemEX: same; BSBM 100M AC: same; BSBM 100M ACC: same; BSBM 100M SchemEX: same

## Appendix: unary

- **TMH_w at EW m=25 against deterministic**
  - thesis now: at most 1% on BSBM, 6% on DBpedia and DyLDO, 14 to 28% on WatDiv, BTC 150M to about a sixth for ACC and SchemEX
  - new: BSBM 1000: AC -1%, ACC -0%, SchemEX -0% | BSBM 100M: AC -1%, ACC -0%, SchemEX -0% | WatDiv 10M: AC -14%, ACC -28%, SchemEX -26% | DBpedia 3.9: AC -1%, ACC -6%, SchemEX -5% | DyLDO: AC -1%, ACC -5%, SchemEX -3% | BTC 150M: AC -26%, ACC -82%, SchemEX -84%

## Threats to validity

- **runtime of one configuration across the measured runs**
  - thesis now: repeated runs of the same partition differ by up to 29% (estimated from det and m=1)
  - new: median 2%, at most 9% over 5 runs
