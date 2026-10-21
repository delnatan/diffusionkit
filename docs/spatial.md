# spatial: classes of D on the track graph

`diffusionkit.spatial` groups tracks by where they are and how fast they move, and hands the groups to the
population models that already exist. It reads `gridpost`'s kept likelihoods and the track centroids; it refits
no track and computes no new likelihood.

```python
from concurrent.futures import ThreadPoolExecutor
from diffusionkit import gridpost, fbmpost, spatial

d = gridpost.analyze_tracks(tracks, acq, keep_likelihoods=True)
with ThreadPoolExecutor() as pool:
    s = spatial.analyze(d, spatial.centroids(tracks), map_fn=pool.map)

s.selection          # score against C: pick C at the elbow (the suggested one is marked)
s.null.p             # is there a spatial pattern? (neighbour gain against shuffled positions)
s.columns()          # per-track columns, join on track_id
s.report()           # per class: counts only

lik = d.likelihoods
for c, ids in s.groups().items():                     # each class's tracks, as its neighbours assign them
    rows = np.isin(lik.track_ids, ids)
    pop_D = gridpost.deconvolve(lik.loglik_D[rows], d.options.u_D())          # the class's D population
    # and alpha: fbmpost.deconvolve_tracks(a.likelihoods.rows(np.isin(a.likelihoods.track_ids, ids)))
```

## Model

- **Graph.** Each track is joined to its k = 8 nearest tracks by centroid, made symmetric (i ~ j if either lists
  the other). A kNN graph adapts to uneven density.
- **Classes.** C normal densities in ln D. Track i's evidence for class c is its D likelihood integrated against
  the class density. Classes are numbered 1..C in increasing order of D and have no names.
- **Spatial prior (hidden Potts).** P(z) ∝ exp(beta * S(z)), S the number of edges whose tracks share a class, so
  P(class c | neighbours) ∝ exp(beta * number of neighbours in c). There are no class weights. beta = 0 is the
  plain mixture with equal weights.
- **Fit.** EM with a mean-field E-step (`spatial.potts.fit`). beta's M-step is exact: it is the beta at which the
  prior expects as many agreeing edges as the fit does, E_beta[S] = E_q[S]. E_beta[S] depends only on the graph
  and C, so it is tabulated once per movie by Swendsen-Wang (`potts.agreement_table`, a few seconds per C) and
  each M-step is a lookup. The log partition function in the exact M-step is what holds beta back; surrogates
  without it let beta run into the Potts model's ordered phase, where every track takes its neighbours' class.
- **Starts.** The C-class fit starts from the plain mixture and from the (C - 1)-class fit with each class split
  in two, and keeps the best score. Equal weights alone can split a large class and merge two small ones.

The class densities only assign tracks. They are not reported, and they are not a model of what a class looks
like: that is the class's population, deconvolved from its tracks' likelihoods with the same smoothing and bands
as every other population (`gridpost.deconvolve`, `fbmpost.deconvolve_tracks`).

## How many classes

C = 1..`max_classes` are fitted, each scored by the neighbour-predictive log-likelihood: each track's data under
the class probabilities its neighbours predict. `selection` lists the score and its gain over C - 1. The
suggested C is the last one whose gain clears `gain_per_track` (0.02 nats per track) counting up from 1; it is
a default for batches, and the plot of score against C is what a person should choose from. C = 1 is a valid
answer. `SpatialOptions(n_classes=...)` overrides the suggestion.

The score is optimistic for large C (the classes were fitted to the same tracks), so the gains of extra classes
read high.

## Is there a pattern

The tracks' positions are shuffled `n_shuffles` times (99) and refitted along the same path as the observed fit.
On a few thousand tracks that is a few minutes; `map_fn` runs the shuffles on a pool.
Shuffling keeps every track's data and destroys only the arrangement, so it is the null with no spatial pattern.

The statistic is the **neighbour gain**: the fit's neighbour-predictive score minus the plain mixture's, i.e. how
many nats the neighbours add. `null.p` = (1 + shuffles with a gain at least the observed) / (1 + n_shuffles);
`patterned` is p <= `p_pattern` (0.05). beta is reported with each shuffle. A shuffle permutes the tracks over the
same point set, so its graph is the observed one with its nodes relabelled and the agreement tables serve it too.

With no pattern, the classes are a plain mixture and a class map says nothing; the caller should not draw one.

## Per-class populations without circularity

`groups()` puts each track in the class its **neighbours** give it (`spatial_class_neighbours`), its own data left
out. A track's estimates of D and alpha have correlated errors (strongly so with noise and blur), so grouping
tracks by their own D and then reading their alpha would create an alpha difference from those errors.
Grouping by the neighbours makes a track's class independent of its own estimation errors.

## Outputs

`columns()`, one row per track (gridpost "ok" tracks with a position):

| column | meaning |
| --- | --- |
| `spatial_class` | most probable class, 1..C, from the track's own data and its neighbours |
| `spatial_p` | that class's probability |
| `spatial_class_neighbours` | most probable class from the neighbours alone: the grouping for populations |
| `spatial_p_neighbours` | that class's probability. Disagreeing classes mark a track unlike its surroundings |
| `D_spatial_mean_um2_s`, `D_spatial_lo_um2_s`, `D_spatial_hi_um2_s` | the track's D posterior with the class mixture as its prior; borrowed from its class, so never in place of the gridpost columns |

`report()`, one row per class: `n_tracks` and `n_detections` in its group, `n_assigned` (tracks whose
`spatial_class` it is), `n_confident` (of those, `spatial_p` > 0.9). Counts are tracks, not particles.

`field(xy)` gives the class probabilities at any point from its k nearest tracks, and the distance to the k-th of
them; a map should leave out points whose k-th track is far (twice `typical_spacing_um()` is a reasonable cut),
since tracks fill only the space the particles can reach.

## Saving a run

`record()` is the run as plain JSON values (options, selection, the classes of the C used, the null, the
verdict, the typical spacing, the report) and `probabilities()` one row per track (`track_id`, `x_um`, `y_um`,
`q_1`..`q_C`). `spatial.restore(likelihoods, u_D, record, probabilities)` rebuilds the analysis from them and the
same gridpost likelihoods, with only the fit of the C used. A record saved by the earlier model (with class weights,
no `model` key) is refused: run the classes again.

## Not here

- No per-class shared D or alpha. A summed likelihood's interval is counting precision under one value per class
  and reads far tighter than the class's spread.
- No Kriging and no Gaussian process. Kriging per-track values found no field on real cells, and Kriging values
  that already borrow from neighbours counts the neighbours twice.
- No classes across movies. Each movie is fitted on its own; matching class numbers across movies is the user's
  call, from the classes' populations.
