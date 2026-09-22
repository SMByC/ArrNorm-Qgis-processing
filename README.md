<p align="center">
  <img src="icons/arrnorm.svg" alt="ArrNorm icon" width="96" height="96">
</p>
<h1 align="center">ArrNorm</h1>

ArrNorm is a QGIS Processing plugin for relative radiometric normalization of multispectral remote sensing imagery. It calibrates a target image to a reference image using invariant pixels identified by IR-MAD.

Normalization relies on the **IR-MAD** algorithm (Iteratively Reweighted Multivariate Alteration Detection) to automatically identify *no-change* pixels shared by both images and use them to calibrate a per-band linear transform. [[1]](#references)

## Algorithm overview

The pipeline has three main stages:

### 1. Alignment

Before any statistics are computed the reference image is reprojected and resampled onto the target's exact pixel grid (same CRS, same spatial extent, same number of pixels) using `gdal.Warp` with bilinear resampling. This guarantees pixel-for-pixel spatial coincidence, which is a hard requirement for the IR-MAD covariance computations. If the two images already share an identical grid the step is skipped.

**Nodata masking is strongly recommended for both reference and target images when nodata is present.** Unmasked fill values can bias the calibration and introduce errors across the entire normalized image.

### 2. IR-MAD — invariant pixel detection

The Multivariate Alteration Detection (MAD) transformation finds *K* pairs of linear combinations of the spectral bands — the **MAD variates** — such that each pair is maximally different between the two dates. For *K* bands the *i*-th MAD variate is:

```
MAD_i = a_i^T · X  −  b_i^T · Y
```

where **X** and **Y** are the reference and target band vectors, and the coefficient vectors **a** and **b** are the solutions of a pair of coupled generalized eigenproblems involving the between-date cross-covariance matrix. The corresponding **canonical correlation** ρᵢ measures how similar the two images are in that combination of bands: ρᵢ → 1 means no change, ρᵢ → 0 means complete change. The variance of each MAD variate is 2(1 − ρᵢ).

Under the null hypothesis of *no change*, the standardized MAD variates

```
χ² = Σ_i  (MAD_i / √(2(1 − ρᵢ)))²
```

follow a chi-squared distribution with *K* degrees of freedom. The **no-change probability** NCP = P(χ² ≥ observed) is then used as a pixel weight for the next iteration: stable pixels get weight ≈ 1, changed pixels get weight ≈ 0.

The **iterative reweighting** loop drives the covariance statistics toward being estimated entirely from invariant ground, progressively suppressing changed pixels. Two independent thresholds and an iteration cap govern the pipeline:

- **Convergence level** τ_conv (default **0.999**): iteration stops when δ = max|ρ_new − ρ_old| falls below 1 − τ_conv (**δ < 0.001** by default). Higher values tighten the numerical stopping tolerance; this is not a confidence level.

- **Maximum iterations** (default **50**): a ceiling, not a fixed number of iterations. Converged runs stop earlier. If the limit is reached first, the result with the smallest δ is selected, but the run is not marked as converged.

- **No-change probability threshold** τ_ncp (default **0.95**): after IR-MAD, only pixels whose chi-square-based no-change score NCP exceeds τ_ncp are admitted to the per-band orthogonal regression in RadCal. Higher values select fewer pixels with higher model-based no-change scores, but may reduce calibration coverage. The 0.95 default follows Canty and Nielsen (2008) [[3]](#references).

**Why these defaults?** The 0.001 correlation-change tolerance and 50-iteration cap follow Canty's reference Python implementation [[2]](#references). This tolerance corresponds to ArrNorm's **0.999** convergence level. The papers provide supporting context: Canty and Nielsen (2008) report satisfactory convergence usually within 20–30 iterations [[3]](#references), while Nielsen (2007) describes correlation-change stopping and its data-dependent behaviour [[4]](#references). These are practical numerical defaults, not a guarantee of normalization accuracy.

### 3. RadCal — radiometric calibration

Using the no-change pixels identified by IR-MAD (those with NCP > τ_ncp), ArrNorm fits a per-band **orthogonal (total-least-squares) regression** of target onto reference:

```
Y_normalized = a + b · Y_target
```

Orthogonal regression is used because both images contain measurement noise, so minimizing residuals in both directions gives a more accurate calibration line than ordinary least squares. The coefficients are applied to the full target image to produce the normalized output.

![](example.jpg)

*Fig. 1 — Example of a Landsat image normalization using a multi-year average as reference. Pixel values are affected by sensor angle, sun position, atmospheric conditions, and seasonal variation; ArrNorm compensates for all of these. Use the same display style (copy/paste style in QGIS) across all layers when comparing before and after.*

> See also the [ArrNorm](https://github.com/SMByC/ArrNorm) cli version.

## References

[1] M. J. Canty (2014): *Image Analysis, Classification and Change Detection in Remote Sensing, with Algorithms for ENVI/IDL and Python* (Third Revised Edition). Taylor & Francis / CRC Press. https://doi.org/10.1201/b17074

The IR-MAD algorithm and the radiometric normalization procedure implemented here are described in detail in Chapter 9 of that book. The iterative reweighting scheme, the use of the chi-squared no-change probability as pixel weights, and the orthogonal regression calibration all follow Canty's formulation directly.

[2] Canty, M. J. (n.d.). *iMad.py* [Python source code]. CRC4Docker, GitHub. https://github.com/mortcanty/CRC4Docker/blob/master/src/scripts/iMad.py (retrieved September 21, 2026).

[3] Canty, M. J., & Nielsen, A. A. (2008). Automatic radiometric normalization of multitemporal satellite imagery with the iteratively re-weighted MAD transformation. *Remote Sensing of Environment, 112*(3), 1025–1036. https://doi.org/10.1016/j.rse.2007.07.013

[4] Nielsen, A. A. (2007). The regularized iteratively reweighted MAD method for change detection in multi- and hyperspectral data. *IEEE Transactions on Image Processing, 16*(2), 463–478. https://doi.org/10.1109/TIP.2006.888195

## About

ArrNorm was designed and implemented by the Forest and Carbon Monitoring System group (SMByC), operated by the Institute of Hydrology, Meteorology and Environmental Studies (IDEAM) — Colombia.

**Author and developer:** Xavier C. Llano <xavier.corredor.llano@gmail.com>  
**Theoretical support, testing and product verification:** SMByC-PDI group

## License

ArrNorm is free/libre software, licensed under the GNU General Public License version 2 (GPL-2.0-or-later).
