# ArrNorm

ArrNorm is a QGIS processing plugin for relative radiometric normalization of a target image against a reference image, using the IR-MAD algorithm to identify spectrally invariant pixels.

## Compatibility

| QGIS version | Qt version | Python bindings | Python version |
|---|---|---|---|
| QGIS 3.16+ | Qt 5 | PyQt5 | 3.9+ |
| QGIS 4.x | Qt 6 | PyQt6 | 3.9+ |

The algorithm exploits the linear and affine invariance of the Multivariate Alteration Detection (MAD) transformation to perform relative radiometric normalization via two stages [1]:

1. **IR-MAD** — Iteratively Reweighted MAD finds spectrally invariant (no-change) pixels by solving a canonical correlation problem and reweighting pixels by their chi-square no-change probability across iterations.
2. **Radcal** — A per-band orthogonal regression is fitted on the no-change pixels to derive the linear transform (intercept + slope) applied to the target.

The IR-MAD iteration stops when the maximum change in canonical correlations between successive iterations (δ) falls below the convergence threshold, or when the maximum number of iterations is reached. When the maximum is reached, the plugin automatically selects the iteration with the smallest δ as the final result. A separate no-change pixel probability threshold then controls which pixels are admitted to the per-band regression in Radcal.

[1] M. J. Canty (2014): Image Analysis, Classification and Change Detection in Remote Sensing, with Algorithms for ENVI/IDL and Python (Third Revised Edition), Taylor and Francis CRC Press.

![](docs/img/example.jpg)

<figcaption>Fig.1 - Example of a Landsat image normalization, using a multi-year average (reference) to normalize a scene. Those pixel changes are because, some remote sensing imagery pixel values are affected by different causes such as: sensor angle, sun position, clouds and seasons. Be sure that you are using the same 'style' for all layers in Qgis (use copy/paste style for that) to visualize and check the normalization visually.</figcaption>

---

> *Note:* To uninstall/update this plugin on Windows (QGIS 3.x and QGIS 4.x), due to some dlls that the plugin has, you must first deactivate, restart Qgis and finally update and activate.

## Source code

The official version control system repository of the plugin:
[https://github.com/SMByC/ArrNorm-Qgis-processing](https://github.com/SMByC/ArrNorm-Qgis-processing)

## Issue Tracker

Issues, ideas and enhancements: [https://github.com/SMByC/ArrNorm-Qgis-processing/issues](https://github.com/SMByC/ArrNorm-Qgis-processing/issues)

## About us

ArrNorm was developed, designed and implemented by the Group of Forest and Carbon Monitoring System (SMByC), operated by the Institute of Hydrology, Meteorology and Environmental Studies (IDEAM) - Colombia.

Author and developer: *Xavier C. Llano* *<xavier.corredor.llano@gmail.com>*  
Theoretical support, tester and product verification: SMByC-PDI group

## License

ArrNorm is a free/libre software and is licensed under the GNU General Public License.
