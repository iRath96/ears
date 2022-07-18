# EARS: Efficiency-Aware Russian Roulette and Splitting

![Teaser](/assets/teaser.jpg)

This repository contains the authors' Mitsuba implementation of the 
["Efficiency-Aware Russian Roulette and Splitting" algorithm](https://graphics.cg.uni-saarland.de/publications/rath-sig2022.html).
We have implemented our algorithm in a recursive path tracer, which can be found under [`mitsuba/src/integrators/path/recursive_path.cpp`](/ears/blob/master/mitsuba/src/integrators/path/recursive_path.cpp).

## Parameters

The following parameters are supported by our `recursive_path` integrator:

### `rrsStrategy`
Choose from the following RRS techniques:
* `"noRR"`: No Russian roulette is performed
* `"classicRR"`: Russian roulette based on throughput (Mitsuba's default for path tracing)
* `"ADRR"`: Adjoint-driven Russian roulette
* `"ADRRS"`: Adjoint-driven Russian roulette and splitting
* `"EAR"`: Efficiency-aware Russian roulette (ours)
* `"EARS"`: Efficiency-aware Russian roulette and splitting (ours)

### `splittingMin` (default 0.05)
The minimum survival probability of a path. Can be set to zero for `classicRR`.
When using learning-based techniques (ADRRS and EARS), this needs to be set to a value larger than zero (`0.05` tends to work well).
Otherwise, a faulty cache that erroneously predicts zero contribution in some region of the scene could cause bias by terminating all paths that land there.

### `splittingMax` (default 20)
The maximum number of splits performed at a point.
While higher values can help problematic regions (like caustics) converge faster, it can also lead to an explosion in the time required to render a single sample per pixel.

Note that we do not limit the breadth of rays (see _Adjoint-driven Russian Roulette and Splitting_), which in our experiments performed poorly for both ADRRS and our method.

### `rrDepth` (default 5)
The depth at which Russian roulette and splitting starts.
For `classicRR`, a value of `5` tends to perform well across scenes.
`ADRRS` requires a value of `2`, as Russian roulette and splitting should start directly after the first bounce.
Our method `EARS` requires a value of `1`, as our method benefits from performing RRS at the primary hitpoint.

### `budget` (default 30)
The time (in seconds) allocated for the render.
This does not include the time required for pre-passes.
In particular, ADRRS and EARS render albedo and surface normal denoising auxilaries before starting their render work.

Note that similar to _"Practical Path Guiding" in Production_, our rendering proceeds in iterations, the results of which are combined using inverse-variance based weighting to produce the final render.
This allows learning-based approaches (ADRRS and EARS) to collect sample statistics and update their data-structures inbetween iterations.
For more details, please consult our paper.

### Debugging AOVs
Our integrator supports outputting many insightful AOVs (e.g., average splitting factors at each depth, computation cost of each pixel, …).
Since those cause overhead, we have disabled outputting them by default.
To enable them, define the preprocessor macro `EARS_INCLUDE_AOVS` either in your scons config or in the `recursive_path.cpp` source file.

## Compilation

To compile the Mitsuba code, please follow the instructions from the [Mitsuba documentation](http://mitsuba-renderer.org/docs.html) (sections 4.1.1 through 4.6). Since our new code uses C++11 features, a slightly more recent compiler and dependencies than reported in the mitsuba documentation may be required. We only support compiling mitsuba with the [scons](https://www.scons.org) build system, but we do support Python 3.

We tested our Mitsuba code on
- macOS (Monterey, `arm64`)
- Linux (Ubuntu 22.04, `x64`)

## License

The new code introduced by this project is licensed under the GNU General Public License (Version 3). Please consult the bundled LICENSE file for the full license text.
