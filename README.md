# Space Background Generator

This is a tool I am developing for [my space game project](https://github.com/floomby/SpaceGame).

It uses opencl to generate large contiguous backgrounds of nebulas and stars and outputs them as a collection of png tiles.

## Usage

| Option | Description |
|--------|-------------|
| --chunkCount arg (=16) | Number of chunks per side to generate (total chunks generated is chunkCount^2) |
| --chunkDimension arg (=2048) | Size of chunks to generate |
| --outputDirectory arg (=output) | Directory to output the pngs |
| --featureFile arg (=features.json) | File containing the features to generate |
| --seed arg (=5782) | Seed for the random number generator |

See the include example `features.json` file to understand feature definition.

## Issues

* The nebulas are weak and don't look that good.
* You should be able to change the diffraction patterns used.
* It could double buffer the output buffer to get better performance.
