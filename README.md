# Space Background Generator

This is a tool I am developing for [my space game project](https://github.com/floomby/SpaceGame).

It uses opencl to generate large contiguous backgrounds of nebulas and stars and outputs them as a collection of png tiles.

The macro feature definition system is missing.

## Usage

| Option | Description |
|--------|-------------|
| --chunkCount arg (=2) | Number of chunks per side to generate (total chunks generated is chunkCount^2) |
| --chunkDimension arg (=2048) | Size of chunks to generate |
| --outputDirectory arg (=output) | Directory to output the pngs |
| --featureFile arg (=features.json) | File containing the feature definitions |

See the include example features.json file for an example of the feature definition file.

## Issues

Once it is generating the required features I will fix these problems.

* The computation plan is quite naive and leads to lots of swapping buffers between system and gpu ram.
* It uses lots of system memory (functionality to swap chunks out to a file on disk exist).
* It uses more threads than optimal when it starts exporting the chunks to pngs.

