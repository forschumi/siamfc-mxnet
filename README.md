# SiamFC - MXNet
MXNet/gluon port of the tracking method described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).

## Running the tracker
1) Set `video` from `params/params.json` to `"all"` or to a specific sequence (e.g. `"Basketball" or "Soccer"`)
1) See if you are happy with the default parameters in `params/params.json`
1) Call the main script (within an active virtualenv session)
`python run_tracker.py`

## References
```
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Joao F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850-865},
  year={2016}
}
```
 