# SiamFC - MXNet
:high_brightness: MXNet/gluon :high_brightness: port of the tracking method described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html)

## Running the tracker
1) Set `root_dataset` in `hyperparams/params.json` to your dataset path
1) Set `all` from `hyperparams/params.json` to `true` or to set one or more specific sequences (e.g. `Basketball` and/or `Soccer`) in `video` with setting `all` to `false`
1) See if you are happy with the default parameters in `hyperparams/params.json`
1) Call the main script `python run_tracker.py`

## AUC (%) on OTB
| Tracker           | OTB2013       | OTB2015       |
| ----------------- | ------------- | ------------- |
| paper (3s)        | 60.8          | 58.2          |
| ours              | 63.8          | 58.8          |

## Precision (%) on OTB
| Tracker           | OTB2013       | OTB2015       |
| ----------------- | ------------- | ------------- |
| paper (3s)        | 80.9          | 77.3          |
| ours              | 82.4          | 76.0          |

## Note
Some errors in `resize` need to be fixed

## References
```
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Joao F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850--865},
  year={2016}
}
```
 