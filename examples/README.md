## CAPTAIN 3 Examples

Working example scripts are in the `examples/` folder:
- **[plot\_input\_data.py](https://github.com/captain-project/captain3preview/blob/main/examples/plot_input_data.py)** - Visualize spatial and time-varying input data
- **[train_policy.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy.py)** - Full training loop with real data
- **[run_inference.py](https://github.com/captain-project/captain3preview/blob/main/examples/run_inference.py)** - Load a trained model and perform optimization

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WJbyDA6ZnwHKKHe). 
A pre-trained model is available [here](https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2). 

The scripts generate a number of plots, included the optimal configuration of protected areas under the given policy constraints and objectives.

<img width="75%" alt="example_protection_through_time" src="https://github.com/user-attachments/assets/f8afeb71-6bc8-472a-ae29-e616d1ae272d" />


To run a script e.g.:

```bash
cd examples
# Sync dependencies and create virtual environment
uv run plot_input_data.py
```

CAPTAIN 3 can also be run from the python console. In a Terminal window, browse to the captain3preview directory and launch Python using:

```
uv run python
```

You can then import CAPTAIN using:

```python
>>> import captain as cn
>>> cn.__version__
```




## Project Structure

```
captain3preview/
├── captain/            # Main package
│   ├── agents/         # Policy network, feature extraction, rewards
│   ├── algorithms/     # Evolution strategies trainer, episode runner
│   ├── data/           # SpatialData, ExtinctionRisk classes
│   ├── environment/    # BioEnv simulation engine
│   └── utils/          # Utilities, data loading
└── examples/           # Usage examples
```

## Citation

If you use CAPTAIN v.3, please cite:

```bibtex
@software{captain_3_2026,
  title = {CAPTAIN v.3 beta: Conservation Area Prioritization Through Artificial INtelligence},
  year = {2026},
  url = {https://github.com/captain-project/}
}

@article{silvestro2022improving,
  title={Improving biodiversity protection through artificial intelligence},
  author={Silvestro, Daniele and Goria, Stefano and Sterner, Thomas and Antonelli, Alexandre},
  journal={Nature sustainability},
  volume={5},
  number={5},
  pages={415--424},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{silvestro2025using,
  title={Using artificial intelligence to optimize ecological restoration for climate and biodiversity},
  author={Silvestro, Daniele and Goria, Stefano and Rau, E-ping and Ferreira de Lima, Renato Augusto and Groom, Ben and Jacobsson, Piotr and Sterner, Thomas and Antonelli, Alexandre},
  journal={bioRxiv},
  pages={2025--01},
  year={2025},
}
```



## License

This project is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International, see [full license](https://github.com/captain-project/captain2/blob/main/CAPTAIN-License.pdf) for detail.

For commercial licensing inquiries or permission to deviate from these terms, please contact the development team.
