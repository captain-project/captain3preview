## CAPTAIN 3 Examples

Working example scripts are in the `examples/` folder:
- **[plot\_input\_data.py](https://github.com/captain-project/captain3preview/blob/main/examples/plot_input_data.py)** - Visualize spatial and time-varying input data
- **[train_policy.py](https://github.com/captain-project/captain3preview/blob/main/examples/train_policy.py)** - Full training loop with real data
- **[run_inference.py](https://github.com/captain-project/captain3preview/blob/main/examples/run_inference.py)** - Load a trained model and perform optimization

Example data can be downloaded [here](https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9). 
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
