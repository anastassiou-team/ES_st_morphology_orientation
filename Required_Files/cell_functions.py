"""Additional axon processing options for all-active models"""


from neuron import h
from bmtk.simulator.bionet.pyfunction_cache import add_cell_processor
from bmtk.simulator.bionet.default_setters.cell_models import set_params_allactive


def csmc_allactive(hobj, cell, dynamics_params):
    set_params_allactive(hobj, dynamics_params)
    return hobj

add_cell_processor(csmc_allactive)
