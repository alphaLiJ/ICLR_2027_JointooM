"""Optimized resident inputs using an isolated warp kernel and valid prefixes."""
import bootstrap
from expert.fixed_magat_plus_runtime import PyGBatchBuilder
from paper_config_compare import PaperCUDA


class PrefixBatchBuilder(PyGBatchBuilder):
    @staticmethod
    def _materialize_valid_edges(edge_index_storage, edge_attr_storage, num_edges):
        # Dynamic PyG input shapes already require host-visible sizes. Read one
        # scalar rather than scan the full capacity twice through boolean masks.
        # This 8-byte device-to-host metadata read is included in all timings.
        count=int(num_edges.item())
        return (edge_index_storage.narrow(1,0,count).contiguous(),
                edge_attr_storage.narrow(0,0,count))


class OptimizedPaperCUDA(PaperCUDA):
    def __init__(self,batch,kernel='warp_cooperative'):
        super().__init__(batch)
        self.builder=PrefixBatchBuilder()
        self.sim.pyg_builder_mode='local_gather'
        self.sim.pyg_local_gather_impl=kernel

    def build(self):
        if self.needs_energy:
            self.sim.update_derived_state();self.needs_energy=False
        self.sim.build_magat_plus_inputs()
        # Materialize here so the fixed model does not fall back to its original
        # capacity-wide mask helper when edge_index is absent.
        return self.builder.view_stateful_outputs(self.sim,materialize_edges=True)
