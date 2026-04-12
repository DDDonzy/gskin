class WeightLayerItem:
    NAME_ATTR = "layerName"
    ENABLED_ATTR = "layerEnabled"
    WEIGHTS_ATTR = "layerWeightsData"
    LOCK_ATTR = "layerLockInfluences"

    __slots__ = (
        "name",
        "enabled",
        "weights",
        "lock_influences",
    )

    name: str
    enabled: bool
    weights: list
    lock_influences: list

    def __init__(
        self,
        name: str,
        enabled: bool,
        weights: list,
        lock_influences: list,
    ):
        self.name = name
        self.enabled = enabled
        self.weights = weights
        self.lock_influences = lock_influences

class WeightLayerManager:
    __slots__ = ("layers",)

    def __init__(self, layers: list[WeightLayerItem] | None = None):
        self.layers = layers or []

    def add_layer(self, layer: WeightLayerItem):
        self.layers.append(layer)


a = WeightLayerItem("base", True, [], [])
