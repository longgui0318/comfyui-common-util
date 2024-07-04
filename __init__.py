from .layer_nodes import NODE_CLASS_MAPPINGS as layer_nodes_class, NODE_DISPLAY_NAME_MAPPINGS as layer_nodes_display

NODE_CLASS_MAPPINGS = {
    **layer_nodes_class,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **layer_nodes_display,
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
