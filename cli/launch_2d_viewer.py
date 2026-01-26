from msight_vision.msight_core import DetectionResults2DViewerNode
from msight_core.utils import get_node_config_from_args, get_default_arg_parser

def main():
    parser = get_default_arg_parser(description="Launch Detection Results 2D Viewer Node", node_class=DetectionResults2DViewerNode)
    args = parser.parse_args()
    detection_node = DetectionResults2DViewerNode(
        configs=get_node_config_from_args(args)
    )
    detection_node.spin()
