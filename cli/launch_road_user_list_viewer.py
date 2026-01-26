from msight_vision.msight_core import RoadUserListViewerNode
from msight_core.utils import get_node_config_from_args, get_default_arg_parser

def main():
    parser = get_default_arg_parser(description="Launch Road User List Viewer Node", node_class=RoadUserListViewerNode)
    parser.add_argument("--basemap", type=str, required=True, help="Path to the basemap image")
    parser.add_argument("--show-trajectory", action='store_true', help="Flag to show trajectory")
    parser.add_argument("--show-heading", action='store_true', help="Flag to show heading")
    args = parser.parse_args()
    configs = get_node_config_from_args(args)
    # sub_topic = get_topic(redis_client, "example_fused_results")

    detection_node = RoadUserListViewerNode(
        configs,
        args.basemap,
        args.show_trajectory,
        # False,
        args.show_heading,
    )
    detection_node.spin()

if __name__ == "__main__":
    main()
