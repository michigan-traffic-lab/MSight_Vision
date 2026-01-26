from msight_vision.msight_core import FuserNode
from msight_core.utils import get_node_config_from_args, get_default_arg_parser
import time

def main():
    parser = get_default_arg_parser(description="Launch Fuser Node", node_class=FuserNode)
    parser.add_argument("--fusion-config", "-fc", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--wait", "-w", type=int, default=0, help="Wait time before starting the node (in seconds)")
    args = parser.parse_args()
    if args.wait > 0:
        print(f"Waiting for {args.wait} seconds before starting the node...")
        time.sleep(args.wait)

    configs = get_node_config_from_args(args)

    detection_node = FuserNode(
        configs,
        args.fusion_config,
    )
    detection_node.spin()

if __name__ == "__main__":
    main()
