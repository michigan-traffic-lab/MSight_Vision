from msight_vision.msight_core.source import MP4FolderSourceNode
from msight_core.utils import get_node_config_from_args, get_default_arg_parser


def main():
    parser = get_default_arg_parser(
        description="Play all .mp4 files in a folder sequentially as a camera source.",
        node_class=MP4FolderSourceNode,
    )
    parser.add_argument("--folder", "-f", required=True, help="Path to folder containing .mp4 files")
    parser.add_argument("--resize-ratio", "-r", type=float, default=None, help="Optional resize ratio")
    args = parser.parse_args()
    configs = get_node_config_from_args(args)
    node = MP4FolderSourceNode(configs, folder=args.folder, resize_ratio=args.resize_ratio)
    node.spin()


if __name__ == "__main__":
    main()
