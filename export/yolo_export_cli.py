"""Command-line parser of the YOLO11 exporter."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the exporter's argument parser."""
    parser = argparse.ArgumentParser(
        description='Export YOLO models in multiple formats for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats for default models
  python export_models.py

  # Export only end2end models
  python export_models.py --formats onnx_end2end trt_end2end

  # Export specific built-in models (nano, small, medium, large, xlarge)
  python export_models.py --models nano small large

  # Export a custom trained model
  python export_models.py --custom-model /path/to/my_model.pt

  # Export custom model with specific name and batch size
  python export_models.py --custom-model /path/to/my_model.pt:custom_name:64

  # Export using a YAML configuration file
  python export_models.py --config-file /path/to/models.yaml

  # Generate Triton configs and labels for exported models
  python export_models.py --models small --generate-config --save-labels
        """,
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt', 'onnx_end2end', 'trt_end2end', 'all'],
        default=['all'],
        help='Formats to export (default: all)',
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help=(
            'Built-in models to export: nano, small, medium, large, xlarge. '
            'If not specified and no --custom-model or --config-file, exports small only.'
        ),
    )

    parser.add_argument(
        '--custom-model',
        nargs='+',
        dest='custom_models',
        metavar='PATH[:NAME][:BATCH]',
        help=(
            'Custom model(s) to export. Format: path/to/model.pt[:triton_name][:max_batch]. '
            'Examples: my_model.pt, my_model.pt:custom_name, my_model.pt:custom_name:64'
        ),
    )

    parser.add_argument(
        '--config-file',
        dest='config_file',
        metavar='PATH',
        help='YAML configuration file with model definitions (overrides --models)',
    )

    parser.add_argument(
        '--normalize-boxes',
        action='store_true',
        default=False,
        help='Output boxes in [0, 1] normalized range instead of pixel coords (recommended)',
    )

    parser.add_argument(
        '--generate-config',
        action='store_true',
        default=False,
        help='Auto-generate Triton config.pbtxt files for exported models',
    )

    parser.add_argument(
        '--overwrite-config',
        action='store_true',
        default=False,
        help='With --generate-config: replace an existing config.pbtxt that differs '
        '(default: write config.pbtxt.generated beside it)',
    )

    parser.add_argument(
        '--save-labels',
        action='store_true',
        default=False,
        help='Save class names to labels.txt in each model directory',
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available built-in models and exit',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging',
    )
    return parser
