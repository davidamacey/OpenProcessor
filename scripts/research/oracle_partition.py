"""Write the ground-truth partition (label = cluster, singleton ``noise*`` labels unassigned).

    oracle_partition.py labels.jsonl out.jsonl

Scoring it with cluster_eval.py gives the ceiling the labels themselves allow (mixed items
labelled with another class of their group still count as false merges).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    with Path(sys.argv[1]).open() as src, Path(sys.argv[2]).open('w') as dst:
        for line in src:
            row = json.loads(line)
            cid = None if str(row['label']).startswith('noise') else row['label']
            dst.write(json.dumps({'item_id': row['item_id'], 'cluster_id': cid}) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
