#!/usr/bin/env python3
"""Prepare separated V32 datasets; never evaluate a target policy."""
import argparse
import json
import shutil
from pathlib import Path

from connectomequest.annotation_v32.data import (
    hash_file, load_public_graph, prepare_hemibrain, prepare_malecns, prepare_flywire,
)

FLYWIRE_FILES = {
    "cq-flywire-v2.1.0.tsv":"30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6",
    "cq-flywire-v3.1.0.tsv":"9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
    "cq-flywire-783-connections.csv.gz":"d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9",
}


def copy_once(source, destination, expected):
    if hash_file(source)!=expected:
        raise ValueError(f"unexpected public input digest: {source}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if hash_file(destination)!=expected:
            raise ValueError(f"existing raw input differs: {destination}")
        return destination
    with open(source,"rb") as src,open(destination,"xb") as dst:
        shutil.copyfileobj(src,dst)
    if hash_file(destination)!=expected:
        raise ValueError(f"copy checksum mismatch: {destination}")
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=["hemibrain","malecns","flywire"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-dir",type=Path)
    parser.add_argument("--source-prepared",type=Path)
    parser.add_argument("--topology-dir",type=Path)
    parser.add_argument("--download-dir",type=Path,default=Path("/tmp"))
    parser.add_argument("--inputs-dir",type=Path,default=Path("inputs/v32/flywire"))
    parser.add_argument("--min-synapses",type=int,default=5)
    args=parser.parse_args()
    if args.dataset=="hemibrain":
        if args.raw_dir is None:parser.error("hemibrain requires --raw-dir")
        out=prepare_hemibrain(args.raw_dir,args.output,min_synapses=args.min_synapses)
    else:
        if args.source_prepared is None:parser.error("validation/target require --source-prepared")
        vocabulary=load_public_graph(args.source_prepared).vocabulary
        if args.dataset=="malecns":
            if args.raw_dir is None:parser.error("malecns requires --raw-dir")
            out=prepare_malecns(args.raw_dir,args.output,vocabulary=vocabulary,min_synapses=args.min_synapses,topology_dir=args.topology_dir)
        else:
            files={name:copy_once(args.download_dir/name,args.inputs_dir/name,digest) for name,digest in FLYWIRE_FILES.items()}
            out=prepare_flywire(files["cq-flywire-v2.1.0.tsv"],files["cq-flywire-v3.1.0.tsv"],
                files["cq-flywire-783-connections.csv.gz"],args.output,vocabulary=vocabulary,min_synapses=args.min_synapses)
    print(json.dumps({"prepared":str(out),"target_policy_evaluated":False}))


if __name__=="__main__":
    main()
