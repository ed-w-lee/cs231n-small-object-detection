import os
import json
import torch
import argparse
import numpy as np

category_dict = {
    "11" : "Fixed-wing Aircraft",
    "12" : "Small Aircraft",
    "13" : "Cargo Plane",
    "15" : "Helicopter",
    "17" : "Passenger Vehicle",
    "18" : "Small Car",
    "19" : "Bus",
    "20" : "Pickup Truck",
    "21" : "Utility Truck",
    "23" : "Truck",
    "24" : "Cargo Truck",
    "25" : "Truck w/Box",
    "26" : "Truck Tractor",
    "27" : "Trailer",
    "28" : "Truck w/Flatbed",
    "29" : "Truck w/Liquid",
    "32" : "Crane Truck",
    "33" : "Railway Vehicle",
    "34" : "Passenger Car",
    "35" : "Cargo Car",
    "36" : "Flat Car",
    "37" : "Tank car",
    "38" : "Locomotive",
    "40" : "Maritime Vessel",
    "41" : "Motorboat",
    "42" : "Sailboat",
    "44" : "Tugboat",
    "45" : "Barge",
    "47" : "Fishing Vessel",
    "49" : "Ferry",
    "50" : "Yacht",
    "51" : "Container Ship",
    "52" : "Oil Tanker",
    "53" : "Engineering Vehicle",
    "54" : "Tower crane",
    "55" : "Container Crane",
    "56" : "Reach Stacker",
    "57" : "Straddle Carrier",
    "59" : "Mobile Crane",
    "60" : "Dump Truck",
    "61" : "Haul Truck",
    "62" : "Scraper/Tractor",
    "63" : "Front loader/Bulldozer",
    "64" : "Excavator",
    "65" : "Cement Mixer",
    "66" : "Ground Grader",
    "71" : "Hut/Tent",
    "72" : "Shed",
    "73" : "Building",
    "74" : "Aircraft Hangar",
    "76" : "Damaged Building",
    "77" : "Facility",
    "79" : "Construction Site",
    "83" : "Vehicle Lot",
    "84" : "Helipad",
    "86" : "Storage Tank",
    "89" : "Shipping container lot",
    "91" : "Shipping Container",
    "93" : "Pylon",
    "94" : "Tower"
}

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-ids",
        help="use category ids instead of category names",
        action="store_true"
    )
    parser.add_argument(
        "--no-compile",
        help="don't create compiled dict with all results",
        action="store_true"
    )
    parser.add_argument(
        "--compiled-dir",
        help="output dir for compiled dict, if compiling",
        metavar="DIR",
        default="."
    )
    parser.add_argument(
        "results",
        help="All coco_results.pth files to json-ify",
        default=None,
        metavar="FILE",
        nargs="+"
    )
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()

    full_d = {}
    for f in args.results:
        print(f)
        x = torch.load(f)
        d = {
            "total": {}
        }
        for k,v in x.results['bbox'].items():
            if isinstance(k, np.int64):
                if args.use_ids:
                    d[int(k)] = v
                else:
                    d[category_dict[str(k)]] = v
            else:
                d['total'][k] = v
        dirname = os.path.dirname(f)
        results_for = os.path.basename(os.path.normpath(dirname))
        with open(os.path.join(dirname, "coco_results.json"), "w+") as fout:
            json.dump(d, fout)

        to_write = ""
        cats = list(d.keys())
        mets = list(d[cats[0]].keys())
        rowf = ("{}\t" * (len(mets)+1)).strip() + "\n"
        to_write += rowf.format("", *mets)
        for cat in cats:
            row = [round(d[cat][met],3) for met in mets]
            to_write += rowf.format(cat, *row)
        with open(os.path.join(dirname, "coco_results.tsv"), "w+") as fout:
            fout.write(to_write)

        full_d[results_for] = d

    if not args.no_compile and len(args.results) > 1:
        with open(os.path.join(args.compiled_dir, "full_results.json"), "w+") as fout:
            json.dump(full_d, fout)

        to_write = ""
        its = list(full_d.keys())
        cats = list(full_d[its[0]].keys())
        rowf = ("{}\t" * (len(its)+1)).strip() + "\n"
        to_write += rowf.format("", *its)
        for cat in cats:
            row = [round(full_d[it][cat]["AP50"],3) if cat in full_d[it] else -1 for it in its]
            to_write += rowf.format(cat, *row)
        with open(os.path.join(args.compiled_dir, "full_results.tsv"), "w+") as fout:
            fout.write(to_write)

