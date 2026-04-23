from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from pathlib import Path


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_asset(payload: dict):
    from vendor.skintokens.rig_package.info.asset import Asset

    return Asset(**payload)


def _serialize_asset(asset) -> dict:
    return dict(asset.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=["export", "transfer", "load"])
    parser.add_argument("--payload-in", required=True)
    parser.add_argument("--payload-out", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    payload_in = Path(args.payload_in)
    payload_out = Path(args.payload_out)

    try:
        from vendor.skintokens.rig_package.parser.bpy import BpyParser, transfer_rigging

        payload = _load_pickle(payload_in)
        if args.op == "export":
            BpyParser.export(_deserialize_asset(payload["asset"]), payload["filepath"], **payload.get("kwargs", {}))
            response = {"status": "ok"}
        elif args.op == "transfer":
            transfer_rigging(
                source_asset=_deserialize_asset(payload["source_asset"]),
                target_path=payload["target_path"],
                export_path=payload["export_path"],
                **payload.get("kwargs", {}),
            )
            response = {"status": "ok"}
        else:
            response = _serialize_asset(BpyParser.load(payload["filepath"], **payload.get("kwargs", {})))
        _dump_pickle(payload_out, response)
        return 0
    except Exception:
        error = traceback.format_exc()
        _dump_pickle(payload_out, {"status": "error", "error": error})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
