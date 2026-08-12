"""Prepare or apply inventory-bound retirement of development-only policy state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calo_rpd_studio.algorithms.calo.policy_retirement import (
    PolicyRetirementManager,
    load_json_document,
    write_authorization_template,
    write_inventory,
    write_plan,
)
from calo_rpd_studio.results.database import ResultDatabase


def _manager(
    arguments: argparse.Namespace,
    *,
    read_only: bool,
) -> PolicyRetirementManager:
    return PolicyRetirementManager(
        arguments.policy_store,
        ResultDatabase(arguments.database, read_only=read_only),
        source_root=arguments.source_root,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-store", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="write a read-only exact inventory")
    _add_common(inventory)
    inventory.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("plan", help="write a non-destructive removal plan")
    _add_common(plan)
    plan.add_argument("--inventory", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    authorization = commands.add_parser(
        "authorization-template",
        help="write a disabled template for a later explicit authorization",
    )
    authorization.add_argument("--plan", type=Path, required=True)
    authorization.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-empty", help="verify no removable policy state remains")
    _add_common(verify)

    execute = commands.add_parser(
        "execute",
        help="apply an exact separately authorized post-freeze removal",
    )
    _add_common(execute)
    execute.add_argument("--inventory", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--phase4-acceptance", type=Path, required=True)
    execute.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.command == "authorization-template":
        destination = write_authorization_template(
            arguments.output,
            load_json_document(arguments.plan),
        )
        print(destination)
        return 0

    manager = _manager(arguments, read_only=arguments.command != "execute")
    if arguments.command == "inventory":
        destination = write_inventory(arguments.output, manager.inventory())
        print(destination)
        return 0
    if arguments.command == "plan":
        selected = load_json_document(arguments.inventory)
        destination = write_plan(arguments.output, manager.dry_run(selected))
        print(destination)
        return 0
    if arguments.command == "verify-empty":
        inventory = manager.inventory()
        empty = (
            not inventory["removable_files"]
            and not inventory["external_existing_artifacts"]
            and not any(inventory["database"].values())
        )
        report = {
            "schema": "calo-policy-store-empty-verification-v1",
            "empty": empty,
            "inventory_sha256": inventory["inventory_sha256"],
            "removable_file_count": len(inventory["removable_files"]),
            "database_row_counts": {
                name: len(rows) for name, rows in sorted(inventory["database"].items())
            },
            "external_existing_artifacts": inventory["external_existing_artifacts"],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if empty else 1
    if arguments.command == "execute":
        receipt = manager.execute(
            inventory=load_json_document(arguments.inventory),
            plan=load_json_document(arguments.plan),
            authorization=load_json_document(arguments.authorization),
            authorization_path=arguments.authorization,
            phase4_acceptance_path=arguments.phase4_acceptance,
            receipt_path=arguments.receipt,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"Unhandled policy-retirement command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
