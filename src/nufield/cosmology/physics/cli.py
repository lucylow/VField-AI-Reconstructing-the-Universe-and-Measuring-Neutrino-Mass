"""Command-line entry points for physical cosmology diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.baryon_acoustics import approximate_sound_horizon_mpc
from nufield.cosmology.physics.neutrinos import (
    neutrino_energy_density_eV_cm3,
    neutrino_temperature_eV,
    omega_nu_massive,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.physics_validation_suite import (
    run_core_physics_validation,
    write_validation_report,
)
from nufield.cosmology.physics.theory_summary import write_background_summary


def _params(path: str | None) -> PhysicalCosmology:
    if path is None:
        return PhysicalCosmology()
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("cosmology configuration must be a YAML mapping")
    if "cosmology" in values:
        values = values["cosmology"]
    if not isinstance(values, dict):
        raise ValueError("the 'cosmology' section must be a mapping")
    return PhysicalCosmology.from_mapping(values)


def _cmd_background(args: argparse.Namespace) -> int:
    params = _params(args.config)
    redshifts = [float(item) for item in args.redshifts.split(",") if item.strip()]
    out = write_background_summary(args.output, params, redshifts)
    print(f"Wrote background summary: {out}")
    print(
        f"Derived Ωm(0)={BackgroundCosmology(params).omega_m0:.8f}; ΩDE(0)={BackgroundCosmology(params).omega_de0:.8f}"
    )
    return 0


def _cmd_neutrinos(args: argparse.Namespace) -> int:
    params = _params(args.config)
    zvals = [float(item) for item in args.redshifts.split(",") if item.strip()]
    masses = params.neutrino_masses_ev
    rows = []
    for z in zvals:
        rows.append(
            {
                "z": z,
                "T_nu_eV": np.asarray(neutrino_temperature_eV(z, params.T_cmb_K)).item(),
                "rho_species_eV_cm3": np.asarray(
                    neutrino_energy_density_eV_cm3(np.asarray(masses), z, T_cmb_K=params.T_cmb_K)
                ).tolist(),
            }
        )
    data = {
        "parameters": params.to_dict(),
        "neutrino_masses_ev": list(masses),
        "omega_nu_massive0_fd": omega_nu_massive(masses, h=params.h, T_cmb_K=params.T_cmb_K),
        "redshift_rows": rows,
        "caveats": [
            "Homogeneous thermal background only; no neutrino perturbation hierarchy.",
            "Mass-squared splittings are defaults that should be pinned to a cited global fit.",
        ],
    }
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
        print(f"Wrote neutrino summary: {args.output}")
    else:
        print(text)
    return 0


def _cmd_sound_horizon(args: argparse.Namespace) -> int:
    params = _params(args.config)
    bg = BackgroundCosmology(params)
    rs = approximate_sound_horizon_mpc(bg, args.z_end, a_min=args.a_min)
    payload = {
        "r_s_approx_mpc": rs,
        "z_end": args.z_end,
        "parameters": params.to_dict(),
        "warning": "Approximate tight-coupling integral with externally supplied z_end; not a precision r_drag.",
    }
    result = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result + "\n")
        print(f"Wrote approximate sound horizon: {args.output}")
    else:
        print(result)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    params = _params(args.config)
    redshifts = np.array([float(item) for item in args.redshifts.split(",") if item.strip()])
    report = run_core_physics_validation(params, redshifts=redshifts)
    out = write_validation_report(args.output, report)
    print(f"Wrote physics validation report: {out}")
    print(
        f"Core consistency checks: {sum(c.passed for c in report.checks)}/{len(report.checks)} passed"
    )
    print(
        "This validates internal equations only; it does not validate against CAMB/CLASS or observations."
    )
    return 0 if report.passed else 1


def _cmd_camb_power(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.camb_backend import CAMBPhysicsBackend

    params = _params(args.config)
    k = np.geomspace(args.k_min, args.k_max, args.n_k)
    spectrum = CAMBPhysicsBackend(
        nonlinear=args.nonlinear, accuracy_boost=args.accuracy_boost
    ).matter_power(params, args.z, k)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        k_mpc_inv=spectrum.k_mpc_inv,
        pk_mpc3=spectrum.pk_mpc3,
        redshift=spectrum.redshift,
        linear=spectrum.linear,
    )
    metadata = {
        "source": spectrum.source,
        "metadata": dict(spectrum.metadata),
        "units": {"k": "1/Mpc", "P": "Mpc^3"},
        "parameters": params.to_dict(),
        "nonlinear": args.nonlinear,
        "warning": "Verify CAMB settings/version and validate against independent reference outputs.",
    }
    sidecar = out.with_suffix(out.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out} and {sidecar}")
    return 0


def _write_json(path: str | None, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")
    else:
        print(text, end="")


def _cmd_sigma8(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.sigma8_validation import (
        sigma8_from_spectrum,
        sigma8_integration_convergence,
    )
    from nufield.cosmology.physics.spectrum import MatterPowerSpectrum

    with np.load(args.spectrum, allow_pickle=False) as data:
        if "k_mpc_inv" not in data or "pk_mpc3" not in data:
            raise ValueError("spectrum NPZ must contain k_mpc_inv and pk_mpc3 arrays")
        z = float(np.asarray(data["redshift"]).item()) if "redshift" in data else args.z
        linear = bool(np.asarray(data["linear"]).item()) if "linear" in data else True
        spectrum = MatterPowerSpectrum(
            data["k_mpc_inv"], data["pk_mpc3"], z, linear, str(args.spectrum)
        )
    value = sigma8_from_spectrum(spectrum, args.h)
    conv = sigma8_integration_convergence(spectrum, args.h, n_k_values=tuple(args.n_k_values))
    _write_json(
        args.output,
        {
            "sigma8": value,
            "h": args.h,
            "radius_mpc": 8 / args.h,
            "redshift": spectrum.redshift,
            "convergence": conv.to_dict(),
            "spectrum_source": spectrum.source,
            "warning": "The integral covers only the spectrum's available k support; convergence does not test missing physical tails.",
        },
    )
    return 0 if conv.converged else 1


def _cmd_topology(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.field_topology import excursion_topology, topology_curve

    field = np.load(args.field, allow_pickle=False)
    if args.thresholds:
        vals = np.asarray([float(x) for x in args.thresholds.split(",") if x.strip()])
        report = {
            "curve": [
                r.to_dict()
                for r in topology_curve(
                    field,
                    vals,
                    args.box_size,
                    threshold_kind=args.threshold_kind,
                    periodic=not args.nonperiodic,
                    occupancy=args.occupancy,
                )
            ]
        }
    else:
        report = {
            "topology": excursion_topology(
                field,
                args.threshold,
                args.box_size,
                threshold_kind=args.threshold_kind,
                periodic=not args.nonperiodic,
                occupancy=args.occupancy,
            ).to_dict()
        }
    _write_json(args.output, report)
    return 0


def _cmd_neutrino_continuity(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.neutrino_background_consistency import (
        validate_neutrino_continuity,
    )

    report = validate_neutrino_continuity(
        args.mass_ev,
        z_min=args.z_min,
        z_max=args.z_max,
        n_samples=args.n_samples,
        tolerance=args.tolerance,
    )
    _write_json(args.output, report.to_dict())
    return 0 if report.passed else 1


def _cmd_mass_prior(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.neutrino_hierarchy_prior import (
        NeutrinoMassPrior,
        draw_neutrino_mass_prior,
        validate_mass_prior_samples,
    )
    from nufield.cosmology.physics.neutrino_mass_observables import minimum_mass_sum_ev

    floor = minimum_mass_sum_ev(args.hierarchy)
    lower = floor if args.lower_sum_ev is None else args.lower_sum_ev
    prior = NeutrinoMassPrior(args.hierarchy, lower, args.upper_sum_ev, args.prior_kind)
    samples = draw_neutrino_mass_prior(prior, args.n_samples, seed=args.seed)
    _write_json(
        args.output,
        {
            "prior": prior.to_dict(),
            "sample_audit": validate_mass_prior_samples(samples, prior),
            "sample_quantiles_ev": {
                str(q): float(np.quantile(samples, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
            },
            "seed": args.seed,
            "n_samples": args.n_samples,
            "warning": "Prior choice is scientifically consequential; these draws are not a posterior or a neutrino-mass constraint.",
        },
    )
    return 0


def _cmd_growth_check(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.growth_equation_validation import validate_growth_history

    with np.load(args.history, allow_pickle=False) as d:
        needed = ("scale_factor", "growth", "H_km_s_mpc", "omega_m_clustering")
        missing = [key for key in needed if key not in d]
        if missing:
            raise ValueError(f"growth history NPZ is missing arrays: {missing}")
        report = validate_growth_history(
            *(d[key] for key in needed), drop_boundary_points=args.drop_boundary_points
        )
    _write_json(args.output, report.to_dict())
    return 0


def _cmd_gate_assess(args: argparse.Namespace) -> int:
    from nufield.cosmology.physics.physics_pipeline_gates import (
        assess_physics_campaign,
        standard_evidence_from_mapping,
        write_physics_gate_report,
    )

    mapping = json.loads(Path(args.evidence).read_text())
    if not isinstance(mapping, dict):
        raise ValueError(
            "evidence file must be a JSON object mapping gate names to evidence objects"
        )
    evidence = standard_evidence_from_mapping(mapping)
    required = (
        tuple(x for x in args.required.split(",") if x)
        if args.required
        else (
            "physics_equations",
            "solver_reference",
            "dataset_contract",
            "unit_conventions",
            "covariance_validation",
            "posterior_calibration",
            "heldout_systematics",
            "provenance",
        )
    )
    report = assess_physics_campaign(
        args.campaign,
        evidence,
        required_checks=required,
        scope=tuple(args.scope.split(",")) if args.scope else (),
    )
    out = write_physics_gate_report(args.output, report)
    print(f"Campaign gate status: {report.status}; wrote {out}")
    return 0 if report.status == "ready_for_scientific_review" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nufield-physics",
        description="Physical-cosmology background, neutrino, sound-horizon and power-spectrum tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("background", help="write H(z), density fractions, and distances")
    p.add_argument("--config", help="YAML mapping of PhysicalCosmology fields")
    p.add_argument("--redshifts", default="0,0.5,1,2,3")
    p.add_argument("--output", default="outputs/physics/background.json")
    p.set_defaults(func=_cmd_background)
    p = sub.add_parser("neutrinos", help="summarize hierarchy masses and homogeneous FD densities")
    p.add_argument("--config")
    p.add_argument("--redshifts", default="0,1,10,1000")
    p.add_argument("--output")
    p.set_defaults(func=_cmd_neutrinos)
    p = sub.add_parser(
        "sound-horizon", help="compute an explicitly approximate tight-coupling sound horizon"
    )
    p.add_argument("--config")
    p.add_argument("--z-end", type=float, default=1059.0)
    p.add_argument("--a-min", type=float, default=1e-8)
    p.add_argument("--output")
    p.set_defaults(func=_cmd_sound_horizon)
    p = sub.add_parser(
        "validate", help="run equation-level background and neutrino consistency checks"
    )
    p.add_argument("--config")
    p.add_argument("--redshifts", default="0,0.5,1,2,3,5")
    p.add_argument("--output", default="outputs/physics/validation.json")
    p.set_defaults(func=_cmd_validate)
    p = sub.add_parser("camb-power", help="use optional CAMB to compute P(k,z) in physical units")
    p.add_argument("--config")
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--k-min", type=float, default=1e-3, help="1/Mpc")
    p.add_argument("--k-max", type=float, default=1.0, help="1/Mpc")
    p.add_argument("--n-k", type=int, default=512)
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument("--accuracy-boost", type=float, default=1.0)
    p.add_argument("--output", default="outputs/physics/camb_power.npz")
    p.set_defaults(func=_cmd_camb_power)
    p = sub.add_parser(
        "sigma8", help="integrate sigma8 from a saved physical-unit power spectrum NPZ"
    )
    p.add_argument("--spectrum", required=True, help="NPZ with k_mpc_inv and pk_mpc3")
    p.add_argument("--h", type=float, default=0.674)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--n-k-values", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--output", default="outputs/physics/sigma8.json")
    p.set_defaults(func=_cmd_sigma8)
    p = sub.add_parser(
        "topology", help="compute threshold topology diagnostics on a cubic 3-D field NPZ/NPY"
    )
    p.add_argument("--field", required=True, help="NumPy .npy cubic density array")
    p.add_argument("--box-size", type=float, required=True, help="periodic box size in Mpc")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--thresholds", help="comma-separated threshold curve; supersedes --threshold")
    p.add_argument("--threshold-kind", choices=("field_value", "quantile"), default="field_value")
    p.add_argument("--occupancy", choices=("above", "below"), default="above")
    p.add_argument("--nonperiodic", action="store_true")
    p.add_argument("--output", default="outputs/physics/topology.json")
    p.set_defaults(func=_cmd_topology)
    p = sub.add_parser(
        "neutrino-continuity", help="check homogeneous neutrino rho/P continuity equation"
    )
    p.add_argument("--mass-ev", type=float, required=True)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=1e4)
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--tolerance", type=float, default=0.03)
    p.add_argument("--output", default="outputs/physics/neutrino-continuity.json")
    p.set_defaults(func=_cmd_neutrino_continuity)
    p = sub.add_parser(
        "mass-prior", help="draw and audit a hierarchy-aware neutrino mass-sum prior"
    )
    p.add_argument("--hierarchy", choices=("normal", "inverted"), default="normal")
    p.add_argument("--lower-sum-ev", type=float)
    p.add_argument("--upper-sum-ev", type=float, required=True)
    p.add_argument(
        "--prior-kind",
        choices=("uniform_sum", "uniform_lightest", "log_lightest"),
        default="uniform_sum",
    )
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/physics/mass-prior.json")
    p.set_defaults(func=_cmd_mass_prior)
    p = sub.add_parser(
        "growth-check", help="check a supplied growth history NPZ against the GR growth ODE"
    )
    p.add_argument(
        "--history",
        required=True,
        help="NPZ arrays: scale_factor,growth,H_km_s_mpc,omega_m_clustering",
    )
    p.add_argument("--drop-boundary-points", type=int, default=2)
    p.add_argument("--output", default="outputs/physics/growth-validation.json")
    p.set_defaults(func=_cmd_growth_check)
    p = sub.add_parser(
        "gate-assess", help="apply explicit publication-readiness gates from a JSON evidence map"
    )
    p.add_argument("--evidence", required=True)
    p.add_argument("--campaign", default="nufield-cosmology-campaign")
    p.add_argument("--required", help="comma-separated required gate names")
    p.add_argument("--scope", help="comma-separated analysis scope labels")
    p.add_argument("--output", default="outputs/physics/gate-report.json")
    p.set_defaults(func=_cmd_gate_assess)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"nufield-physics: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
