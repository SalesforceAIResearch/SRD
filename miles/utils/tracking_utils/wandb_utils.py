import logging
import os
import secrets
import string
from copy import deepcopy

import wandb

from miles.utils.env_report import decode_env_report

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _wandb_settings(**kwargs):
    return wandb.Settings(init_timeout=300.0, **kwargs)


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 8 length string with characters -- generated locally, not via
    # wandb.util.generate_id(), which is an undocumented internal that moved/
    # vanished across wandb versions (AttributeError on some container pulls).
    if args.wandb_random_suffix:
        suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        group = args.wandb_group + "_" + suffix
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = _wandb_settings(mode="offline")
    else:
        init_kwargs["settings"] = _wandb_settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if env_report_raw := args.env_report:
        if launcher_report := decode_env_report(env_report_raw):
            output["launcher_env_report"] = launcher_report

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": _wandb_settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    # loss/* is logged in the same call as train/* (step_key="train/step"), so it
    # must share that step axis — otherwise wandb falls back to its internal global
    # _step (which increments on every wandb.log call, ~5x per rollout) and the
    # x-axis looks multiplied by ~5.
    wandb.define_metric("loss/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    # response_len/* is logged alongside rollout/* (step_key="rollout/step").
    wandb.define_metric("response_len/*", step_metric="rollout/step")
    # All skill/* metrics live in a top-level skill/ panel. The rollout-side ones
    # (length/ppl/count/response_prefix_is_skill_frac) share the rollout/step axis;
    # the train-side ones (entropy/kl) share the train/step axis.
    wandb.define_metric("skill/length_mean", step_metric="rollout/step")
    wandb.define_metric("skill/length_min", step_metric="rollout/step")
    wandb.define_metric("skill/length_max", step_metric="rollout/step")
    wandb.define_metric("skill/count", step_metric="rollout/step")
    wandb.define_metric("skill/ppl", step_metric="rollout/step")
    wandb.define_metric("skill/response_prefix_is_skill_frac", step_metric="rollout/step")
    wandb.define_metric("skill/entropy", step_metric="train/step")
    wandb.define_metric("skill/kl", step_metric="train/step")
    # skill/kl split by provenance (correct-trace self-success/blind-correct vs
    # failed-trace pitfall-condense) -- see losses.py's skill_correct_tok_mask.
    wandb.define_metric("skill/kl_correct", step_metric="train/step")
    wandb.define_metric("skill/kl_pitfall", step_metric="train/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    # domain/* = per-task-type (math/code/search) train breakdown, logged
    # alongside rollout/* (step_key="rollout/step"). Without this binding it
    # falls back to wandb's internal _step (increments per wandb.log call,
    # ~5-7x/rollout) -> jumpy x-axis (1,7,13,19,...). Bind it to rollout/step.
    wandb.define_metric("domain/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    # val-core/* and val-aux/* are logged alongside eval/* (step_key="eval/step").
    wandb.define_metric("val-core/*", step_metric="eval/step")
    wandb.define_metric("val-aux/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
