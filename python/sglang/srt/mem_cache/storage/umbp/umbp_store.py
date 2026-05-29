"""UMBPStore — HiCache L3 storage backend using UMBP (local DRAM + SSD).

Follows the same pattern as MooncakeStore:
- Zero-copy v1 interface (batch_get_v1 / batch_set_v1)
- Uses mem_pool_host.get_page_buffer_meta() for pointer/size extraction
- Key suffix generation per TP rank / PP rank
"""

import atexit
import json
import logging
import os
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Stable per-process token that survives fork-after-import but is unique
# across containers whose PID namespaces may alias the same PID numbers.
_PROCESS_INSTANCE_TOKEN = uuid.uuid4().hex[:8]


def _import_umbp_client():
    """Import UMBPClient from mori.umbp (requires mori built with BUILD_UMBP=ON)."""
    import mori.umbp as umbp_mod

    UMBPClient = umbp_mod.UMBPClient
    UMBPConfig = umbp_mod.UMBPConfig
    UMBPRole = umbp_mod.UMBPRole
    UMBPIoBackend = getattr(umbp_mod, "UMBPIoBackend", None)
    UMBPDurabilityMode = getattr(umbp_mod, "UMBPDurabilityMode", None)
    UMBPDistributedConfig = getattr(umbp_mod, "UMBPDistributedConfig", None)

    return (
        UMBPClient,
        UMBPConfig,
        UMBPRole,
        UMBPIoBackend,
        UMBPDurabilityMode,
        UMBPDistributedConfig,
    )


def _optional_env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else None


def _optional_env_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None and value != "" else None


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _parse_tag_values(value: Any) -> List[str]:
    if value is None:
        return []

    def _clean(items):
        return [str(item).strip() for item in items if str(item).strip()]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return _clean(part for part in stripped.split(","))

    if isinstance(value, (list, tuple, set)):
        return _clean(value)

    return _clean([value])


def _default_node_address() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _select_rank_config_value(
    value: Any,
    rank_index: int,
    field_name: str,
    cast_type,
    auto_increment_scalar: bool = False,
):
    if value is None:
        raise ValueError(f"{field_name} must not be None")

    candidates = value
    if isinstance(value, str) and "," in value:
        candidates = [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(candidates, (list, tuple)):
        if not candidates:
            raise ValueError(f"{field_name} must not be empty")
        if rank_index >= len(candidates):
            raise ValueError(
                f"{field_name} has {len(candidates)} entries, but rank_index={rank_index}"
            )
        return cast_type(candidates[rank_index])

    selected = cast_type(candidates)
    if auto_increment_scalar:
        selected = cast_type(selected + rank_index)
    return selected


class UMBPStore(HiCacheStorage):
    """Local DRAM+SSD storage backend for HiCache L3 caching.

    Compatible with the zero-copy v1 interface used by CacheController.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        (
            UMBPClient,
            UMBPConfig,
            UMBPRole,
            UMBPIoBackend,
            UMBPDurabilityMode,
            UMBPDistributedConfig,
        ) = _import_umbp_client()

        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
            self.tp_size = storage_config.tp_size
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1
            self.tp_size = 1

        cfg = UMBPConfig.from_environment()
        # UMBPStore owns role selection explicitly. Do not inherit LOCAL_RANK /
        # UMBP_ROLE-based multi-process defaults from mori here, otherwise
        # ordinary multi-rank sglang runs can accidentally become follower-only
        # and skip writes.
        cfg.role = UMBPRole.Standalone
        extra = getattr(storage_config, "extra_config", None) or {}
        explicit_tenant_id = (
            os.getenv("UMBP_SPDK_PROXY_TENANT_ID") is not None
            or "spdk_proxy_tenant_id" in extra
        )
        tenant_id_base = (
            int(extra["spdk_proxy_tenant_id_base"])
            if "spdk_proxy_tenant_id_base" in extra
            else _optional_env_int("UMBP_SPDK_PROXY_TENANT_ID_BASE")
        )
        dp_rank_hint = _optional_env_int("SGLANG_DP_RANK")
        dp_size_hint = _optional_env_int("SGLANG_DP_SIZE")
        local_rank_hint = _optional_env_int("LOCAL_RANK")

        if dp_rank_hint is None:
            try:
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_rank,
                    get_attention_dp_size,
                    is_dp_attention_enabled,
                )

                if is_dp_attention_enabled():
                    dp_rank_hint = get_attention_dp_rank()
                    dp_size_hint = get_attention_dp_size()
            except (ImportError, AssertionError):
                pass

        if local_rank_hint is not None:
            unique_rank = local_rank_hint
        else:
            base_rank = dp_rank_hint if dp_rank_hint is not None else 0
            unique_rank = ((base_rank * max(self.pp_size, 1)) + self.pp_rank) * max(
                self.tp_size, 1
            ) + self.local_rank

        # Load settings from extra_config if available
        if "dram_capacity_bytes" in extra:
            cfg.dram.capacity_bytes = int(extra["dram_capacity_bytes"])
        if "ssd_enabled" in extra:
            cfg.ssd.enabled = bool(extra["ssd_enabled"])
        if "ssd_storage_dir" in extra:
            cfg.ssd.storage_dir = str(extra["ssd_storage_dir"])
        if "ssd_capacity_bytes" in extra:
            cfg.ssd.capacity_bytes = int(extra["ssd_capacity_bytes"])
        if "copy_to_ssd_async" in extra:
            cfg.copy_pipeline.async_enabled = bool(extra["copy_to_ssd_async"])
        if "copy_to_ssd_queue_depth" in extra:
            cfg.copy_pipeline.queue_depth = int(extra["copy_to_ssd_queue_depth"])
        if "ssd_segment_size_bytes" in extra:
            cfg.ssd.segment_size_bytes = int(extra["ssd_segment_size_bytes"])
        if "ssd_batch_max_ops" in extra:
            cfg.copy_pipeline.batch_max_ops = int(extra["ssd_batch_max_ops"])
        if "ssd_queue_depth" in extra:
            cfg.ssd.io.queue_depth = int(extra["ssd_queue_depth"])
        if "ssd_writer_threads" in extra:
            cfg.copy_pipeline.worker_threads = int(extra["ssd_writer_threads"])
        if "ssd_enable_background_gc" in extra:
            cfg.ssd.durability.enable_background_gc = bool(
                extra["ssd_enable_background_gc"]
            )
        if "auto_promote_on_read" in extra:
            cfg.eviction.auto_promote_on_read = bool(extra["auto_promote_on_read"])
        if "eviction_policy" in extra:
            cfg.eviction.policy = str(extra["eviction_policy"])
        if "eviction_candidate_window" in extra:
            cfg.eviction.candidate_window = int(extra["eviction_candidate_window"])
        if "ssd_io_backend" in extra and UMBPIoBackend is not None:
            backend = str(extra["ssd_io_backend"]).lower()
            if backend in ("pthread", "posix"):
                cfg.ssd.io.backend = UMBPIoBackend.PThread
            elif backend in ("io_uring", "uring"):
                cfg.ssd.io.backend = UMBPIoBackend.IoUring
        if "ssd_durability_mode" in extra and UMBPDurabilityMode is not None:
            durability = str(extra["ssd_durability_mode"]).lower()
            if durability in ("strict", "sync"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Strict
            elif durability in ("relaxed", "async"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Relaxed
        if "ssd_backend" in extra:
            ssd_backend = str(extra["ssd_backend"]).strip().lower()
            if ssd_backend not in ("posix", "spdk", "spdk_proxy"):
                raise ValueError(
                    "extra_config['ssd_backend'] must be one of: "
                    "posix, spdk, spdk_proxy"
                )
            cfg.ssd_backend = ssd_backend
        if "spdk_nvme_pci_addr" in extra:
            cfg.spdk_nvme_pci_addr = str(extra["spdk_nvme_pci_addr"])
        if "spdk_proxy_shm_name" in extra:
            cfg.spdk_proxy_shm_name = str(extra["spdk_proxy_shm_name"])
        if "spdk_proxy_startup_timeout_ms" in extra:
            cfg.spdk_proxy_startup_timeout_ms = int(
                extra["spdk_proxy_startup_timeout_ms"]
            )
        if "spdk_proxy_bin" in extra:
            cfg.spdk_proxy_bin = str(extra["spdk_proxy_bin"])
        if "spdk_proxy_tenant_id" in extra:
            cfg.spdk_proxy_tenant_id = int(extra["spdk_proxy_tenant_id"])
        if "spdk_proxy_tenant_quota_bytes" in extra:
            cfg.spdk_proxy_tenant_quota_bytes = int(
                extra["spdk_proxy_tenant_quota_bytes"]
            )
        if "spdk_proxy_max_channels" in extra:
            cfg.spdk_proxy_max_channels = int(extra["spdk_proxy_max_channels"])
        if "spdk_proxy_data_per_channel_mb" in extra:
            cfg.spdk_proxy_data_per_channel_mb = int(
                extra["spdk_proxy_data_per_channel_mb"]
            )
        if "spdk_proxy_auto_start" in extra:
            cfg.spdk_proxy_auto_start = bool(extra["spdk_proxy_auto_start"])
        if "spdk_proxy_idle_exit_timeout_ms" in extra:
            cfg.spdk_proxy_idle_exit_timeout_ms = int(
                extra["spdk_proxy_idle_exit_timeout_ms"]
            )
        if "spdk_proxy_allow_borrow" in extra:
            cfg.spdk_proxy_allow_borrow = bool(extra["spdk_proxy_allow_borrow"])
        if "spdk_proxy_reserved_shared_bytes" in extra:
            cfg.spdk_proxy_reserved_shared_bytes = int(
                extra["spdk_proxy_reserved_shared_bytes"]
            )

        # Operator-controlled escape hatch for hosts whose RDMA NIC cannot
        # register a single memory region as large as the full host KV buffer
        # (e.g. AINIC has a per-MR size cap).  When set, skip the one-shot
        # register_memory() call in register_mem_pool_host() and stay on the
        # staging-buffer fallback path (each transfer copies through a
        # staging_buffer_size-bounded MR that the IO engine pre-registers).
        disable_zero_copy_register = extra.get(
            "disable_zero_copy_register",
            _optional_env_str("UMBP_DISABLE_ZERO_COPY_REGISTER"),
        )
        self._disable_zero_copy_register = (
            _bool_from_any(disable_zero_copy_register)
            if disable_zero_copy_register is not None
            else False
        )

        master_address = extra.get(
            "master_address", _optional_env_str("UMBP_MASTER_ADDRESS")
        )
        if master_address and UMBPDistributedConfig is not None:
            dist_cfg = UMBPDistributedConfig()
            dist_cfg.master_config.master_address = str(master_address)

            node_address = extra.get(
                "node_address", _optional_env_str("UMBP_NODE_ADDRESS")
            )
            if node_address is None:
                node_address = _default_node_address()
            else:
                node_address = _select_rank_config_value(
                    node_address,
                    unique_rank,
                    "node_address",
                    str,
                )
            dist_cfg.master_config.node_address = node_address

            node_id = extra.get("node_id", _optional_env_str("UMBP_NODE_ID"))
            if node_id is None:
                dist_cfg.master_config.node_id = (
                    f"{node_address}:dp{dp_rank_hint if dp_rank_hint is not None else 0}"
                    f":pp{self.pp_rank}:tp{self.local_rank}"
                    f":{_PROCESS_INSTANCE_TOKEN}"
                )
            else:
                dist_cfg.master_config.node_id = _select_rank_config_value(
                    node_id,
                    unique_rank,
                    "node_id",
                    str,
                )

            if "auto_heartbeat" in extra:
                dist_cfg.master_config.auto_heartbeat = _bool_from_any(
                    extra["auto_heartbeat"]
                )

            io_engine_host = extra.get(
                "io_engine_host", _optional_env_str("UMBP_IO_ENGINE_HOST")
            )
            if io_engine_host is None:
                io_engine_host = node_address
            else:
                io_engine_host = _select_rank_config_value(
                    io_engine_host,
                    unique_rank,
                    "io_engine_host",
                    str,
                )
            dist_cfg.io_engine.host = io_engine_host

            io_engine_port = extra.get(
                "io_engine_port", _optional_env_str("UMBP_IO_ENGINE_PORT")
            )
            if io_engine_port is not None:
                dist_cfg.io_engine.port = _select_rank_config_value(
                    io_engine_port,
                    unique_rank,
                    "io_engine_port",
                    int,
                    auto_increment_scalar=True,
                )

            if "staging_buffer_size" in extra:
                dist_cfg.staging_buffer_size = int(extra["staging_buffer_size"])

            peer_service_port = extra.get(
                "peer_service_port", _optional_env_str("UMBP_PEER_SERVICE_PORT")
            )
            if peer_service_port is not None:
                dist_cfg.peer_service_port = _select_rank_config_value(
                    peer_service_port,
                    unique_rank,
                    "peer_service_port",
                    int,
                    auto_increment_scalar=True,
                )

            cache_remote_fetches = extra.get(
                "cache_remote_fetches",
                _optional_env_str("UMBP_CACHE_REMOTE_FETCHES"),
            )
            if cache_remote_fetches is not None:
                dist_cfg.cache_remote_fetches = _bool_from_any(cache_remote_fetches)

            env_tags = _parse_tag_values(_optional_env_str("SGLANG_UMBP_TAGS"))
            if env_tags:
                dist_cfg.master_config.tags = env_tags

            # Auto-compute master's PageBitmapAllocator page_size so every
            # UMBPStore Put/Get maps to exactly one master page (no partial
            # tail, 1 RDMA per page).  Resolution order:
            #   1. extra_config["dram_page_size"] — explicit operator override
            #      (escape hatch for debugging / forced experiments).
            #   2. derived from mem_pool_host (the normal production path).
            #   3. left at 0 when neither source is available; mori's
            #      UMBPDistributedConfig.dram_page_size defaults to 0, which
            #      delegates to the master-side ClientRegistryConfig
            #      .default_dram_page_size (2 MiB by default). The
            #      partial-tail safety net in PoolClient handles any
            #      size mismatch.
            page_byte_size = None
            if "dram_page_size" in extra:
                page_byte_size = int(extra["dram_page_size"])
            elif mem_pool_host is not None:
                # Probe element_size from the same buffer-meta helper that
                # batch_preprocess will actually use; this matches per-call
                # Put/Get size byte-for-byte for MHA / MHA-split / MLA / NSA
                # without per-case formulas (NSA in particular: get_ksize_per_token
                # would over-count by the indexer buffer that is never put to UMBP).
                dummy = torch.zeros(mem_pool_host.page_size, dtype=torch.int64)
                if self.is_mla_backend:
                    _, esz = mem_pool_host.get_page_buffer_meta(dummy)
                elif storage_config is not None and getattr(
                    storage_config, "should_split_heads", False
                ):
                    sf = storage_config.tp_lcm_size // storage_config.tp_size
                    _, esz = mem_pool_host.get_split_heads_page_buffer_meta(dummy, sf)
                else:
                    _, esz = mem_pool_host.get_page_buffer_meta(dummy)
                page_byte_size = int(esz[0]) if esz else 0

            if (
                page_byte_size is not None
                and page_byte_size > 0
                and hasattr(dist_cfg, "dram_page_size")
            ):
                dist_cfg.dram_page_size = int(page_byte_size)
                logger.info(
                    "UMBPStore: setting master dram_page_size=%d "
                    "(ksize_per_token=%s × page_size=%s%s)",
                    dist_cfg.dram_page_size,
                    (
                        mem_pool_host.get_ksize_per_token()
                        if mem_pool_host is not None
                        else "n/a"
                    ),
                    (mem_pool_host.page_size if mem_pool_host is not None else "n/a"),
                    (
                        f" / split_factor={storage_config.tp_lcm_size // storage_config.tp_size}"
                        if (
                            mem_pool_host is not None
                            and storage_config is not None
                            and getattr(storage_config, "should_split_heads", False)
                        )
                        else ""
                    ),
                )

            cfg.distributed = dist_cfg
            logger.info(
                "UMBPStore distributed mode: master=%s, node_id=%s, node_addr=%s, "
                "io=%s:%s, peer_port=%s",
                dist_cfg.master_config.master_address,
                dist_cfg.master_config.node_id,
                dist_cfg.master_config.node_address,
                dist_cfg.io_engine.host,
                dist_cfg.io_engine.port,
                dist_cfg.peer_service_port,
            )

        self.storage_config = storage_config
        io_bw_stats = extra.get(
            "io_bandwidth_stats", _optional_env_str("UMBP_IO_BW_STATS")
        )
        self._io_bw_stats_enabled = (
            True if io_bw_stats is None else _bool_from_any(io_bw_stats)
        )
        self._io_bw_stats_max_records = int(
            extra.get(
                "io_bandwidth_stats_max_records",
                os.getenv("UMBP_IO_BW_STATS_MAX_RECORDS", "100000"),
            )
        )
        self._io_bw_records = []
        self._io_bw_aggregate = {}
        self._io_bw_records_dropped = 0
        self._io_bw_stats_printed = False
        # JSONL side-channel: survives SIGKILL / logger flush issues.
        # File is opened lazily on the first record so we already know the
        # final rank / pp_rank / tp_size at construction time.
        self._io_bw_jsonl_enabled = _bool_from_any(
            extra.get(
                "io_bandwidth_stats_jsonl",
                os.getenv("UMBP_IO_BW_STATS_JSONL", "true"),
            )
        )
        self._io_bw_jsonl_fsync = _bool_from_any(
            extra.get(
                "io_bandwidth_stats_jsonl_fsync",
                os.getenv("UMBP_IO_BW_STATS_JSONL_FSYNC", "false"),
            )
        )
        self._io_bw_jsonl_dir = extra.get(
            "io_bandwidth_stats_dir",
            _optional_env_str("UMBP_IO_BW_STATS_DIR"),
        )
        self._io_bw_jsonl_file = None
        self._io_bw_jsonl_path = None

        # MLA + TP > 1: shared SSD mode (standalone only).
        # In distributed mode every rank is a peer of the master-led pool; we
        # must NOT short-circuit followers (would leave their DRAM pool empty
        # while the master still routes keys to them, causing Get misses).
        self.is_mla_follower = False
        tp_size = self.tp_size
        use_spdk = cfg.ssd_backend in ("spdk", "spdk_proxy")
        distributed_enabled = cfg.distributed is not None
        if not distributed_enabled and self.is_mla_backend and tp_size > 1:
            cfg.ssd.enabled = True
            if self.local_rank == 0:
                # Leader: copy every DRAM write to shared SSD.
                cfg.role = UMBPRole.SharedSSDLeader
            else:
                # Follower: read-only access.
                cfg.role = UMBPRole.SharedSSDFollower
                self.is_mla_follower = True
                # SPDK: follower must use the proxy path rather than direct
                # SpdkSsdTier.  Give a longer startup timeout so followers can
                # wait for the shared proxy service to become READY.
                if use_spdk:
                    cfg.ssd_backend = "spdk_proxy"
                    if cfg.spdk_proxy_startup_timeout_ms < 60000:
                        cfg.spdk_proxy_startup_timeout_ms = 60000
            logger.info(
                "UMBPStore MLA+TP>1: rank=%d, role=%s, ssd_backend=%s, shared_ssd=%s",
                self.local_rank,
                "leader" if self.local_rank == 0 else "follower",
                cfg.ssd_backend,
                cfg.ssd.storage_dir,
            )

        try:
            from sglang.srt.layers.dp_attention import (
                get_attention_dp_rank,
                get_attention_dp_size,
                is_dp_attention_enabled,
            )

            if is_dp_attention_enabled():
                dp_rank = get_attention_dp_rank()
                dp_size = get_attention_dp_size()
                dp_rank_hint = dp_rank
                dp_size_hint = dp_size
                if cfg.ssd.enabled:
                    if cfg.ssd_backend in ("spdk", "spdk_proxy"):
                        # DP + SPDK must always use the proxy service path.
                        # Direct SpdkSsdTier is single-process and cannot
                        # provide tenant isolation across DP ranks.
                        cfg.ssd_backend = "spdk_proxy"
                        if cfg.spdk_proxy_startup_timeout_ms < 60000:
                            cfg.spdk_proxy_startup_timeout_ms = 60000
                        if tenant_id_base is not None:
                            cfg.spdk_proxy_tenant_id = tenant_id_base + dp_rank
                        elif not explicit_tenant_id:
                            cfg.spdk_proxy_tenant_id = dp_rank
                        elif dp_size > 1:
                            logger.warning(
                                "UMBPStore DP isolation: using explicit fixed tenant_id=%s "
                                "with dp_size=%d; all DP groups will share one tenant "
                                "unless you set spdk_proxy_tenant_id_base",
                                cfg.spdk_proxy_tenant_id,
                                dp_size,
                            )
                        if cfg.spdk_proxy_tenant_quota_bytes <= 0 and dp_size > 1:
                            # Reserve 5% headroom for offset allocator bin
                            # rounding (small-float bins round up each
                            # allocation by up to ~12.5%).
                            safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                            cfg.spdk_proxy_tenant_quota_bytes = max(
                                1, safe_cap // dp_size
                            )
                        # Validate: total tenant quotas must fit within SSD
                        # capacity after allocator rounding.
                        if dp_size > 1:
                            total_quota = cfg.spdk_proxy_tenant_quota_bytes * dp_size
                            if total_quota > cfg.ssd.capacity_bytes:
                                old_quota = cfg.spdk_proxy_tenant_quota_bytes
                                safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                                cfg.spdk_proxy_tenant_quota_bytes = max(
                                    1, safe_cap // dp_size
                                )
                                logger.warning(
                                    "UMBPStore: tenant_quota_bytes=%d × dp_size=%d = %d "
                                    "exceeds ssd_capacity=%d. Reduced to %d to "
                                    "avoid SPDK proxy NO_SPACE. Consider "
                                    "increasing UMBP_SSD_BYTES.",
                                    old_quota,
                                    dp_size,
                                    total_quota,
                                    cfg.ssd.capacity_bytes,
                                    cfg.spdk_proxy_tenant_quota_bytes,
                                )
                        logger.info(
                            "UMBPStore DP isolation: dp_rank=%d, dp_size=%d, tenant_id=%s, tenant_quota_bytes=%s",
                            dp_rank,
                            dp_size,
                            getattr(cfg, "spdk_proxy_tenant_id", "n/a"),
                            getattr(cfg, "spdk_proxy_tenant_quota_bytes", "n/a"),
                        )
                    else:
                        cfg.ssd.storage_dir = f"{cfg.ssd.storage_dir}/dp{dp_rank}"
                        logger.info(
                            "UMBPStore DP isolation: dp_rank=%d, dp_size=%d, ssd_dir=%s",
                            dp_rank,
                            dp_size,
                            cfg.ssd.storage_dir,
                        )
        except (ImportError, AssertionError):
            pass

        if (
            cfg.ssd.enabled
            and not self.is_mla_follower
            and not (self.is_mla_backend and tp_size > 1)
            and cfg.ssd_backend not in ("spdk", "spdk_proxy")
        ):
            rank_dir_parts = []
            if dp_rank_hint is not None:
                rank_dir_parts.append(f"dp{dp_rank_hint}")
            if self.pp_size > 1:
                rank_dir_parts.append(f"pp{self.pp_rank}")
            if self.tp_size > 1:
                rank_dir_parts.append(f"tp{self.local_rank}")
            if not rank_dir_parts and unique_rank != 0:
                rank_dir_parts.append(f"rank{unique_rank}")
            if rank_dir_parts:
                cfg.ssd.storage_dir = os.path.join(
                    cfg.ssd.storage_dir, "_".join(rank_dir_parts)
                )
                logger.info(
                    "UMBPStore local SSD isolation: unique_rank=%d, ssd_dir=%s",
                    unique_rank,
                    cfg.ssd.storage_dir,
                )

        if cfg.ssd.enabled and cfg.ssd_backend in ("spdk", "spdk_proxy"):
            if dp_rank_hint is not None and tenant_id_base is not None:
                cfg.spdk_proxy_tenant_id = tenant_id_base + dp_rank_hint
            elif dp_rank_hint is not None and not explicit_tenant_id:
                cfg.spdk_proxy_tenant_id = dp_rank_hint
            if (
                dp_rank_hint is not None
                and dp_size_hint is not None
                and cfg.spdk_proxy_tenant_quota_bytes <= 0
                and dp_size_hint > 1
            ):
                safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                cfg.spdk_proxy_tenant_quota_bytes = max(1, safe_cap // dp_size_hint)

        self.client = UMBPClient(cfg)
        if self._io_bw_stats_enabled:
            atexit.register(self._print_io_bandwidth_stats)
        if mem_pool_host is not None:
            self.register_mem_pool_host(mem_pool_host)

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        self.split_factor = 0
        if storage_config and storage_config.should_split_heads:
            self.split_factor = storage_config.tp_lcm_size // storage_config.tp_size
            base_rank = self.local_rank * self.split_factor
            target_ranks = [base_rank + i for i in range(self.split_factor)]
            if self.enable_pp:
                self.mha_suffix = [f"{rank}_{self.pp_rank}" for rank in target_ranks]
            else:
                self.mha_suffix = [f"{rank}" for rank in target_ranks]

        logger.info(
            "UMBPStore initialized: dram=%d MB, ssd=%s, mla=%s, rank=%d, ssd_backend=%s",
            cfg.dram.capacity_bytes // (1024 * 1024),
            cfg.ssd.enabled,
            self.is_mla_backend,
            self.local_rank,
            cfg.ssd_backend,
        )

    # ------------------------------------------------------------------
    # Host memory pool registration
    # ------------------------------------------------------------------
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
        ], "UMBP store only supports page_first, page_first_direct, or page_head layout"

        # In distributed mode, pre-register the entire host KV buffer with the
        # underlying RDMA IOEngine so PoolClient can take the zero-copy path
        # for batch_get_into_ptr / batch_put_from_ptr (skips the staging
        # buffer memcpy + lock and removes the per-call `staging_buffer_size`
        # cap).  Standalone returns true as no-op by IUMBPClient contract;
        # we still gate on is_distributed() below to avoid a pointless call.
        self._zero_copy_registered = False
        if self.client is None:
            return
        try:
            is_distributed = bool(self.client.is_distributed())
        except Exception:
            is_distributed = False
        if not is_distributed:
            return
        if not hasattr(self.client, "register_memory"):
            return
        if getattr(self, "_disable_zero_copy_register", False):
            logger.info(
                "UMBPStore: skipping host KV buffer RDMA registration because "
                "disable_zero_copy_register=true (UMBP_DISABLE_ZERO_COPY_REGISTER). "
                "Falling back to the staging-buffer transfer path; per-transfer "
                "size is capped by distributed.staging_buffer_size."
            )
            return
        try:
            kv_buffer = mem_pool_host.kv_buffer
            host_ptr = int(kv_buffer.data_ptr())
            host_size = int(kv_buffer.numel() * kv_buffer.element_size())
            # When the buffer is backed by hugepages the mmap region is
            # rounded up to the hugepage boundary.  RDMA ibv_reg_mr on
            # some NICs (AINIC / ROCm) requires the registered region to
            # cover complete hugepages, so use the full mapped_size
            # instead of the logical tensor size.
            allocator = getattr(mem_pool_host, "allocator", None)
            mapped_size_fn = getattr(allocator, "mapped_size_for", None)
            if mapped_size_fn is not None:
                mapped_size = mapped_size_fn(host_ptr)
            else:
                mapped_size = getattr(allocator, "mapped_size", 0)
            if mapped_size > host_size:
                host_size = mapped_size
            ok = bool(self.client.register_memory(host_ptr, host_size))
        except Exception as exc:
            logger.warning(
                "UMBPStore: register_memory failed (%s); falling back to staging "
                "buffer path. Per-transfer size will be capped by "
                "distributed.staging_buffer_size.",
                exc,
            )
            return
        if ok:
            self._zero_copy_registered = True
            logger.info(
                "UMBPStore: registered host KV buffer for RDMA zero-copy "
                "(ptr=0x%x, size=%d MB)",
                host_ptr,
                host_size // (1024 * 1024),
            )
        else:
            logger.warning(
                "UMBPStore: register_memory returned false; staying on staging "
                "buffer fallback path."
            )

    # ------------------------------------------------------------------
    # Key suffix generation — mirrors MooncakeStore
    # ------------------------------------------------------------------
    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[bool], is_set_operate=False):
        """Convert per-key-component results to per-page results.

        For MHA: each page has K+V → group pairs.
        For MLA: each page has K only.
        """
        if self.is_mla_backend:
            return list(results)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                group_size = self.split_factor * 2
                groups = [
                    results[i : i + group_size]
                    for i in range(0, len(results), group_size)
                ]
                return [all(g) for g in groups]
            else:
                # Group K/V pairs
                kv_pairs = zip(results[::2], results[1::2])
                return [k and v for k, v in kv_pairs]

    # ------------------------------------------------------------------
    # Zero-copy v1 interface
    # ------------------------------------------------------------------
    def _io_bw_jsonl_open(self) -> None:
        """Open the IOBW JSONL side-channel lazily on first use.

        Resolution order for the output directory:
          1. ``UMBP_IO_BW_STATS_DIR`` / ``io_bandwidth_stats_dir`` extra
          2. ``SGLANG_LOG_DIR``
          3. cwd ``./umbp_iobw_logs``
          4. ``${TMPDIR}/umbp_iobw_logs`` (fallback when cwd is not writable)
        """
        if self._io_bw_jsonl_file is not None:
            return
        if not self._io_bw_jsonl_enabled or not self._io_bw_stats_enabled:
            return

        candidates = []
        if self._io_bw_jsonl_dir:
            candidates.append(self._io_bw_jsonl_dir)
        sglang_log_dir = _optional_env_str("SGLANG_LOG_DIR")
        if sglang_log_dir:
            candidates.append(os.path.join(sglang_log_dir, "umbp_iobw_logs"))
        candidates.append(os.path.join(os.getcwd(), "umbp_iobw_logs"))
        candidates.append(os.path.join(tempfile.gettempdir(), "umbp_iobw_logs"))

        target_dir = None
        for c in candidates:
            try:
                os.makedirs(c, exist_ok=True)
                # Probe writability by opening a small file.
                probe = os.path.join(c, ".iobw_writable_probe")
                with open(probe, "w"):
                    pass
                os.unlink(probe)
                target_dir = c
                break
            except OSError:
                continue

        if target_dir is None:
            logger.warning(
                "[UMBPStore][IOBW] no writable directory found among %s; "
                "JSONL side-channel disabled",
                candidates,
            )
            self._io_bw_jsonl_enabled = False
            return

        host = socket.gethostname()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = (
            f"iobw_{host}_pid{os.getpid()}_dp{getattr(self, 'dp_rank', 'x')}_"
            f"tp{self.local_rank}_pp{self.pp_rank}_{ts}_{_PROCESS_INSTANCE_TOKEN}.jsonl"
        )
        path = os.path.join(target_dir, fname)
        try:
            self._io_bw_jsonl_file = open(path, "a", buffering=1)
        except OSError as exc:
            logger.warning(
                "[UMBPStore][IOBW] failed to open JSONL file %s: %s; "
                "JSONL side-channel disabled",
                path,
                exc,
            )
            self._io_bw_jsonl_enabled = False
            return

        self._io_bw_jsonl_path = path

        # Write a one-shot "open" header for cross-referencing with server.log.
        header = {
            "type": "open",
            "ts": time.time(),
            "host": host,
            "pid": os.getpid(),
            "local_rank": self.local_rank,
            "pp_rank": self.pp_rank,
            "tp_size": self.tp_size,
            "process_token": _PROCESS_INSTANCE_TOKEN,
        }
        self._io_bw_jsonl_write(header)
        logger.info(
            "[UMBPStore][IOBW] JSONL side-channel opened: %s (fsync=%s)",
            path,
            self._io_bw_jsonl_fsync,
        )

    def _io_bw_jsonl_write(self, record: dict) -> None:
        f = self._io_bw_jsonl_file
        if f is None:
            return
        try:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            # line-buffered open() already flushes on \n, but be explicit.
            f.flush()
            if self._io_bw_jsonl_fsync:
                os.fsync(f.fileno())
        except (OSError, ValueError):
            # Don't let a logging failure abort the IO path.
            pass

    def _io_bw_jsonl_close(self) -> None:
        f = getattr(self, "_io_bw_jsonl_file", None)
        if f is None:
            return
        try:
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
            f.close()
        except (OSError, ValueError):
            pass
        self._io_bw_jsonl_file = None

    def _record_io_bandwidth(
        self,
        op: str,
        total_bytes: int,
        success_bytes: int,
        request_count: int,
        expanded_count: int,
        success_count: int,
        elapsed_s: float,
    ) -> None:
        if not getattr(self, "_io_bw_stats_enabled", False):
            return

        elapsed_s = max(elapsed_s, 1e-12)
        bandwidth_gib_s = success_bytes / elapsed_s / (1024**3)

        stats = self._io_bw_aggregate.setdefault(
            op,
            {
                "calls": 0,
                "requests": 0,
                "expanded": 0,
                "success": 0,
                "total_bytes": 0,
                "success_bytes": 0,
                "elapsed_s": 0.0,
                "max_bandwidth_gib_s": 0.0,
            },
        )
        stats["calls"] += 1
        stats["requests"] += request_count
        stats["expanded"] += expanded_count
        stats["success"] += success_count
        stats["total_bytes"] += total_bytes
        stats["success_bytes"] += success_bytes
        stats["elapsed_s"] += elapsed_s
        stats["max_bandwidth_gib_s"] = max(
            stats["max_bandwidth_gib_s"], bandwidth_gib_s
        )

        if len(self._io_bw_records) < self._io_bw_stats_max_records:
            self._io_bw_records.append(
                {
                    "op": op,
                    "requests": request_count,
                    "expanded": expanded_count,
                    "success": success_count,
                    "total_bytes": total_bytes,
                    "success_bytes": success_bytes,
                    "elapsed_s": elapsed_s,
                    "bandwidth_gib_s": bandwidth_gib_s,
                }
            )
        else:
            self._io_bw_records_dropped += 1

        # Side-channel: append-and-flush per call so SIGKILL / logger buffering
        # can't lose per-call records.  This is critical when the process is
        # torn down by `timeout --signal=TERM --kill-after=30` or by an OOM
        # killer before atexit runs.
        if self._io_bw_jsonl_enabled:
            if self._io_bw_jsonl_file is None:
                self._io_bw_jsonl_open()
            self._io_bw_jsonl_write(
                {
                    "type": "call",
                    "ts": time.time(),
                    "op": op,
                    "requests": request_count,
                    "expanded": expanded_count,
                    "success": success_count,
                    "total_bytes": total_bytes,
                    "success_bytes": success_bytes,
                    "elapsed_s": elapsed_s,
                    "bandwidth_gib_s": bandwidth_gib_s,
                }
            )

    def _print_io_bandwidth_stats(self) -> None:
        if not getattr(self, "_io_bw_stats_enabled", False):
            return
        if getattr(self, "_io_bw_stats_printed", False):
            return
        self._io_bw_stats_printed = True

        if not self._io_bw_aggregate:
            logger.info("[UMBPStore][IOBW] no BatchGet/BatchPut calls recorded")
            if self._io_bw_jsonl_enabled and self._io_bw_jsonl_file is not None:
                self._io_bw_jsonl_write(
                    {
                        "type": "summary",
                        "ts": time.time(),
                        "ops": {},
                        "records_collected": 0,
                        "records_dropped": 0,
                    }
                )
                self._io_bw_jsonl_close()
            return

        logger.info(
            "[UMBPStore][IOBW] per-call storage bandwidth records: count=%d dropped=%d "
            "rank=%d pp_rank=%d tp_size=%d",
            len(self._io_bw_records),
            self._io_bw_records_dropped,
            self.local_rank,
            self.pp_rank,
            self.tp_size,
        )
        for idx, record in enumerate(self._io_bw_records, start=1):
            logger.info(
                "[UMBPStore][IOBW] #%05d op=%s requests=%d expanded=%d "
                "success=%d/%d total_bytes=%d success_bytes=%d elapsed_ms=%.3f "
                "bandwidth_gib_s=%.3f",
                idx,
                record["op"],
                record["requests"],
                record["expanded"],
                record["success"],
                record["expanded"],
                record["total_bytes"],
                record["success_bytes"],
                record["elapsed_s"] * 1000,
                record["bandwidth_gib_s"],
            )

        summary_payload = {}
        for op, stats in sorted(self._io_bw_aggregate.items()):
            aggregate_bandwidth = (
                stats["success_bytes"] / max(stats["elapsed_s"], 1e-12) / (1024**3)
            )
            logger.info(
                "[UMBPStore][IOBW] summary op=%s calls=%d requests=%d expanded=%d "
                "success=%d total_bytes=%d success_bytes=%d elapsed_ms=%.3f "
                "avg_bandwidth_gib_s=%.3f max_call_bandwidth_gib_s=%.3f",
                op,
                stats["calls"],
                stats["requests"],
                stats["expanded"],
                stats["success"],
                stats["total_bytes"],
                stats["success_bytes"],
                stats["elapsed_s"] * 1000,
                aggregate_bandwidth,
                stats["max_bandwidth_gib_s"],
            )
            summary_payload[op] = {
                "calls": stats["calls"],
                "requests": stats["requests"],
                "expanded": stats["expanded"],
                "success": stats["success"],
                "total_bytes": stats["total_bytes"],
                "success_bytes": stats["success_bytes"],
                "elapsed_s": stats["elapsed_s"],
                "avg_bandwidth_gib_s": aggregate_bandwidth,
                "max_call_bandwidth_gib_s": stats["max_bandwidth_gib_s"],
            }

        if self._io_bw_jsonl_enabled and self._io_bw_jsonl_file is not None:
            self._io_bw_jsonl_write(
                {
                    "type": "summary",
                    "ts": time.time(),
                    "ops": summary_payload,
                    "records_collected": len(self._io_bw_records),
                    "records_dropped": self._io_bw_records_dropped,
                }
            )
            self._io_bw_jsonl_close()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        # Normalize sizes to list of per-key sizes
        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        total_bytes = sum(sizes)
        logger.info(
            "[UMBPStore] batch_get_v1: calling UMBP BatchGet: "
            "keys=%d expanded_keys=%d total_bytes=%d",
            len(keys),
            len(key_strs),
            total_bytes,
        )
        start_time = time.perf_counter()
        get_results = self.client.batch_get_into_ptr(key_strs, list(buffer_ptrs), sizes)
        elapsed_s = time.perf_counter() - start_time
        success_count = sum(1 for r in get_results if r)
        success_bytes = sum(size for size, ok in zip(sizes, get_results) if ok)
        self._record_io_bandwidth(
            "BatchGet",
            total_bytes,
            success_bytes,
            len(keys),
            len(key_strs),
            success_count,
            elapsed_s,
        )
        logger.info(
            "[UMBPStore] batch_get_v1: UMBP BatchGet done: success=%d/%d "
            "elapsed_ms=%.3f bandwidth_gib_s=%.3f",
            success_count,
            len(get_results),
            elapsed_s * 1000,
            success_bytes / max(elapsed_s, 1e-12) / (1024**3),
        )
        return self._batch_postprocess(get_results)

    def _compute_expanded_depths(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> List[int]:
        """Compute per-expanded-key depth values from prefix_keys metadata.

        depth = len(prefix_keys) + page_index_within_node.
        All key variants of the same page (K, V, multi-rank) share the same depth.
        Returns an empty list if no metadata is available (caller falls back to plain LRU).
        """
        prefix_keys = getattr(extra_info, "prefix_keys", None) if extra_info else None
        if prefix_keys is None:
            return []

        prefix_len = len(prefix_keys)
        depths_per_page = [prefix_len + i for i in range(len(keys))]

        # Expand to match the key_strs layout produced by _batch_preprocess.
        expanded = []
        for d in depths_per_page:
            if self.is_mla_backend:
                expanded.append(d)  # MLA: 1 key per page
            elif self.storage_config and self.storage_config.should_split_heads:
                # split heads: 2 keys per split rank, split_factor ranks per page
                for _ in range(self.split_factor):
                    expanded.append(d)
                    expanded.append(d)
            else:
                expanded.append(d)  # K
                expanded.append(d)  # V
        return expanded

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Follower never writes (CacheController also sets backup_skip, but guard here too)
        if self.is_mla_follower:
            page_count = len(host_indices) // self.mem_pool_host.page_size
            return [True] * page_count

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        expanded_depths = self._compute_expanded_depths(keys, extra_info)

        total_bytes = sum(sizes)
        logger.info(
            "[UMBPStore] batch_set_v1: calling UMBP BatchPut: "
            "keys=%d expanded_keys=%d total_bytes=%d with_depth=%s",
            len(keys),
            len(key_strs),
            total_bytes,
            bool(expanded_depths),
        )

        start_time = time.perf_counter()
        if expanded_depths:
            put_results = self.client.batch_put_from_ptr_with_depth(
                key_strs, list(buffer_ptrs), sizes, expanded_depths
            )
        else:
            put_results = self.client.batch_put_from_ptr(
                key_strs, list(buffer_ptrs), sizes
            )
        elapsed_s = time.perf_counter() - start_time

        success_count = sum(1 for r in put_results if r)
        success_bytes = sum(size for size, ok in zip(sizes, put_results) if ok)
        self._record_io_bandwidth(
            "BatchPut",
            total_bytes,
            success_bytes,
            len(keys),
            len(key_strs),
            success_count,
            elapsed_s,
        )
        logger.info(
            "[UMBPStore] batch_set_v1: UMBP BatchPut done: success=%d/%d "
            "elapsed_ms=%.3f bandwidth_gib_s=%.3f",
            success_count,
            len(put_results),
            elapsed_s * 1000,
            success_bytes / max(elapsed_s, 1e-12) / (1024**3),
        )
        return self._batch_postprocess(put_results, is_set_operate=True)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """Return count of consecutive existing keys from start."""
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.storage_config and self.storage_config.should_split_heads:
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        hit_count = self.client.batch_exists_consecutive(query_keys)
        return hit_count // key_multiplier

    # ------------------------------------------------------------------
    # Legacy ABC interface (required by HiCacheStorage)
    # ------------------------------------------------------------------
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        if target_location is None or target_sizes is None:
            return None
        ok = self.client.get_into_ptr(key, target_location, target_sizes)
        return target_location if ok else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        if not keys:
            return 0
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_get_into_ptr(
            keys,
            list(target_locations),
            list(target_sizes),
        )
        for i, ok in enumerate(results):
            if not ok:
                return i
        return len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.is_mla_follower:
            return True
        if target_location is None or target_sizes is None:
            return False
        return self.client.put_from_ptr(key, target_location, target_sizes)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if not keys:
            return False
        if self.is_mla_follower:
            return True
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_put_from_ptr(
            keys,
            list(target_locations),
            list(target_sizes),
        )
        return all(results)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def clear(self) -> None:
        if not self.client.clear():
            raise RuntimeError("UMBP clear full-sync failed")

    def flush(self) -> bool:
        if self.client is None or not hasattr(self.client, "flush"):
            return True
        return bool(self.client.flush())

    def close(self) -> None:
        if getattr(self, "client", None) is None:
            return
        try:
            self.flush()
        except Exception:
            logger.exception("UMBPStore flush during close failed")
        self._print_io_bandwidth_stats()
        self.client = None
