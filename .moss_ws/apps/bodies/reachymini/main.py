"""Reachy Mini Body App — 通过 AppStore cell 地址空间暴露 body channel 给 Ghost。

启动流程:
1. Matrix.discover() -> 从环境变量加载配置, IoC 容器就绪
2. Matrix.run() -> 启动 Matrix 生命周期 (包含 Zenoh session, port 20770)
3. 构建 MossInReachyMini channel (连接机器人硬件)
4. matrix.provide_channel() -> 通过 ZenohChannelProvider 暴露 channel
5. Ghost 的 AppStoreChannel 自动创建 proxy 并连接

环境变量:
  REACHY_ROBOT_HOST: 机器人 IP/hostname (默认 reachy-mini.local)
  REACHY_MEDIA_BACKEND: media backend (默认 no_media, 跳过 GStreamer camera)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.resource import ResourceRegistry
from ghoshell_moss.core.resources.memory_registry import InMemoryResourcesRegistry

from ghoshell_moss_contrib.moss_in_reachy_mini.main import MossInReachyMini

# Pre-register ResourceRegistry provider to avoid bootstrap order issue in app subprocess
_original_discover = Matrix.discover


@classmethod
def _patched_discover(cls):
    instance = _original_discover.__func__(cls)
    if not instance._container.bound(ResourceRegistry):
        instance._container.set(ResourceRegistry, InMemoryResourcesRegistry())
    return instance


Matrix.discover = _patched_discover


def _patch_reachy_ws_liveness() -> None:
    """Avoid false SDK disconnects when daemon heartbeat is sparse."""
    try:
        from reachy_mini.io.ws_client import WSClient
    except Exception:
        return
    if getattr(WSClient, "_moss_liveness_patched", False):
        return

    def _is_connected(self) -> bool:
        return getattr(self, "_ws", None) is not None and not getattr(self, "_stop_event").is_set()

    WSClient.is_connected = _is_connected
    WSClient._moss_liveness_patched = True


async def _connect_robot(logger, robot_host: str, media_backend: str, max_attempts: int = 5) -> 'ReachyMini':
    """Connect to robot with retry logic. Returns ReachyMini instance or raises."""
    from reachy_mini import ReachyMini

    for attempt in range(max_attempts):
        try:
            mini = ReachyMini(host=robot_host, media_backend=media_backend)
            logger.info("[ReachyMini Body App] Robot connected successfully")
            return mini
        except Exception as e:
            wait = (attempt + 1) * 2
            logger.warning(
                "[ReachyMini Body App] Connection attempt %d/%d failed: %s, retrying in %ds",
                attempt + 1, max_attempts, e, wait,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait)

    raise RuntimeError(f"Failed to connect to robot at {robot_host} after {max_attempts} attempts")


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _ensure_motors_enabled(logger, mini) -> None:
    if not _env_enabled("MOSS_REACHY_AUTO_ENABLE_MOTORS", "1"):
        return
    try:
        mini.enable_motors()
        logger.info("[ReachyMini Body App] Motors enabled on startup")
    except Exception as e:
        logger.warning("[ReachyMini Body App] Enable motors on startup failed: %s", e)


async def _connection_watchdog(matrix: Matrix, mini, robot_host: str, media_backend: str) -> None:
    """Monitor robot connection health and log warnings on disconnect.

    The watchdog checks the WebSocket client liveness every few seconds.
    If the connection drops, it attempts to reconnect the underlying client
    so that in-flight commands can resume without restarting the whole app.
    """
    logger = matrix.logger
    check_interval = 5.0
    reconnect_attempts = 0
    max_reconnect = 10

    while True:
        await asyncio.sleep(check_interval)

        try:
            client = mini.client
            if not getattr(client, '_is_alive', True):
                raise ConnectionError("WebSocket client reports not alive")
            # Lightweight health check: get_status uses the WebSocket
            client.get_status()
            if reconnect_attempts > 0:
                logger.info("[ReachyMini Watchdog] Connection recovered after %d reconnect attempts", reconnect_attempts)
                reconnect_attempts = 0
        except Exception as e:
            reconnect_attempts += 1
            logger.warning(
                "[ReachyMini Watchdog] Connection lost (attempt %d/%d): %s",
                reconnect_attempts, max_reconnect, e,
            )

            if reconnect_attempts > max_reconnect:
                logger.error("[ReachyMini Watchdog] Max reconnection attempts reached. Giving up.")
                break

            # Try to reconnect the client
            try:
                from reachy_mini import ReachyMini
                _orig_release = ReachyMini.release_media
                ReachyMini.release_media = lambda self: None
                try:
                    new_mini = ReachyMini(host=robot_host, media_backend=media_backend)
                finally:
                    ReachyMini.release_media = _orig_release
                # Swap the client reference so existing components use the new connection
                mini.client = new_mini.client
                mini.media_manager = new_mini.media_manager
                logger.info("[ReachyMini Watchdog] Reconnected successfully")
                # Re-enable motors after reconnect
                try:
                    mini.enable_motors()
                    mini.wake_up()
                except Exception as motor_err:
                    logger.warning("[ReachyMini Watchdog] Re-enable motors failed: %s", motor_err)
            except Exception as reconnect_err:
                wait = min(reconnect_attempts * 3, 30)
                logger.warning(
                    "[ReachyMini Watchdog] Reconnect failed: %s, waiting %ds",
                    reconnect_err, wait,
                )
                await asyncio.sleep(wait)


async def provide_channel(matrix: Matrix) -> None:
    """构建并暴露 Reachy Mini body channel。

    AppStoreChannel 会自动通过 matrix.channel_proxy(address=cell_address) 创建 proxy,
    Ghost 不需要额外配置即可发现和使用 body 命令。
    """
    logger = matrix.logger
    logger.info("[ReachyMini Body App] Matrix ready, building body channel...")
    _patch_reachy_ws_liveness()

    from reachy_mini import ReachyMini
    robot_host = os.environ.get("REACHY_ROBOT_HOST", "reachy-mini.local")
    media_backend = os.environ.get("REACHY_MEDIA_BACKEND", "no_media")
    allow_vision = media_backend != "no_media"
    logger.info("[ReachyMini Body App] Connecting to robot at %s (media=%s, vision=%s)", robot_host, media_backend, allow_vision)

    # Prevent SDK from releasing media (Ghost's audio player owns it)
    _orig_release = ReachyMini.release_media
    ReachyMini.release_media = lambda self: None

    try:
        mini = await _connect_robot(logger, robot_host, media_backend)
    except RuntimeError:
        # Degraded mode: try simulation
        logger.error("[ReachyMini Body App] All connection attempts failed. Starting in degraded mode.")
        ReachyMini.release_media = _orig_release
        try:
            mini = ReachyMini(use_sim=True, spawn_daemon=True, media_backend="no_media", connection_mode="localhost_only")
            logger.info("[ReachyMini Body App] Fallback to simulation mode")
        except Exception:
            logger.error("[ReachyMini Body App] Simulation also failed. Exiting.")
            raise RuntimeError(f"Failed to connect to robot at {robot_host}")

    # Restore original release_media
    ReachyMini.release_media = _orig_release
    _ensure_motors_enabled(logger, mini)

    # Build channel
    reachy = MossInReachyMini(
        mini=mini,
        ws=matrix.workspace,
        logger=logger,
        allow_vision=allow_vision,
    )
    channel = reachy.as_channel()
    logger.info("[ReachyMini Body App] Channel built, providing via cell address...")

    # Provide channel to Ghost via Fractal protocol
    await matrix.provide_channel(channel)

    # Start connection watchdog as background task
    await matrix.create_task(_connection_watchdog(matrix, mini, robot_host, media_backend))
    logger.info("[ReachyMini Body App] Connection watchdog started")


if __name__ == "__main__":
    Matrix.discover().run(provide_channel)
