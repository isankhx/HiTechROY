"""Mission clock. SCALE > 1 only in simulation (time acceleration)."""
import asyncio
import time

SCALE = 1.0


def now():
    return time.monotonic() * SCALE


async def sleep(seconds):
    await asyncio.sleep(seconds / SCALE)
