"""
Entry point for the Deriv CFD bot (replaces Polymarket logic).
"""

import asyncio

from .deriv_cfd_bot import main


if __name__ == "__main__":
    asyncio.run(main())
