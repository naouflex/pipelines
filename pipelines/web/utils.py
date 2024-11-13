"""Utilities for the Playwright browser tools."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, TypeVar
import nest_asyncio

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def aget_current_page(browser: AsyncBrowser) -> AsyncPage:
    """
    Asynchronously get the current page of the browser.

    Args:
        browser: The browser (AsyncBrowser) to get the current page from.

    Returns:
        AsyncPage: The current page.
    """
    if not browser.contexts:
        context = await browser.new_context()
        return await context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return await context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


def get_current_page(browser: SyncBrowser) -> SyncPage:
    """
    Get the current page of the browser.
    Args:
        browser: The browser to get the current page from.

    Returns:
        SyncPage: The current page.
    """
    if not browser.contexts:
        context = browser.new_context()
        return context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return context.new_page()
    # Assuming the last page in the list is the active one
    return context.pages[-1]


async def create_async_playwright_browser(
    headless: bool = True, args: Optional[List[str]] = None
) -> AsyncBrowser:
    """
    Create an async playwright browser.

    Args:
        headless: Whether to run the browser in headless mode. Defaults to True.
        args: arguments to pass to browser.chromium.launch

    Returns:
        AsyncBrowser: The playwright browser.
    """
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    return await playwright.chromium.launch(headless=headless, args=args)


def create_sync_playwright_browser(
    headless: bool = True, args: Optional[List[str]] = None
) -> SyncBrowser:
    """
    Create a playwright browser.

    Args:
        headless: Whether to run the browser in headless mode. Defaults to True.
        args: arguments to pass to browser.chromium.launch

    Returns:
        SyncBrowser: The playwright browser.
    """
    from playwright.sync_api import sync_playwright

    browser = sync_playwright().start()
    return browser.chromium.launch(headless=headless, args=args)


T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine.

    Args:
        coro: The coroutine to run. Coroutine[Any, Any, T]

    Returns:
        T: The result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
