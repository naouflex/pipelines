"""Utilities for the Playwright browser tools."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, TypeVar
import os
import sys
import platform

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "none"



def configure_event_loop():
    """Configure the event loop based on the platform and environment."""
    try:
        # Only try to use uvloop if nest_asyncio is not needed
        if not any(name in sys.modules for name in ['jupyter', 'ipykernel', 'notebook']):
            try:
                import uvloop
                uvloop.install()
            except ImportError:
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        else:
            # If we're in a Jupyter environment, prioritize nest_asyncio over uvloop
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                print("Warning: nest_asyncio not available")
                    
    except Exception as e:
        print(f"Warning: Error configuring event loop: {e}")
        # Ensure we have a working event loop policy
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Configure the event loop when the module is imported
configure_event_loop()

# Replace sqlite3 with pysqlite3 if needed
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage

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
    
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=headless, 
            args=args or ['--no-sandbox']  # Add default args for better stability
        )
        # Create a default context and page
        context = await browser.new_context()
        await context.new_page()
        return browser
    except Exception as e:
        print(f"Error creating browser: {e}")
        raise


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


def get_or_create_event_loop():
    """Get the current event loop or create a new one."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return loop
    except RuntimeError:
        pass
    
    # Create new loop if needed
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine.

    Args:
        coro: The coroutine to run. Coroutine[Any, Any, T]

    Returns:
        T: The result of the coroutine.
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)
