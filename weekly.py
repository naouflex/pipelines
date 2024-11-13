async def init_browser(self):
    self.browser = await create_async_playwright_browser(headless=self.valves.HEADLESS) 