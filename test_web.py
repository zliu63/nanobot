import asyncio
import sys
sys.path.insert(0, '.')

from nanobot.agent.tools.web import WebSearchTool

async def test_web():
    tool = WebSearchTool()
    result = await tool.execute("test AI money making 2026", count=3)
    print("=== WEB SEARCH TEST RESULT (Blue) ===")
    print(result)
    print("=== END ===")

if __name__ == "__main__":
    asyncio.run(test_web())
