#!/usr/bin/env python3
# Test minimal pour isoler le problème de syntaxe

import asyncio
import json

async def test_generator():
    """Générateur test"""
    yield "test1"
    yield "test2"

async def test_function():
    """Fonction qui utilise le générateur"""
    result = []
    async for item in test_generator():
        result.append(item)
    return result

async def main():
    try:
        result = await test_function()
        print("✅ Test réussi:", result)
    except Exception as e:
        print("❌ Erreur:", e)

if __name__ == "__main__":
    asyncio.run(main())