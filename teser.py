import asyncio
import aiohttp
import json

async def test_upscale():
    """Test the upscale endpoint with a real image"""
    url = "http://localhost:8000/upscale"
    
    payload = {
        "img_url": "https://cdn.hswstatic.com/gif/how-to-draw-buildings-15.jpg",
        "scale": 2
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                print(f"Status: {response.status}")
                print(json.dumps(result, indent=2))
                
                if response.status == 200 and result.get("success"):
                    print(f"\n✓ Upscaling successful!")
                    print(f"  Original: {result['original_size']['width']}x{result['original_size']['height']}")
                    print(f"  Upscaled: {result['upscaled_size']['width']}x{result['upscaled_size']['height']}")
                    print(f"  Time: {result['processing_time']}s")
                    print(f"  File: {result['file_path']}")
                else:
                    print(f"\n✗ Upscaling failed: {result.get('error')}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing upscale endpoint with scale=2...")
    asyncio.run(test_upscale())