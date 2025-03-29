(videoai) d-Rev@d-Rev videoai % python main.py --timeline

⚠️ MODIFIED DEFAULTS: Processing only channel 1, YouTube upload disabled
2025-03-20 00:51:06 [INFO] main: Arguments: channel=1, steps=script,voice,captions,video,title, delay=60, timeline=True
2025-03-20 00:51:06 [WARNING] main: Timeline processing requested but timeline rendering is disabled in config. Enabling it for this run.
2025-03-20 00:51:06 [INFO] main: Using timeline-aware processing pipeline
2025-03-20 00:51:06 [INFO] main: Processing channel 1
2025-03-20 00:51:06 [INFO] main: Starting timeline-aware processing for Channel 1
2025-03-20 00:51:06 [INFO] main: Running steps: script, voice, captions, video, title
2025-03-20 00:51:06 [INFO] main: Output directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1
2025-03-20 00:51:06 [INFO] timeline_manager: Created v3 timeline with resolution 120x720, 24 fps
2025-03-20 00:51:06 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 00:51:06 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/initial_timeline.json.bak
2025-03-20 00:51:06 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/initial_timeline.json
2025-03-20 00:51:06 [INFO] main: --- Step 1: Generating Script with Timeline Integration ---
Starting script generation process...

Generating script using deepseek-ai/DeepSeek-R1...

Target word count: 50-100 words


Generating script...

<think>
Okay, let's tackle this script. The user wants a YouTube script about zombie plants, specifically resurrection plants from the latest news. First, I need to make sure I understand the key points from the provided news article. The main takeaway is that these plants can survive extreme drought by vitrifying their cells and then revive quickly when water returns. Scientists are looking into their genes to help crops withstand climate change.

The user emphasized including novel thoughts and opinions. So, I should add personal insights, maybe comparing the plants' survival tactics to something relatable. The example scripts used metaphors, so I need to do that too. Also, avoid over-the-top phrases and keep it conversational.

I need to start with an attention-grabbing intro. Maybe mention the plants' "zombie" trait. Then explain the science briefly, using a metaphor like "nature's pause button." Next, connect it to agriculture and climate change, highlighting the potential impact. Include a quote from the article, like Jill Farrant's statement about needing invention. Then add a personal take, perhaps on human ingenuity versus natural solutions. End with the YouTube outro.

Check the rules: no markdown, 50-100 words, first-person opinions, quotes if possible. Avoid cringe language. Make sure it's structured logically but without headers. Let me piece it together, keeping it concise and engaging. Let me count the words to stay within the limit.
</think>

Zombie plants aren't science fiction – they're real botanical survivors rewriting drought resilience. I'm fascinated by how resurrection plants like Selaginella lepidophylla essentially hit nature's pause button through cellular vitrification, turning their insides into biological glass. Jill Farrant's research suggesting these genes could help crops "remember" their dormant seed states feels revolutionary. But here's my take: while gene editing offers hope, we're playing catch-up with 350 million years of plant evolution. The real irony? We're scrambling to borrow survival tricks from organisms we've overlooked, while our agricultural systems crumble under climate pressures we created. Stay tuned, give this video a like, and consider subscribing.

Script generated - 104 words

Extracting topic using gemini-2.0-flash...
Extracted topic: Resurrection Plants and Drought Resilience
Updated topics_covered.json with new topic: Resurrection Plants and Drought Resilience
Script saved to channel directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
Script saved to: /Volumes/drev-ventura/video-repos/videoai/Resurrection Plants and Drought Resilience_generated_script.txt

Script generation completed successfully.
Using dynamic script file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
2025-03-20 00:51:16 [INFO] main: Adding 1 script segments to timeline
2025-03-20 00:51:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 00:51:16 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/script_timeline.json.bak
2025-03-20 00:51:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/script_timeline.json
2025-03-20 00:51:16 [INFO] main: Added script with 104 words to timeline
2025-03-20 00:51:16 [INFO] main: --- Step 2: Generating Voice-over with Timeline Integration ---
Using dynamic script path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
Reading script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt...
Generating voice using ElevenLabs API (voice ID: UgBBYS2sOqTuMpoF3BR0)...
Voice files saved to:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3
Voice generation completed successfully.
Using dynamic voice file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3
2025-03-20 00:51:28 [INFO] main: Adding voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3 to timeline
2025-03-20 00:51:28 [INFO] main: Added complete voice-over (50.10s) to timeline
2025-03-20 00:51:28 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 00:51:28 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/voice_timeline.json.bak
2025-03-20 00:51:28 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/voice_timeline.json
2025-03-20 00:51:28 [INFO] main: Added voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3 to timeline
2025-03-20 00:51:28 [INFO] main: --- Step 3: Generating Captions with Timeline Integration ---
Using dynamic voice file path for captions: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3
Using dynamic captions file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_voice.srt
Generating subtitles from audio: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3
Output SRT path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_voice.srt
Transcribing audio from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3...
Created temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_852cd20608064be99e60df8e8d0d8689.mp3
Transcribing audio using OpenAI Whisper API...
2025-03-20 00:51:35 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/audio/transcriptions "HTTP/1.1 200 OK"
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_852cd20608064be99e60df8e8d0d8689.mp3
Subtitles successfully saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_voice.srt
2025-03-20 00:51:35 [INFO] main: Extracted 9 caption segments from SRT file
2025-03-20 00:51:35 [INFO] main: Added 9 caption segments to timeline
2025-03-20 00:51:35 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 00:51:35 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/captions_timeline.json.bak
2025-03-20 00:51:35 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/captions_timeline.json
2025-03-20 00:51:35 [INFO] main: --- Step 4: Editing Video with Timeline Integration ---
2025-03-20 00:51:35 [INFO] video_edit: Starting video editing process...
2025-03-20 00:51:35 [INFO] video_edit: Timeline mode: True
2025-03-20 00:51:35 [INFO] video_edit: Using provided timeline object for processing
2025-03-20 00:51:35 [INFO] video_edit: Ensuring timeline directory exists: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines
Found 1000 clip entries in the CSV file
Loaded 1000 clips from CSV metadata
Scanning clips directory at /Volumes/drev-ventura/video-repos/videoai/clips for video files...
Final clip count: 1000 valid clips
Clip 1: 1500.mp4, Duration: 5s
Clip 2: 1501.mp4, Duration: 5s
Clip 3: 1502.mp4, Duration: 10s
Clip 4: 1503.mp4, Duration: 5s
Clip 5: 1504.mp4, Duration: 5s
... and 995 more clips
Looking for script in these locations:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/Resurrection Plants and Drought Resilience_generated_script.txt
Successfully read script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_script.txt
2025-03-20 00:51:35 [INFO] video_edit: Looking for voice file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought Resilience_generated_voice.mp3
2025-03-20 00:51:35 [INFO] video_edit: Voice duration: 50.102813 seconds
2025-03-20 00:51:35 [INFO] video_edit: Looking for captions file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_voice.srt
Reading subtitles from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought Resilience_generated_voice.srt
Found 9 subtitle segments
2025-03-20 00:51:35 [INFO] video_edit: Expected number of clip segments: 9
2025-03-20 00:51:35 [INFO] video_edit: Matching clips to script (attempt 1/1)...
2025-03-20 00:53:15 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Available clips: 2000.mp4, 2299.mp4, 1693.mp4, 2232.mp4, 2250.mp4, 2394.mp4, 1757.mp4, 1847.mp4, 1984.mp4, 2220.mp4, 1653.mp4, 2199.mp4, 2340.mp4, 1925.mp4, 2098.mp4, 2065.mp4, 2060.mp4, 1659.mp4, 2019.mp4, 1902.mp4, 2124.mp4, 1677.mp4, 2244.mp4, 2262.mp4, 2042.mp4, 2270.mp4, 2291.mp4, 2248.mp4, 2156.mp4, 1658.mp4, 1922.mp4, 2400.mp4, 1910.mp4, 2043.mp4, 1590.mp4, 2230.mp4, 1884.mp4, 2362.mp4, 1568.mp4, 2125.mp4, 1518.mp4, 1990.mp4, 1545.mp4, 1617.mp4, 1589.mp4, 2062.mp4, 2297.mp4, 2015.mp4, 1610.mp4, 2040.mp4, 2252.mp4, 2480.mp4, 1526.mp4, 1933.mp4, 2035.mp4, 1951.mp4, 1516.mp4, 2476.mp4, 2025.mp4, 2304.mp4, 2410.mp4, 1552.mp4, 2498.mp4, 1878.mp4, 1576.mp4, 2184.mp4, 1584.mp4, 2215.mp4, 1751.mp4, 1563.mp4, 1754.mp4, 2137.mp4, 2459.mp4, 2226.mp4, 2011.mp4, 1812.mp4, 2010.mp4, 2485.mp4, 1897.mp4, 2118.mp4, 2433.mp4, 2422.mp4, 1893.mp4, 2131.mp4, 2203.mp4, 1567.mp4, 2126.mp4, 2167.mp4, 1596.mp4, 1774.mp4, 2412.mp4, 1656.mp4, 2374.mp4, 1547.mp4, 1644.mp4, 2038.mp4, 2069.mp4, 2493.mp4, 1717.mp4, 1678.mp4, 1870.mp4, 1755.mp4, 2268.mp4, 2051.mp4, 2331.mp4, 1713.mp4, 2385.mp4, 2402.mp4, 2446.mp4, 2172.mp4, 2196.mp4, 1963.mp4, 1820.mp4, 1978.mp4, 2089.mp4, 1627.mp4, 2465.mp4, 2423.mp4, 2273.mp4, 1782.mp4, 2477.mp4, 2306.mp4, 2041.mp4, 1569.mp4, 2261.mp4, 2190.mp4, 1708.mp4, 1550.mp4, 2067.mp4, 1942.mp4, 2122.mp4, 1704.mp4, 2066.mp4, 2316.mp4, 2111.mp4, 1564.mp4, 1849.mp4, 1607.mp4, 1852.mp4, 1623.mp4, 2170.mp4, 2272.mp4, 2440.mp4, 2457.mp4, 2009.mp4, 1945.mp4, 2185.mp4, 2189.mp4, 1738.mp4, 2088.mp4, 1539.mp4, 1850.mp4, 2368.mp4, 1826.mp4, 2002.mp4, 2322.mp4, 1635.mp4, 1775.mp4, 1931.mp4, 1833.mp4, 2358.mp4, 2415.mp4, 1711.mp4, 1923.mp4, 2295.mp4, 1822.mp4, 2490.mp4, 1699.mp4, 2169.mp4, 2475.mp4, 1640.mp4, 1642.mp4, 2056.mp4, 2332.mp4, 2017.mp4, 2081.mp4, 1905.mp4, 1779.mp4, 1851.mp4, 2097.mp4, 1740.mp4, 2396.mp4, 2491.mp4, 2421.mp4, 1646.mp4, 2495.mp4, 1915.mp4, 2003.mp4, 1553.mp4, 1867.mp4, 1980.mp4, 1628.mp4, 1669.mp4, 2109.mp4, 2442.mp4, 1895.mp4, 2426.mp4, 2395.mp4, 2444.mp4, 2164.mp4, 1551.mp4, 2263.mp4, 2013.mp4, 1505.mp4, 1882.mp4, 1768.mp4, 1748.mp4, 2200.mp4, 2205.mp4, 1801.mp4, 1542.mp4, 1837.mp4, 2492.mp4, 2222.mp4, 2494.mp4, 1570.mp4, 2279.mp4, 2275.mp4, 1950.mp4, 2310.mp4, 1948.mp4, 2149.mp4, 1565.mp4, 1673.mp4, 1999.mp4, 1799.mp4, 1841.mp4, 1721.mp4, 2034.mp4, 1844.mp4, 2234.mp4, 1839.mp4, 1686.mp4, 1543.mp4, 1637.mp4, 2132.mp4, 1770.mp4, 1649.mp4, 2105.mp4, 1765.mp4, 1729.mp4, 1562.mp4, 1903.mp4, 2046.mp4, 1600.mp4, 1802.mp4, 1961.mp4, 2078.mp4, 1975.mp4, 2288.mp4, 1525.mp4, 2363.mp4, 1761.mp4, 1864.mp4, 2351.mp4, 2240.mp4, 1776.mp4, 1501.mp4, 1783.mp4, 1500.mp4, 2197.mp4, 2307.mp4, 2361.mp4, 2070.mp4, 2093.mp4, 2133.mp4, 1606.mp4, 1603.mp4, 2414.mp4, 1807.mp4, 2251.mp4, 2437.mp4, 1528.mp4, 1987.mp4, 1631.mp4, 2435.mp4, 1797.mp4, 2420.mp4, 1789.mp4, 2481.mp4, 2367.mp4, 1514.mp4, 1876.mp4, 2144.mp4, 2246.mp4, 2217.mp4, 2075.mp4, 2166.mp4, 1938.mp4, 2447.mp4, 2128.mp4, 2084.mp4, 2110.mp4, 1992.mp4, 1626.mp4, 1946.mp4, 2007.mp4, 1883.mp4, 1503.mp4, 2356.mp4, 2150.mp4, 2346.mp4, 2464.mp4, 2296.mp4, 2212.mp4, 2055.mp4, 1681.mp4, 2152.mp4, 1926.mp4, 2333.mp4, 2373.mp4, 2383.mp4, 1772.mp4, 1718.mp4, 2211.mp4, 2413.mp4, 1853.mp4, 1857.mp4, 2277.mp4, 1523.mp4, 1739.mp4, 1819.mp4, 2428.mp4, 1811.mp4, 1764.mp4, 1814.mp4, 2006.mp4, 1834.mp4, 1532.mp4, 1892.mp4, 2175.mp4, 1639.mp4, 1726.mp4, 1972.mp4, 2336.mp4, 1796.mp4, 1597.mp4, 2018.mp4, 1574.mp4, 2334.mp4, 1906.mp4, 2397.mp4, 2417.mp4, 1586.mp4, 2026.mp4, 2323.mp4, 2283.mp4, 1939.mp4, 2380.mp4, 2364.mp4, 2194.mp4, 1858.mp4, 2100.mp4, 2148.mp4, 1650.mp4, 2249.mp4, 1609.mp4, 2147.mp4, 2079.mp4, 2284.mp4, 1985.mp4, 2366.mp4, 1530.mp4, 2256.mp4, 1652.mp4, 1625.mp4, 2320.mp4, 2221.mp4, 2353.mp4, 1953.mp4, 1667.mp4, 2357.mp4, 2059.mp4, 2312.mp4, 2048.mp4, 1614.mp4, 2330.mp4, 1912.mp4, 2409.mp4, 1803.mp4, 1831.mp4, 2112.mp4, 2174.mp4, 1632.mp4, 2047.mp4, 2173.mp4, 1954.mp4, 2119.mp4, 2258.mp4, 2282.mp4, 2474.mp4, 2264.mp4, 1823.mp4, 1964.mp4, 1827.mp4, 2389.mp4, 2369.mp4, 2471.mp4, 1969.mp4, 2074.mp4, 2242.mp4, 1760.mp4, 2106.mp4, 2004.mp4, 2384.mp4, 2431.mp4, 1860.mp4, 1572.mp4, 1509.mp4, 1612.mp4, 1888.mp4, 1620.mp4, 2095.mp4, 1934.mp4, 1671.mp4, 2392.mp4, 1879.mp4, 2016.mp4, 1719.mp4, 2278.mp4, 2021.mp4, 2387.mp4, 2390.mp4, 2107.mp4, 1873.mp4, 2305.mp4, 1731.mp4, 1836.mp4, 2045.mp4, 2087.mp4, 2382.mp4, 2029.mp4, 2138.mp4, 1622.mp4, 1674.mp4, 1956.mp4, 2313.mp4, 1502.mp4, 1989.mp4, 1947.mp4, 2014.mp4, 1696.mp4, 2053.mp4, 2265.mp4, 1991.mp4, 2195.mp4, 1554.mp4, 2463.mp4, 1890.mp4, 1941.mp4, 1512.mp4, 1742.mp4, 2162.mp4, 1715.mp4, 2080.mp4, 2458.mp4, 1557.mp4, 1657.mp4, 1863.mp4, 2139.mp4, 2438.mp4, 2483.mp4, 1672.mp4, 1993.mp4, 2022.mp4, 2141.mp4, 1534.mp4, 1935.mp4, 1868.mp4, 2386.mp4, 1688.mp4, 1865.mp4, 1940.mp4, 2096.mp4, 1566.mp4, 1593.mp4, 1981.mp4, 2127.mp4, 2180.mp4, 1666.mp4, 1856.mp4, 1578.mp4, 2388.mp4, 1909.mp4, 1643.mp4, 2143.mp4, 1735.mp4, 1655.mp4, 2381.mp4, 1687.mp4, 1519.mp4, 1840.mp4, 1919.mp4, 2403.mp4, 2099.mp4, 2058.mp4, 2061.mp4, 1619.mp4, 1875.mp4, 2432.mp4, 2104.mp4, 2267.mp4, 2472.mp4, 1732.mp4, 2177.mp4, 1917.mp4, 1968.mp4, 2153.mp4, 1881.mp4, 1982.mp4, 2202.mp4, 2208.mp4, 1835.mp4, 1988.mp4, 1932.mp4, 1664.mp4, 1556.mp4, 2337.mp4, 2406.mp4, 1916.mp4, 1716.mp4, 1818.mp4, 1918.mp4, 1952.mp4, 2068.mp4, 2231.mp4, 2091.mp4, 2411.mp4, 1960.mp4, 2225.mp4, 2092.mp4, 2245.mp4, 1594.mp4, 2347.mp4, 1592.mp4, 1665.mp4, 2292.mp4, 1752.mp4, 1504.mp4, 1630.mp4, 1698.mp4, 1855.mp4, 2460.mp4, 1615.mp4, 2259.mp4, 2237.mp4, 2207.mp4, 2192.mp4, 1929.mp4, 2123.mp4, 2281.mp4, 2339.mp4, 1977.mp4, 1998.mp4, 2076.mp4, 2377.mp4, 1720.mp4, 2063.mp4, 1601.mp4, 1806.mp4, 1832.mp4, 1707.mp4, 1889.mp4, 2309.mp4, 1766.mp4, 1605.mp4, 2315.mp4, 2398.mp4, 2134.mp4, 2372.mp4, 2115.mp4, 2235.mp4, 2473.mp4, 2399.mp4, 2298.mp4, 1579.mp4, 1706.mp4, 2453.mp4, 2028.mp4, 1896.mp4, 2102.mp4, 1920.mp4, 2114.mp4, 2496.mp4, 1535.mp4, 1538.mp4, 1522.mp4, 1974.mp4, 2427.mp4, 1618.mp4, 2321.mp4, 1769.mp4, 1787.mp4, 2160.mp4, 1587.mp4, 1641.mp4, 1663.mp4, 1825.mp4, 1746.mp4, 2355.mp4, 2461.mp4, 1697.mp4, 1695.mp4, 2031.mp4, 1780.mp4, 2001.mp4, 1524.mp4, 2352.mp4, 1846.mp4, 1577.mp4, 2255.mp4, 2425.mp4, 2179.mp4, 2326.mp4, 1531.mp4, 1885.mp4, 1685.mp4, 1813.mp4, 1966.mp4, 2072.mp4, 1795.mp4, 1558.mp4, 1520.mp4, 2090.mp4, 2247.mp4, 2218.mp4, 1771.mp4, 2328.mp4, 2183.mp4, 2052.mp4, 2314.mp4, 2354.mp4, 1724.mp4, 1753.mp4, 1904.mp4, 2359.mp4, 1880.mp4, 1705.mp4, 1899.mp4, 2484.mp4, 1582.mp4, 2103.mp4, 1861.mp4, 2129.mp4, 2487.mp4, 2140.mp4, 2236.mp4, 1970.mp4, 1955.mp4, 2210.mp4, 2319.mp4, 1602.mp4, 1763.mp4, 1848.mp4, 2308.mp4, 2424.mp4, 2064.mp4, 2094.mp4, 2348.mp4, 2455.mp4, 2101.mp4, 1683.mp4, 1965.mp4, 2008.mp4, 2159.mp4, 1710.mp4, 2302.mp4, 1583.mp4, 1709.mp4, 2441.mp4, 1702.mp4, 2343.mp4, 2378.mp4, 1694.mp4, 1571.mp4, 1928.mp4, 2037.mp4, 1995.mp4, 2254.mp4, 1736.mp4, 2135.mp4, 2071.mp4, 1670.mp4, 2468.mp4, 1645.mp4, 1949.mp4, 2024.mp4, 1575.mp4, 2467.mp4, 2443.mp4, 1830.mp4, 1994.mp4, 1691.mp4, 2257.mp4, 2327.mp4, 1900.mp4, 1784.mp4, 1794.mp4, 1816.mp4, 2085.mp4, 1967.mp4, 1872.mp4, 1817.mp4, 1894.mp4, 1924.mp4, 2360.mp4, 2219.mp4, 2182.mp4, 2223.mp4, 2145.mp4, 1624.mp4, 2338.mp4, 2294.mp4, 1778.mp4, 2233.mp4, 2033.mp4, 2158.mp4, 2274.mp4, 1976.mp4, 1521.mp4, 1785.mp4, 1585.mp4, 2462.mp4, 1937.mp4, 2344.mp4, 2456.mp4, 2188.mp4, 2130.mp4, 1508.mp4, 2452.mp4, 2224.mp4, 2329.mp4, 2206.mp4, 2318.mp4, 1815.mp4, 2450.mp4, 2253.mp4, 1745.mp4, 1921.mp4, 2241.mp4, 1907.mp4, 1871.mp4, 1604.mp4, 1636.mp4, 2401.mp4, 1957.mp4, 2379.mp4, 1661.mp4, 2391.mp4, 2120.mp4, 2451.mp4, 1773.mp4, 2165.mp4, 2470.mp4, 1838.mp4, 2488.mp4, 1800.mp4, 2349.mp4, 1648.mp4, 2285.mp4, 2300.mp4, 1580.mp4, 1843.mp4, 1762.mp4, 2116.mp4, 2478.mp4, 1786.mp4, 1887.mp4, 2479.mp4, 1529.mp4, 1701.mp4, 2054.mp4, 2178.mp4, 1511.mp4, 1886.mp4, 1533.mp4, 1728.mp4, 2289.mp4, 2136.mp4, 2405.mp4, 1679.mp4, 2077.mp4, 2214.mp4, 1997.mp4, 1788.mp4, 2293.mp4, 2083.mp4, 1891.mp4, 1986.mp4, 2430.mp4, 2117.mp4, 2436.mp4, 2499.mp4, 1911.mp4, 2005.mp4, 2012.mp4, 2335.mp4, 2155.mp4, 1810.mp4, 1809.mp4, 1959.mp4, 1700.mp4, 1781.mp4, 1983.mp4, 2023.mp4, 1515.mp4, 2191.mp4, 1913.mp4, 1527.mp4, 1544.mp4, 1629.mp4, 1854.mp4, 2239.mp4, 1598.mp4, 1744.mp4, 1914.mp4, 2439.mp4, 1758.mp4, 1692.mp4, 1690.mp4, 1793.mp4, 2027.mp4, 1675.mp4, 1958.mp4, 1680.mp4, 1828.mp4, 1901.mp4, 1943.mp4, 2198.mp4, 1638.mp4, 1595.mp4, 1654.mp4, 1829.mp4, 2176.mp4, 2271.mp4, 2286.mp4, 1996.mp4, 2365.mp4, 1559.mp4, 1689.mp4, 1842.mp4, 1513.mp4, 2213.mp4, 1866.mp4, 1517.mp4, 2204.mp4, 1540.mp4, 2445.mp4, 1869.mp4, 1859.mp4, 1944.mp4, 1546.mp4, 1971.mp4, 2370.mp4, 2371.mp4, 2466.mp4, 1743.mp4, 2049.mp4, 2290.mp4, 1727.mp4, 1722.mp4, 1804.mp4, 2113.mp4, 2404.mp4, 1874.mp4, 2454.mp4, 2260.mp4, 2229.mp4, 1908.mp4, 1898.mp4, 1616.mp4, 2324.mp4, 1581.mp4, 2342.mp4, 1668.mp4, 1634.mp4, 2469.mp4, 2311.mp4, 2269.mp4, 2482.mp4, 2486.mp4, 1767.mp4, 1821.mp4, 1734.mp4, 1877.mp4, 2039.mp4, 1747.mp4, 2030.mp4, 2418.mp4, 1506.mp4, 2266.mp4, 1973.mp4, 2227.mp4, 1733.mp4, 2161.mp4, 1510.mp4, 2168.mp4, 1608.mp4, 2287.mp4, 2325.mp4, 1549.mp4, 1749.mp4, 1777.mp4, 2036.mp4, 1682.mp4, 1979.mp4, 1712.mp4, 1714.mp4, 1808.mp4, 1647.mp4, 2151.mp4, 2280.mp4, 1676.mp4, 2044.mp4, 1591.mp4, 2350.mp4, 2393.mp4, 1548.mp4, 1703.mp4, 2154.mp4, 2228.mp4, 2434.mp4, 1651.mp4, 1660.mp4, 2345.mp4, 1930.mp4, 2449.mp4, 1862.mp4, 1756.mp4, 2416.mp4, 1633.mp4, 1599.mp4, 2243.mp4, 2429.mp4, 2341.mp4, 1561.mp4, 2301.mp4, 2050.mp4, 2276.mp4, 2193.mp4, 2057.mp4, 2376.mp4, 1573.mp4, 2108.mp4, 2407.mp4, 1845.mp4, 2073.mp4, 1798.mp4, 1962.mp4, 2419.mp4, 2086.mp4, 2209.mp4, 1792.mp4, 1805.mp4, 2216.mp4, 1790.mp4, 2171.mp4, 1555.mp4, 2032.mp4, 1588.mp4, 2020.mp4, 1611.mp4, 2187.mp4, 1723.mp4, 2497.mp4, 1824.mp4, 1684.mp4, 1725.mp4, 2157.mp4, 1541.mp4, 1536.mp4, 2238.mp4, 2408.mp4, 2448.mp4, 1730.mp4, 1741.mp4, 2303.mp4, 1621.mp4, 2489.mp4, 1791.mp4, 1507.mp4, 1537.mp4, 1936.mp4, 2181.mp4, 2317.mp4, 1613.mp4, 1927.mp4, 2146.mp4, 1750.mp4, 2082.mp4, 2201.mp4, 1662.mp4, 2121.mp4, 2142.mp4, 2163.mp4, 2375.mp4, 2186.mp4, 1759.mp4, 1560.mp4, 1737.mp4...
2025-03-20 00:53:15 [WARNING] video_edit: Mismatch: expected 9 segments, but got 10 segments. Re-instructing...
2025-03-20 00:53:15 [WARNING] video_edit: Obtained 10 segments, but expected 9. Continuing anyway.
2025-03-20 00:53:15 [INFO] video_edit: Validating clip sequence...
Total segments to process: 10
Total duration before validation: 50.24 seconds

Processing segment 1/10
Clip: 1551.mp4, Original Duration: 5.44s
Short clip segment: Start=2.30s, Duration=5.44s

Processing segment 2/10
Clip: 2431.mp4, Original Duration: 3.92s
Short clip segment: Start=5.50s, Duration=3.92s

Processing segment 3/10
Clip: 1576.mp4, Original Duration: 3.76s
Short clip segment: Start=0.98s, Duration=3.76s

Processing segment 4/10
Clip: 2362.mp4, Original Duration: 6.32s
Short clip segment: Start=0.00s, Duration=6.32s
Applied safety adjustment: Start=0.00s, Duration=6.32s

Processing segment 5/10
Clip: 2244.mp4, Original Duration: 4.80s
Short clip segment: Start=1.68s, Duration=4.80s

Processing segment 6/10
Clip: 2237.mp4, Original Duration: 7.20s
Short clip segment: Start=0.00s, Duration=7.20s
Applied safety adjustment: Start=0.00s, Duration=7.20s

Processing segment 7/10
Clip: 2372.mp4, Original Duration: 5.84s
Short clip segment: Start=2.72s, Duration=5.84s

Processing segment 8/10
Clip: 2401.mp4, Original Duration: 5.12s
Short clip segment: Start=0.37s, Duration=5.12s

Processing segment 9/10
Clip: 2347.mp4, Original Duration: 3.92s
Short clip segment: Start=5.14s, Duration=3.92s

Processing segment 10/10
Clip: 2481.mp4, Original Duration: 3.92s
Short clip segment: Start=0.46s, Duration=3.92s

Total duration after validation: 50.24 seconds
2025-03-20 00:53:15 [INFO] video_edit: Enforcing maximum clip durations...
2025-03-20 00:53:15 [INFO] video_edit: Using existing timeline and updating with clip sequence...
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 00:53:15 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 00:53:15 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_005315.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_005315.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_005315_visualization.txt.json
2025-03-20 00:53:15 [INFO] video_edit: Timeline 'edit_20250320_005315' created and saved successfully
Warning: Timeline contains no video clips
Timeline integrity validation passed
Error! Input file doesn't exist: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Art Forgery_generated_voice.mp3

---

(videoai) d-Rev@d-Rev videoai % python main.py --timeline

⚠️ MODIFIED DEFAULTS: Processing only channel 1, YouTube upload disabled
2025-03-20 01:05:29 [INFO] main: Arguments: channel=1, steps=script,voice,captions,video,title, delay=60, timeline=True
2025-03-20 01:05:29 [WARNING] main: Timeline processing requested but timeline rendering is disabled in config. Enabling it for this run.
2025-03-20 01:05:29 [INFO] main: Using timeline-aware processing pipeline
2025-03-20 01:05:29 [INFO] main: Processing channel 1
2025-03-20 01:05:29 [INFO] main: Starting timeline-aware processing for Channel 1
2025-03-20 01:05:29 [INFO] main: Running steps: script, voice, captions, video, title
2025-03-20 01:05:29 [INFO] main: Output directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1
2025-03-20 01:05:29 [INFO] timeline_manager: Created v3 timeline with resolution 120x720, 24 fps
2025-03-20 01:05:29 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:05:29 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/initial_timeline.json
2025-03-20 01:05:29 [INFO] main: --- Step 1: Generating Script with Timeline Integration ---
Starting script generation process...

Generating script using deepseek-ai/DeepSeek-R1...
Target word count: 50-100 words
Generating script...
Script generated - 106 words

Extracting topic using gemini-2.0-flash...
Extracted topic: Resurrection Plants and Drought-Resistant Crops
Updated topics_covered.json with new topic: Resurrection Plants and Drought-Resistant Crops
Script saved to channel directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
Script saved to: /Volumes/drev-ventura/video-repos/videoai/Resurrection Plants and Drought-Resistant Crops_generated_script.txt

Script generation completed successfully.
Using dynamic script file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
2025-03-20 01:05:37 [INFO] main: Adding 1 script segments to timeline
2025-03-20 01:05:37 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:05:37 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/script_timeline.json
2025-03-20 01:05:37 [INFO] main: Added script with 106 words to timeline
2025-03-20 01:05:37 [INFO] main: --- Step 2: Generating Voice-over with Timeline Integration ---
Using dynamic script path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
Reading script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt...
Generating voice using ElevenLabs API (voice ID: UgBBYS2sOqTuMpoF3BR0)...
Voice files saved to:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3
Voice generation completed successfully.
Using dynamic voice file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3
2025-03-20 01:05:48 [INFO] main: Adding voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3 to timeline
2025-03-20 01:05:48 [INFO] main: Added complete voice-over (47.07s) to timeline
2025-03-20 01:05:48 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:05:48 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/voice_timeline.json
2025-03-20 01:05:48 [INFO] main: Added voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3 to timeline
2025-03-20 01:05:48 [INFO] main: --- Step 3: Generating Captions with Timeline Integration ---
Using dynamic voice file path for captions: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3
Using dynamic captions file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_voice.srt
Generating subtitles from audio: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3
Output SRT path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_voice.srt
Transcribing audio from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3...
Created temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_c729f1349c8d4963b683ed5a85b8d1a5.mp3
Transcribing audio using OpenAI Whisper API...
2025-03-20 01:05:54 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/audio/transcriptions "HTTP/1.1 200 OK"
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_c729f1349c8d4963b683ed5a85b8d1a5.mp3
Subtitles successfully saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_voice.srt
2025-03-20 01:05:54 [INFO] main: Extracted 12 caption segments from SRT file
2025-03-20 01:05:54 [INFO] main: Added 12 caption segments to timeline
2025-03-20 01:05:54 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:05:54 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/captions_timeline.json
2025-03-20 01:05:54 [INFO] main: --- Step 4: Editing Video with Timeline Integration ---
2025-03-20 01:05:54 [INFO] video_edit: Starting video editing process...
2025-03-20 01:05:54 [INFO] video_edit: Timeline mode: True
2025-03-20 01:05:54 [INFO] video_edit: Using provided timeline object for processing
2025-03-20 01:05:54 [INFO] video_edit: Ensuring timeline directory exists: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines
Found 1000 clip entries in the CSV file
Loaded 1000 clips from CSV metadata
Scanning clips directory at /Volumes/drev-ventura/video-repos/videoai/clips for video files...
Final clip count: 1000 valid clips
Looking for script in these locations:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
Successfully read script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
2025-03-20 01:05:54 [INFO] video_edit: Looking for voice file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Resurrection Plants and Drought-Resistant Crops_generated_voice.mp3
2025-03-20 01:05:54 [INFO] video_edit: Voice duration: 47.072625 seconds
2025-03-20 01:05:54 [INFO] video_edit: Looking for captions file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_voice.srt
Reading subtitles from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_voice.srt
Found 12 subtitle segments
2025-03-20 01:05:54 [INFO] video_edit: Expected number of clip segments: 12
2025-03-20 01:05:54 [INFO] video_edit: Matching clips to script (attempt 1/1)...
2025-03-20 01:08:11 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

2025-03-20 01:08:11 [INFO] video_edit: Successfully matched clip sequence with the correct number of segments.
2025-03-20 01:08:11 [INFO] video_edit: Validating clip sequence...
Total segments to process: 12
Total duration before validation: 46.96 seconds

Total duration after validation: 46.96 seconds
2025-03-20 01:08:11 [INFO] video_edit: Enforcing maximum clip durations...
Adjusting clip '1567.mp4' duration from 1.4400000000000013s to 2.0s for better viewing experience.
2025-03-20 01:08:11 [INFO] video_edit: Using existing timeline and updating with clip sequence...
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:08:11 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:11 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811_visualization.txt.json
2025-03-20 01:08:11 [INFO] video_edit: Timeline 'edit_20250320_010811' created and saved successfully
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:08:11 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:11 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit_visualization.txt.json
2025-03-20 01:08:11 [INFO] video_edit: Current edit timeline saved
2025-03-20 01:08:11 [INFO] video_edit: Creating video sequence...
2025-03-20 01:08:11 [INFO] video_edit: Using dynamic output video file: Resurrection Plants and Drought-Resistant Crops_output_video.mp4
2025-03-20 01:08:11 [INFO] video_edit: Attempting timeline-based rendering (force_fallback=False)...
Timeline-based rendering is disabled. Using fallback approach.
Using timeline-to-clip-sequence conversion for backward compatibility
Warning: Timeline has no video tracks
Error: Could not convert timeline to clip sequence
2025-03-20 01:08:11 [WARNING] video_edit: Timeline-based rendering not available. Using traditional video generation.
2025-03-20 01:08:11 [INFO] timeline_manager: Created v3 timeline with resolution 120x720, 24 fps
2025-03-20 01:08:11 [INFO] timeline_manager: Converted clip sequence with 12 clips to timeline
Timeline integrity validation passed
2025-03-20 01:08:11 [INFO] timeline_manager: Successfully deserialized v3 timeline with 0 video tracks and 1 audio tracks
2025-03-20 01:08:11 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:11 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811_backup_20250320_010811.json
Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811_backup_20250320_010811.json
2025-03-20 01:08:11 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:11 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811.json.bak
2025-03-20 01:08:11 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_010811_visualization.txt.json
Timeline integrity validation passed
2025-03-20 01:08:11 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:11 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit_visualization.txt.json
Timeline created and saved successfully
Using traditional video generation as timeline rendering is not fully implemented yet
Found matching clip with full_path: /Volumes/drev-ventura/video-repos/videoai/clips/1504.mp4
Processing clip with aspect ratio: 9:16 using simple scaling
ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers

Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_61872f48b1614126a2ce599be657f6b0.srt
Successfully added subtitles to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_final_output_with_subtitles.mp4
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_33315e90196740368737b1f0e5c736a5.srt
2025-03-20 01:08:59 [INFO] video_edit: Video editing process completed successfully. Final output: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_final_output_with_subtitles.mp4
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:08:59 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:59 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output_visualization.txt.json
2025-03-20 01:08:59 [INFO] main: Final timeline duration: 47.04 seconds (1129 frames)
2025-03-20 01:08:59 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:08:59 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_timeline.json
2025-03-20 01:08:59 [INFO] timeline_manager: Timeline visualization exported to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timeline_visualizations/timeline_viz_20250320_010859_detailed.txt
2025-03-20 01:08:59 [INFO] main: Timeline visualization exported to: True
2025-03-20 01:08:59 [INFO] main: --- Step 5: Generating Title & Description ---
Using dynamic script path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
Generating title and description for channel 1
Reading script from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Resurrection Plants and Drought-Resistant Crops_generated_script.txt
Generating title...
Generating description...
YouTube info saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/youtube_info_channel1.json
YouTube info also saved to /Volumes/drev-ventura/video-repos/videoai/youtube_info.json

Generated Title:
Monsanto's HIDDEN Tech: Resurrection Plants Will TERRIFY You!

Generated Description:
Resurrection plants defy death! WTFLOW explores Selaginella's drought survival & potential for crops. Is it evolution's cheat code or a risky gamble? https://www.youtube.com/@w_t_flow Like & Subscribe for more!
2025-03-20 01:09:00 [INFO] main: Channel 1 timeline-aware processing completed successfully.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742422143.134484  637492 init.cc:232] grpc_wait_for_shutdown_with_timeout() timed out.


---

(videoai) d-Rev@d-Rev videoai % python main.py --timeline

⚠️ MODIFIED DEFAULTS: Processing only channel 1, YouTube upload disabled
2025-03-20 01:19:18 [INFO] main: Arguments: channel=1, steps=script,voice,captions,video,title, delay=60, timeline=True
2025-03-20 01:19:18 [WARNING] main: Timeline processing requested but timeline rendering is disabled in config. Enabling it for this run.
2025-03-20 01:19:18 [INFO] main: Using timeline-aware processing pipeline
2025-03-20 01:19:18 [INFO] main: Processing channel 1
2025-03-20 01:19:18 [INFO] main: Starting timeline-aware processing for Channel 1
2025-03-20 01:19:18 [INFO] main: Running steps: script, voice, captions, video, title
2025-03-20 01:19:18 [INFO] main: Output directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1
2025-03-20 01:19:18 [INFO] timeline_manager: Created v3 timeline with resolution 120x720, 24 fps
2025-03-20 01:19:18 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:19:18 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/initial_timeline.json.bak
2025-03-20 01:19:18 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/initial_timeline.json
2025-03-20 01:19:18 [INFO] main: --- Step 1: Generating Script with Timeline Integration ---
Starting script generation process...

Generating script using deepseek-ai/DeepSeek-R1...

Target word count: 50-100 words


Generating script...

<think>
Okay, let's tackle this script. The user wants a YouTube script based on the latest news provided. The topics include Zelensky and resurrection plants. The latest news has two parts: one about resurrection plants and another about Zelensky's call with Trump.

First, I need to pick one topic. The user mentioned preferring topics that align with the latest news. Both are part of the latest news, but the resurrection plants story is more science-focused, while Zelensky's is political. Since the example given earlier was about cats but the latest news includes Zelensky, maybe the user expects the Zelensky topic. Plus, the resurrection plants might be less timely compared to the political update.

Next, the script needs to be 50-100 words, spoken narrative without formatting. Must include quotes from the latest news if available. The user's personal opinion is required, using first-person statements. Avoid over-the-top language and follow the rules strictly.

Looking at the Zelensky news: Trump and Zelensky had a positive call after Trump spoke with Putin. Zelensky says lasting peace possible this year. Both sides accuse each other of attacks despite ceasefire talks. Prisoner exchange happened. Need to highlight the key points: the call, the ceasefire efforts, the mutual accusations, and the prisoner swap.

Quotes to include: Zelensky's "We believe that together with America... lasting peace can be achieved this year." Also, mention Trump's statement about aligning requests and needs. Mention the prisoner exchange as a positive step.

Personal take: Maybe express skepticism about the ceasefire holding, given the history and recent attacks. Compare it to fragile agreements that look good on paper but hard to maintain. Use a metaphor like a house of cards or a delicate ecosystem.

Check rules: No markdown, conversational, novel thoughts, first-person, outro with "Stay tuned...".

Putting it all together: Start with the main news, quote Zelensky, mention Trump's involvement, the ceasefire rejection by Putin, recent attacks, prisoner swap, then personal opinion on the fragility, end with outro.
</think>

Zelensky’s latest call with Trump feels like watching two firefighters argue while the house burns. The Ukrainian president called it "positive" and "substantive," claiming "lasting peace can be achieved this year." But let’s be real—Putin rejected a full ceasefire, and both sides immediately launched attacks after their talks. Even the prisoner swap, while significant, feels like rearranging deck chairs on the Titanic. I’m skeptical of grand promises when missiles keep flying. Peace deals without mutual trust are like trying to grow crops in a desert—optimistic, but the roots aren’t deep enough. Stay tuned, give this video a like, and consider subscribing.

Script generated - 101 words

Extracting topic using gemini-2.0-flash...
Extracted topic: Zelensky and Trump's call and potential peace in Ukraine
Updated topics_covered.json with new topic: Zelensky and Trump's call and potential peace in Ukraine
Script saved to channel directory: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
Script saved to: /Volumes/drev-ventura/video-repos/videoai/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt

Script generation completed successfully.
Using dynamic script file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
2025-03-20 01:19:27 [INFO] main: Adding 1 script segments to timeline
2025-03-20 01:19:27 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:19:27 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/script_timeline.json.bak
2025-03-20 01:19:27 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/script_timeline.json
2025-03-20 01:19:27 [INFO] main: Added script with 101 words to timeline
2025-03-20 01:19:27 [INFO] main: --- Step 2: Generating Voice-over with Timeline Integration ---
Using dynamic script path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
Reading script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt...
Generating voice using ElevenLabs API (voice ID: UgBBYS2sOqTuMpoF3BR0)...
Voice files saved to:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3
Voice generation completed successfully.
Using dynamic voice file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3
2025-03-20 01:19:37 [INFO] main: Adding voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3 to timeline
2025-03-20 01:19:37 [INFO] main: Added complete voice-over (41.93s) to timeline
2025-03-20 01:19:37 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:19:37 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/voice_timeline.json.bak
2025-03-20 01:19:37 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/voice_timeline.json
2025-03-20 01:19:37 [INFO] main: Added voice-over from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3 to timeline
2025-03-20 01:19:37 [INFO] main: --- Step 3: Generating Captions with Timeline Integration ---
Using dynamic voice file path for captions: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3
Using dynamic captions file path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
Generating subtitles from audio: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3
Output SRT path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
Transcribing audio from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3...
Created temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_257b285dc3744e26b5d463c46ee653e1.mp3
Transcribing audio using OpenAI Whisper API...
2025-03-20 01:19:45 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/audio/transcriptions "HTTP/1.1 200 OK"
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_257b285dc3744e26b5d463c46ee653e1.mp3
Subtitles successfully saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
2025-03-20 01:19:45 [INFO] main: Extracted 12 caption segments from SRT file
2025-03-20 01:19:45 [INFO] main: Added 12 caption segments to timeline
2025-03-20 01:19:45 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:19:45 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/captions_timeline.json.bak
2025-03-20 01:19:45 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/captions_timeline.json
2025-03-20 01:19:45 [INFO] main: --- Step 4: Editing Video with Timeline Integration ---
2025-03-20 01:19:45 [INFO] video_edit: Starting video editing process...
2025-03-20 01:19:45 [INFO] video_edit: Timeline mode: True
2025-03-20 01:19:45 [INFO] video_edit: Using provided timeline object for processing
2025-03-20 01:19:45 [INFO] video_edit: Ensuring timeline directory exists: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines
Found 1000 clip entries in the CSV file
Loaded 1000 clips from CSV metadata
Scanning clips directory at /Volumes/drev-ventura/video-repos/videoai/clips for video files...
Final clip count: 1000 valid clips
Clip 1: 1500.mp4, Duration: 5s
Clip 2: 1501.mp4, Duration: 5s
Clip 3: 1502.mp4, Duration: 10s
Clip 4: 1503.mp4, Duration: 5s
Clip 5: 1504.mp4, Duration: 5s
... and 995 more clips
Looking for script in these locations:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
Successfully read script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
2025-03-20 01:19:45 [INFO] video_edit: Looking for voice file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/audio/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.mp3
2025-03-20 01:19:48 [INFO] video_edit: Voice duration: 41.9265 seconds
2025-03-20 01:19:48 [INFO] video_edit: Looking for captions file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
Reading subtitles from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
Found 12 subtitle segments
2025-03-20 01:19:48 [INFO] video_edit: Expected number of clip segments: 12
2025-03-20 01:19:48 [INFO] video_edit: Matching clips to script (attempt 1/1)
2025-03-20 01:21:53 [WARNING] video_edit: Mismatch: expected 12 segments, but got 11 segments. Re-instructing...
2025-03-20 01:21:53 [WARNING] video_edit: Obtained 11 segments, but expected 12. Continuing anyway.
2025-03-20 01:21:53 [INFO] video_edit: Validating clip sequence...
Total segments to process: 11
Total duration before validation: 38.14 seconds

Processing segment 1/11
Clip: 1543.mp4, Original Duration: 3.54s
Short clip segment: Start=0.74s, Duration=3.54s

Processing segment 2/11
Clip: 1553.mp4, Original Duration: 2.12s
Short clip segment: Start=2.19s, Duration=2.12s

Processing segment 3/11
Clip: 1612.mp4, Original Duration: 4.76s
Short clip segment: Start=0.02s, Duration=4.76s

Processing segment 4/11
Clip: 1520.mp4, Original Duration: 2.42s
Short clip segment: Start=1.74s, Duration=2.42s

Processing segment 5/11
Clip: 1548.mp4, Original Duration: 4.86s
Short clip segment: Start=4.03s, Duration=4.86s

Processing segment 6/11
Clip: 1583.mp4, Original Duration: 2.78s
Short clip segment: Start=0.13s, Duration=2.78s

Processing segment 7/11
Clip: 2329.mp4, Original Duration: 4.30s
Short clip segment: Start=0.40s, Duration=4.30s

Processing segment 8/11
Clip: 2175.mp4, Original Duration: 1.84s
Short clip segment: Start=1.62s, Duration=1.84s

Processing segment 9/11
Clip: 1502.mp4, Original Duration: 4.08s
Short clip segment: Start=0.13s, Duration=4.08s

Processing segment 10/11
Clip: 1571.mp4, Original Duration: 3.88s
Short clip segment: Start=0.74s, Duration=3.88s

Processing segment 11/11
Clip: 2261.mp4, Original Duration: 3.56s
Short clip segment: Start=1.32s, Duration=3.56s

Total duration after validation: 38.14 seconds
2025-03-20 01:21:53 [INFO] video_edit: Enforcing maximum clip durations...
Adjusting clip '2175.mp4' duration from 1.8399999999999999s to 2.0s for better viewing experience.
2025-03-20 01:21:53 [INFO] video_edit: Using existing timeline and updating with clip sequence...
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:21:53 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:21:53 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012153.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012153.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012153_visualization.txt.json
2025-03-20 01:21:53 [INFO] video_edit: Timeline 'edit_20250320_012153' created and saved successfully
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:21:53 [INFO] timeline_manager: Successfully deserialized v3 timeline with 0 video tracks and 1 audio tracks
2025-03-20 01:21:53 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:21:53 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit_backup_20250320_012153.json
Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit_backup_20250320_012153.json
2025-03-20 01:21:53 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:21:53 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit.json.bak
2025-03-20 01:21:53 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/current_edit_visualization.txt.json
2025-03-20 01:21:53 [INFO] video_edit: Current edit timeline saved
2025-03-20 01:21:53 [INFO] video_edit: Creating video sequence...
2025-03-20 01:21:53 [INFO] video_edit: Using dynamic output video file: Zelensky and Trump's call and potential peace in Ukraine_output_video.mp4
2025-03-20 01:21:53 [INFO] video_edit: Attempting timeline-based rendering (force_fallback=False)...
Timeline-based rendering is disabled. Using fallback approach.
Using timeline-to-clip-sequence conversion for backward compatibility
Warning: Timeline has no video tracks
Error: Could not convert timeline to clip sequence
2025-03-20 01:21:53 [WARNING] video_edit: Timeline-based rendering not available. Using traditional video generation.
2025-03-20 01:21:53 [INFO] timeline_manager: Created v3 timeline with resolution 120x720, 24 fps
2025-03-20 01:21:54 [INFO] timeline_manager: Converted clip sequence with 11 clips to timeline
Timeline integrity validation passed
2025-03-20 01:21:54 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:21:54 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012154.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012154.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/edit_20250320_012154_visualization.txt.json
Timeline integrity validation passed
2025-03-20 01:21:54 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:21:54 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit.json.bak
2025-03-20 01:21:54 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/latest_edit_visualization.txt.json
Timeline created and saved successfully
Using traditional video generation as timeline rendering is not fully implemented yet
Found matching clip with full_path: /Volumes/drev-ventura/video-repos/videoai/clips/1543.mp4
Processing clip with aspect ratio: 9:16 using simple scaling
ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers
.
.
.
[Parsed_aloop_2 @ 0x7fd380905ec0] [Eval @ 0x7ff7b5462888] Invalid chars 's' at the end of expression '2s'
[Parsed_aloop_2 @ 0x7fd380905ec0] Unable to parse option value "2s"
Error applying option 'size' to filter 'aloop': Invalid argument
Failed to set value '[0:a]volume=1.4[main];[1:a]volume=0.4,aloop=loop=-1:size=2s[bgm];[main][bgm]amix=inputs=2:duration=first[aout]' for option 'filter_complex': Invalid argument
Error parsing global options: Invalid argument
Error adding background music: Command '['ffmpeg', '-y', '-i', '/Volumes/drev-ventura/video-repos/videoai/temp_files/temp_voice_merge_30c13f59edc940d0ad1872bd7fedcc1f.mp4', '-i', '/Volumes/drev-ventura/video-repos/videoai/background_music/background_music.mp3', '-filter_complex', '[0:a]volume=1.4[main];[1:a]volume=0.4,aloop=loop=-1:size=2s[bgm];[main][bgm]amix=inputs=2:duration=first[aout]', '-map', '0:v', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', "/Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output.mp4"]' returned non-zero exit status 234.
Error output: None
Using voice-only version as fallback
Cleaned up temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/temp_voice_merge_30c13f59edc940d0ad1872bd7fedcc1f.mp4
2025-03-20 01:22:31 [INFO] video_edit: Adding subtitles to final video...
Using dynamic final video file: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output.mp4
Using dynamic captions file: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_voice.srt
Using dynamic final subtitled video file: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output_with_subtitles.mp4
Created temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_de753d6e29f8424597ffa0c0c0340fc9.srt
Using drawtext filter for subtitles (basic but reliable)
Found 12 subtitle entries
Burning subtitles directly into video...
Created temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_c57d2f1d7fff477d83009276d00fc5c7.srt
ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers
  built with Apple clang version 16.0.0 (clang-1600.0.26.4)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/7.1_4 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags='-Wl,-ld_classic' --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libharfbuzz --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox
  libavutil      59. 39.100 / 59. 39.100
  libavcodec     61. 19.100 / 61. 19.100
  libavformat    61.  7.100 / 61.  7.100
  libavdevice    61.  3.100 / 61.  3.100
  libavfilter    10.  4.100 / 10.  4.100
  libswscale      8.  3.100 /  8.  3.100
  libswresample   5.  3.100 /  5.  3.100
  libpostproc    58.  3.100 / 58.  3.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    encoder         : Lavf61.7.100
  Duration: 00:00:41.93, start: 0.000000, bitrate: 4842 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1080x1920 [SAR 1:1 DAR 9:16], 4768 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
      Metadata:
        handler_name    : VideoHandler
        vendor_id       : [0][0][0][0]
        encoder         : Lavc61.19.100 libx264
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 72 kb/s (default)
      Metadata:
        handler_name    : SoundHandler
        vendor_id       : [0][0][0][0]
[Parsed_subtitles_0 @ 0x7f9f09804f40] libass API version: 0x1703000
[Parsed_subtitles_0 @ 0x7f9f09804f40] libass source: tarball: 0.17.3
[Parsed_subtitles_0 @ 0x7f9f09804f40] Shaper: FriBidi 1.0.15 (SIMPLE) HarfBuzz-ng 10.1.0 (COMPLEX)
[Parsed_subtitles_0 @ 0x7f9f09804f40] Using font provider coretext
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> h264 (libx264))
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
[Parsed_subtitles_0 @ 0x7f9f29712740] libass API version: 0x1703000
[Parsed_subtitles_0 @ 0x7f9f29712740] libass source: tarball: 0.17.3
[Parsed_subtitles_0 @ 0x7f9f29712740] Shaper: FriBidi 1.0.15 (SIMPLE) HarfBuzz-ng 10.1.0 (COMPLEX)
[Parsed_subtitles_0 @ 0x7f9f29712740] Using font provider coretext
[Parsed_subtitles_0 @ 0x7f9f29712740] fontselect: (DIN Condensed Bold, 400, 0) -> /System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf, -1, DINCondensed-Bold
[libx264 @ 0x7f9f2a005900] using SAR=1/1
[libx264 @ 0x7f9f2a005900] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
[libx264 @ 0x7f9f2a005900] profile High, level 4.0, 4:2:0, 8-bit
[libx264 @ 0x7f9f2a005900] 264 - core 164 r3108 31e19f9 - H.264/MPEG-4 AVC codec - Copyleft 2003-2023 - http://www.videolan.org/x264.html - options: cabac=1 ref=2 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=6 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=24 lookahead_threads=4 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=1 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=30 rc=crf mbtree=1 crf=22.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to '/Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output_with_subtitles.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    encoder         : Lavf61.7.100
  Stream #0:0(und): Video: h264 (avc1 / 0x31637661), yuv420p(tv, progressive), 1080x1920 [SAR 1:1 DAR 9:16], q=2-31, 30 fps, 15360 tbn (default)
      Metadata:
        handler_name    : VideoHandler
        vendor_id       : [0][0][0][0]
        encoder         : Lavc61.19.100 libx264
      Side data:
        cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 72 kb/s (default)
      Metadata:
        handler_name    : SoundHandler
        vendor_id       : [0][0][0][0]
[out#0/mp4 @ 0x7f9f2a006bc0] video:21804KiB audio:369KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.201195%
frame= 1256 fps= 76 q=-1.0 Lsize=   22218KiB time=00:00:41.80 bitrate=4354.3kbits/s speed=2.52x    
[libx264 @ 0x7f9f2a005900] frame I:7     Avg QP:17.98  size:154823
[libx264 @ 0x7f9f2a005900] frame P:601   Avg QP:21.49  size: 25372
[libx264 @ 0x7f9f2a005900] frame B:648   Avg QP:24.44  size:  9251
[libx264 @ 0x7f9f2a005900] consecutive B-frames: 13.4% 51.3%  6.7% 28.7%
[libx264 @ 0x7f9f2a005900] mb I  I16..4: 14.1% 67.7% 18.3%
[libx264 @ 0x7f9f2a005900] mb P  I16..4:  3.4%  9.1%  0.6%  P16..4: 32.4%  9.8%  3.5%  0.0%  0.0%    skip:41.2%
[libx264 @ 0x7f9f2a005900] mb B  I16..4:  1.2%  3.4%  0.2%  B16..8: 21.0%  3.7%  0.2%  direct: 7.7%  skip:62.6%  L0:46.7% L1:45.8% BI: 7.5%
[libx264 @ 0x7f9f2a005900] 8x8 transform intra:69.4% inter:76.1%
[libx264 @ 0x7f9f2a005900] coded y,uvDC,uvAC intra: 43.0% 67.0% 23.6% inter: 9.5% 16.3% 0.3%
[libx264 @ 0x7f9f2a005900] i16 v,h,dc,p: 30% 26% 14% 30%
[libx264 @ 0x7f9f2a005900] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 27% 21% 20%  4%  5%  6%  6%  6%  5%
[libx264 @ 0x7f9f2a005900] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 27% 21% 12%  5%  9%  8%  8%  6%  5%
[libx264 @ 0x7f9f2a005900] i8c dc,h,v,p: 50% 20% 22%  9%
[libx264 @ 0x7f9f2a005900] Weighted P-Frames: Y:0.7% UV:0.5%
[libx264 @ 0x7f9f2a005900] ref P L0: 82.3% 17.7%
[libx264 @ 0x7f9f2a005900] ref B L0: 90.1%  9.9%
[libx264 @ 0x7f9f2a005900] ref B L1: 99.0%  1.0%
[libx264 @ 0x7f9f2a005900] kb/s:4266.30
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_c57d2f1d7fff477d83009276d00fc5c7.srt
Successfully added subtitles to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output_with_subtitles.mp4
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_de753d6e29f8424597ffa0c0c0340fc9.srt
2025-03-20 01:22:48 [INFO] video_edit: Video editing process completed successfully. Final output: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_final_output_with_subtitles.mp4
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 01:22:48 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:22:48 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output.json.bak
2025-03-20 01:22:48 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_output_visualization.txt.json
2025-03-20 01:22:48 [INFO] main: Final timeline duration: 41.92 seconds (1006 frames)
2025-03-20 01:22:48 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 01:22:48 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_timeline.json.bak
2025-03-20 01:22:48 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timelines/final_timeline.json
2025-03-20 01:22:48 [INFO] timeline_manager: Timeline visualization exported to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/timeline_visualizations/timeline_viz_20250320_012248_detailed.txt
2025-03-20 01:22:48 [INFO] main: Timeline visualization exported to: True
2025-03-20 01:22:48 [INFO] main: --- Step 5: Generating Title & Description ---
Using dynamic script path: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
Generating title and description for channel 1
Reading script from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/Zelensky and Trump's call and potential peace in Ukraine_generated_script.txt
Generating title...
Generating description...
YouTube info saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_1/youtube_info_channel1.json
YouTube info also saved to /Volumes/drev-ventura/video-repos/videoai/youtube_info.json

Generated Title:
Zelensky's TRUMP Call: Is Ukraine's Peace Deal a PUTIN TRAP?!

Generated Description:
Zelensky & Trump's call: Peace or just talk? WTF is really going on? Putin's playing games, and Ukraine's still under attack. Check out WTFLOW's take! https://www.youtube.com/@w_t_flow Like & Subscribe for more!
2025-03-20 01:22:49 [INFO] main: Channel 1 timeline-aware processing completed successfully.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742422972.247311  653197 init.cc:232] grpc_wait_for_shutdown_with_timeout() timed out.


---


Transcribing audio using OpenAI Whisper API...
2025-03-20 02:49:55 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/audio/transcriptions "HTTP/1.1 200 OK"
Removed temporary file: /Volumes/drev-ventura/video-repos/videoai/temp_files/tmp_c003dea99f84430e99ac985d684343a2.mp3
Subtitles successfully saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_voice.srt
2025-03-20 02:49:55 [INFO] main: Extracted 15 caption segments from SRT file
2025-03-20 02:49:55 [INFO] main: Added 15 caption segments to timeline
2025-03-20 02:49:55 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:49:55 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/captions_timeline.json
2025-03-20 02:49:55 [INFO] main: --- Step 4: Editing Video with Timeline Integration ---
2025-03-20 02:49:55 [INFO] video_edit: Starting video editing process...
2025-03-20 02:49:55 [INFO] video_edit: Timeline mode: True
2025-03-20 02:49:55 [INFO] video_edit: Using provided timeline object for processing
2025-03-20 02:49:55 [INFO] video_edit: Ensuring timeline directory exists: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines
Found 1000 clip entries in the CSV file
Loaded 1000 clips from CSV metadata
Scanning clips directory at /Volumes/drev-ventura/video-repos/videoai/clips for video files...
Final clip count: 1000 valid clips
Clip 1: 1500.mp4, Duration: 5s
Clip 2: 1501.mp4, Duration: 5s
Clip 3: 1502.mp4, Duration: 10s
Clip 4: 1503.mp4, Duration: 5s
Clip 5: 1504.mp4, Duration: 5s
... and 995 more clips
Looking for script in these locations:
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_script.txt
  - /Volumes/drev-ventura/video-repos/videoai/Newly Released JFK Files_generated_script.txt
Successfully read script from /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_script.txt
2025-03-20 02:49:56 [INFO] video_edit: Looking for voice file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/audio/Newly Released JFK Files_generated_voice.mp3
2025-03-20 02:49:56 [INFO] video_edit: Voice duration: 44.016313 seconds
2025-03-20 02:49:56 [INFO] video_edit: Looking for captions file at: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_voice.srt
Reading subtitles from: /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/Newly Released JFK Files_generated_voice.srt
Found 15 subtitle segments
2025-03-20 02:49:56 [INFO] video_edit: Expected number of clip segments: 15
2025-03-20 02:49:56 [INFO] video_edit: Matching clips to script (attempt 1/1)...
2025-03-20 02:52:16 [INFO] httpx: HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

2025-03-20 02:52:16 [WARNING] video_edit: Mismatch: expected 15 segments, but got 14 segments. Re-instructing...
2025-03-20 02:52:16 [WARNING] video_edit: Obtained 14 segments, but expected 15. Continuing anyway.
2025-03-20 02:52:16 [INFO] video_edit: Validating clip sequence...
Total segments to process: 14
Total duration before validation: 41.00 seconds

Total duration after validation: 41.00 seconds
2025-03-20 02:52:16 [INFO] video_edit: Enforcing maximum clip durations...
Adjusting clip '1588.mp4' duration from 1.88s to 2.0s for better viewing experience.
2025-03-20 02:52:16 [INFO] video_edit: Using existing timeline and updating with clip sequence...
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216_visualization.txt.json
2025-03-20 02:52:16 [INFO] video_edit: Timeline 'edit_20250320_025216' created and saved successfully
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/current_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/current_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/current_edit_visualization.txt.json
Warning: Timeline contains no video clips
Timeline integrity validation passed
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/final_timeline.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/final_timeline.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/final_timeline_visualization.txt.json
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline visualization exported to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timeline_visualizations/timeline_viz_20250320_025216_detailed.txt
2025-03-20 02:52:16 [INFO] video_edit: Final timeline visualization exported to: True
2025-03-20 02:52:16 [INFO] video_edit: Current edit and final timelines saved
2025-03-20 02:52:16 [INFO] video_edit: Creating video sequence...
2025-03-20 02:52:16 [INFO] video_edit: Using dynamic output video file: Newly Released JFK Files_output_video.mp4
2025-03-20 02:52:16 [INFO] video_edit: Attempting timeline-based rendering (force_fallback=False)...
Timeline-based rendering is disabled. Using fallback approach.
Using timeline-to-clip-sequence conversion for backward compatibility
Warning: Timeline has no video tracks
Error: Could not convert timeline to clip sequence
2025-03-20 02:52:16 [WARNING] video_edit: Timeline-based rendering not available. Using traditional video generation.
2025-03-20 02:52:16 [INFO] timeline_manager: Created v3 timeline with resolution 1080x1920, 30 fps
2025-03-20 02:52:16 [INFO] timeline_manager: Converted clip sequence with 14 clips to timeline
Timeline integrity validation passed
2025-03-20 02:52:16 [INFO] timeline_manager: Successfully deserialized v3 timeline with 0 video tracks and 1 audio tracks
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216_backup_20250320_025216.json
Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216_backup_20250320_025216.json
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Created backup of existing timeline at /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216.json.bak
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/edit_20250320_025216_visualization.txt.json
Timeline integrity validation passed
2025-03-20 02:52:16 [INFO] timeline_manager: Timeline JSON validated successfully against schema
2025-03-20 02:52:16 [INFO] timeline_manager: Saved timeline to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/latest_edit.json
Timeline successfully output to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/latest_edit.json
Timeline visualization saved to /Volumes/drev-ventura/video-repos/videoai/outputs/channel_3/timelines/latest_edit_visualization.txt.json
Timeline created and saved successfully
Using traditional video generation as timeline rendering is not fully implemented yet
Found matching clip with full_path: /Volumes/drev-ventura/video-repos/videoai/clips/1559.mp4
Processing clip with aspect ratio: 9:16 using simple scaling
ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers