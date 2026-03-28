from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time

chrome_options = Options()
chrome_options.add_argument("--headless")

chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.modot.org/improvei70kc/live-cameras")
time.sleep(5)

logs = driver.get_log("performance")

m3u8_urls = set()

for log in logs:
    message = json.loads(log["message"])["message"]

    if message["method"] == "Network.requestWillBeSent":
        url = message["params"]["request"]["url"]
        if ".m3u8" in url:
            m3u8_urls.add(url)

driver.quit()

for url in m3u8_urls:
    print(url)