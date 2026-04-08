import time
import requests

def get_category_links(category_name):
    url = "https://en.wikipedia.org/w/api.php"
    
    # add a user agent header, otherwise Wikipedia blocks requests without one
    headers = {
        "User-Agent": "MyWikiScript/2.0"
    }
    
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": 500,# max per request
        "format": "json"
    }
    
    links = []
    while True:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 429:
            print("Rate limited, waiting 60s...")
            time.sleep(60) 
            continue
        
        # check for HTTP errors before parsing JSON
        response.raise_for_status()
        
        data = response.json()
        print('data = ', data)
        members = data["query"]["categorymembers"]
        print('members = ', members)
        links += [
            f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}"
            for m in members
        ]
        
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    
    return links



def get_article_content(title):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MyWikiScript/1.0"}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,  # clean plain text, no markup
        "titles": title,
        "format": "json"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 429:
            print("Rate limited, waiting 15 minutes...")
            time.sleep(60*15) 
    
    response.raise_for_status()
    data = response.json()
    
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))  # get the single page
    
    return page.get("extract", "")

def get_category_content(links):
    dataset = []
    for i, url in enumerate(links):
        title = url.split("/wiki/")[-1].replace("_", " ")
        try:
            content = get_article_content(title)
            if content:
                dataset.append({
                    "title": title,
                    "content": content
                })
            
            # keep track of progress
            if i % 50 == 0:
                print(f"Progress: {i}/{len(links)}")
        except Exception as e:
            print(f"Skipped {title}: {e}")
    return dataset


