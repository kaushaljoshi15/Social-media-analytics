"""
PulseGuard Twitter / X Client & Account Inspector
Inspects brand handles (@Brand) and viral hashtags (#BrandDown)
"""
import random
from typing import List, Dict, Any, Optional

class TwitterClient:
    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token

    def inspect_target(self, query: str, max_tweets: int = 40) -> List[Dict[str, Any]]:
        """
        Inspect a Twitter handle (e.g. @Tesla) or hashtag (e.g. #CrowdStrikeOutage).
        """
        query = query.strip()
        is_hashtag = query.startswith("#")
        is_handle = query.startswith("@")
        target_name = query if (is_hashtag or is_handle) else f"@{query}"

        # Context-aware generator based on whether it's a crisis hashtag or general brand
        is_crisis_query = any(k in query.lower() for k in ["down", "outage", "scam", "boycott", "broken", "bug", "crash", "leak"])

        normal_templates = [
            f"Really enjoying the new features from {target_name}! Super clean UI.",
            f"Anyone else noticed how fast {target_name} is running today? Impressive.",
            f"Just upgraded my setup with {target_name}. Worth every penny!",
            f"Customer service from {target_name} resolved my ticket within an hour. Great support.",
            f"Solid update from {target_name}, hope they keep this momentum going.",
            f"Can someone recommend the best settings for {target_name}?",
            f"Using {target_name} for my daily workflow now. 10/10 experience."
        ]

        crisis_templates = [
            f"Total disaster with {target_name} today! Servers are completely down.",
            f"{target_name} your latest update broke my entire system. Fix this ASAP!",
            f"Been waiting 4 days for a refund response from {target_name}. Unacceptable.",
            f"App keeps crashing on startup after the patch. Anyone else experiencing this on {target_name}?",
            f"Boycott {target_name}! They locked my account without any explanation or warning.",
            f"Massive data breach reported on {target_name}. Change your passwords immediately!",
            f"Why does {target_name} always fail when we need it most? Ridiculous service.",
            f"{target_name} support is practically useless. Automated bots giving copy-paste replies."
        ]

        tweets = []
        for i in range(max_tweets):
            if is_crisis_query:
                text = random.choice(crisis_templates if random.random() < 0.85 else normal_templates)
            else:
                text = random.choice(normal_templates if random.random() < 0.75 else crisis_templates)

            tweets.append({
                "platform": "Twitter",
                "source_target": target_name,
                "author": f"User_{random.randint(1000, 9999)}",
                "text": text,
                "retweets": random.randint(2, 450),
                "likes": random.randint(5, 1200),
                "published_at": "Recent"
            })
        return tweets
