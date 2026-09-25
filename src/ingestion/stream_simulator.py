"""
PulseGuard Multi-Platform Poisson Stream Simulator
Covers GTU Unit 5 (Poisson Process Modeling)
Feeds incoming comments across YouTube, Twitter, Reddit, and Telegram in real time
with an instant 'Inject Crisis' toggle for viva demonstration.
"""
import random
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from src.config import PLATFORMS, POISSON_LAMBDA_NORMAL, POISSON_LAMBDA_CRISIS

class StreamSimulator:
    def __init__(self):
        self.is_crisis_active = False
        self.crisis_theme = "Service Outage & Refund Delays"

        # Baseline positive & neutral discussions
        self.normal_comments = [
            ("YouTube", "The display color accuracy is impressive for video editing.", "TechEnthusiast"),
            ("Twitter", "Really loving the fast performance after today's patch! 🚀", "DailyGamer"),
            ("Reddit", "Build quality feels very solid, aluminum frame has no flex at all.", "LaptopGeek"),
            ("Telegram", "Customer service helped resolve my onboarding query within minutes.", "MobileUser"),
            ("YouTube", "Sound quality from the speakers is surprisingly rich and loud.", "AudioGuy"),
            ("Twitter", "Battery lasted a full 10 hours during my travel today. Highly satisfied!", "TravelBlogger"),
            ("Reddit", "Ubuntu 24.04 works out of the box with sleep and Wi-Fi working fine.", "DevOpsPro"),
            ("Telegram", "Just received the delivery today, packaging was pristine.", "OnlineShopper"),
            ("YouTube", "Cleanest design on a tech product this year. Good job team.", "DesignFan"),
            ("Twitter", "Great value for money compared to the overpriced competitors.", "SmartBuyer"),
            ("Reddit", "Thermals are well controlled during light programming and web browsing.", "SysadminDave"),
            ("Telegram", "Smooth checkout experience on the mobile application.", "FinTechFan")
        ]

        # Sudden viral crisis backlash comments
        self.crisis_comments = [
            ("Twitter", "🚨 Boycott this company! They took my money and the product won't even power on!", "FuriousBuyer"),
            ("YouTube", "Total scam! Battery died after 45 minutes and customer support is ghosting me. 😡", "DisgustedUser"),
            ("Reddit", "Fatal kernel crash on boot! 50 laptops in our office are now bricked. DO NOT BUY!", "AngryAdmin"),
            ("Telegram", "Server timeout error 500! Cannot process any withdrawals or refunds!", "PanicTrader"),
            ("Twitter", "Broken hardware and pathetic customer care. Demanding an immediate full refund!", "RipOffVictim"),
            ("YouTube", "Massive overheating issue reaching 98C! Smells like burnt plastic. 🤮", "HardwareReview"),
            ("Reddit", "Data leak alert! My account was compromised and password reset emails are failing.", "SecResearcher"),
            ("Telegram", "Worst customer support in history. Automated bots just running in loops. 💩", "StuckUser"),
            ("Twitter", "Class action lawsuit incoming! You cannot sell defective junk and refuse returns.", "LegalEagle"),
            ("YouTube", "Screen flickering constantly and audio buzzing. Total hardware defect!", "ScreenDisaster"),
            ("Reddit", "Fraudulent advertising! Features promised on the box do not even exist in firmware.", "TruthTeller"),
            ("Telegram", "App crashed and deleted my entire unsaved work! Unusable trash! 🤬", "LostWorkUser")
        ]

    def set_crisis_mode(self, active: bool, theme: str = "Outage & Defect Backlash"):
        """Toggle crisis mode on or off for live viva demonstration."""
        self.is_crisis_active = active
        self.crisis_theme = theme

    def generate_comment(self, forced_platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a single incoming comment according to current state (Normal vs. Crisis).
        Uses Poisson probability weighting for crisis comments.
        """
        # In crisis mode: 85% probability of angry/toxic backlash
        if self.is_crisis_active and random.random() < 0.85:
            platform, text, author = random.choice(self.crisis_comments)
        else:
            # In normal mode: 88% normal, 12% minor complaint
            if random.random() < 0.12:
                platform, text, author = random.choice(self.crisis_comments)
            else:
                platform, text, author = random.choice(self.normal_comments)

        chosen_platform = forced_platform if (forced_platform and forced_platform != "All") else platform

        return {
            "platform": chosen_platform,
            "source_target": f"Live Stream ({chosen_platform})",
            "author": f"{author}_{random.randint(10, 999)}",
            "raw_text": text,
            "created_at": datetime.datetime.now().isoformat()
        }

    def generate_burst(self, count: int = 20, forced_platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate a micro-batch of comments simulating Poisson arrival burst."""
        return [self.generate_comment(forced_platform) for _ in range(count)]
