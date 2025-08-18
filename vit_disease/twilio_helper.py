# twilio_helper.py
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Twilio credentials from .env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Numbers
TWILIO_SMS_NUMBER = os.getenv("TWILIO_SMS_NUMBER")              # e.g. +123456789
TARGET_SMS_NUMBER = os.getenv("TARGET_SMS_NUMBER")              # e.g. +919876543210
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")               # Twilio sandbox number
TARGET_WHATSAPP_NUMBER = os.getenv("TARGET_WHATSAPP_NUMBER")    # e.g. whatsapp:+91XXXXXXXXXX

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_via_twilio(message: str):
    """
    Send message via Twilio to both SMS and WhatsApp.
    """
    # Send SMS
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_SMS_NUMBER,
            to=TARGET_SMS_NUMBER
        )
        print("✅ SMS sent successfully.")
    except Exception as e:
        print(f"❌ SMS send failed: {e}")

    # Send WhatsApp
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=TARGET_WHATSAPP_NUMBER
        )
        print("✅ WhatsApp message sent successfully.")
    except Exception as e:
        print(f"❌ WhatsApp send failed: {e}")
