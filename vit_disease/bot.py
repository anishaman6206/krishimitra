# bot.py
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from vit_model import predict_disease
from llm_helper import generate_disease_info
from twilio_helper import send_via_twilio   # 👈 import here

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌱 Send me a clear photo of a plant leaf, and I'll tell you the disease & cure."
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    image_path = f"temp_{update.message.from_user.id}.jpg"
    await file.download_to_drive(image_path)

    try:
        disease_label = predict_disease(image_path)
        cure_info = generate_disease_info(disease_label)
        final_message = f"🩺 Prediction: {disease_label}\n\n{cure_info}"

        # Reply to Telegram user
        await update.message.reply_text(final_message, parse_mode="Markdown")

        # Forward to SMS & WhatsApp
        send_via_twilio(final_message)

    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
