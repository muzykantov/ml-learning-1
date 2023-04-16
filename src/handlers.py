from io import BytesIO

from PIL import Image
from telegram import Update
from telegram.ext import ContextTypes

import ml.recognize as recognize


async def start(
    update: Update,
    _: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Send a message when the command /start is issued."""

    user = update.effective_user
    await update.message.reply_text(
        text=(f"Привет, {user.first_name}! Отправь мне фотографию, и я "
              "попробую распознать цифру на ней.")
    )


async def photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Process photo and send recognized value."""

    file_id = update.message.photo[-1].file_id
    photo = await context.bot.get_file(file_id)
    photo_data = BytesIO(await photo.download_as_bytearray())

    image = Image.open(photo_data)
    digits = recognize.recognize_digits(image)

    if digits is not None:
        await update.message.reply_text(f"Распознанная цифра: {digits}")
    else:
        await update.message.reply_text("Не удалось распознать цифру на "
                                        "фотографии.")
