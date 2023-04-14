from telegram.ext import Application, CommandHandler, MessageHandler, filters

import config
import handlers


def main():
    application = Application.builder().token(config.TOKEN).build()
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(
        MessageHandler(filters.PHOTO, handlers.photo)
    )
    application.run_polling()


if __name__ == "__main__":
    main()
