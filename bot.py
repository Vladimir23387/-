import logging
import os
import re
import asyncio
import time
import io
from dotenv import load_dotenv
import nest_asyncio

import openai
import faiss
import numpy as np

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Применяем nest_asyncio для корректной работы вложенных циклов событий
nest_asyncio.apply()

# Загружаем переменные окружения из файла .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# URL-ы фотографий руководителей
PHOTO_PRESIDENT = "https://static.tildacdn.com/tild6139-3564-4162-a636-613162376438/ca38d0e45e485eeaa214.jpg"
# URL для фото руководителя тюменского отделения (Эдуарда Омарова)
PHOTO_VP_URL = "https://static.tildacdn.com/tild3033-3264-4032-a362-393764633238/12312312314444444444.jpg"  

# Преимущества членства 
MEMBERSHIP_ADVANTAGES = (
    "• Развитие профессиональных навыков: Участие в образовательных программах, таких как 'Школа управления проектами' и 'Вечерняя школа предпринимателя'. Доступ к мастер-классам, вебинарам и интенсивам, которые проводят ведущие эксперты.\n"
    "• Расширение деловых связей: Возможность участвовать в бизнес-играх, таких как 'Ты — предприниматель', где участники создают реальную сеть контактов. Членство в клубах самозанятых и предпринимателей для обмена опытом и поиска партнёров. Закрытый канал телеграм предпринимателей региона.\n"
    "• Юридическая и административная поддержка: Бесплатные консультации юристов по вопросам ведения бизнеса. Доступ к готовым шаблонам документов.\n"
    "• Индивидуальные консультации: Возможность получить рекомендации от экспертов в области бизнеса и управления. Личный разбор бизнес-задач участников.\n"
    "• Экономические выгоды: Участие в госпрограммах поддержки бизнеса. Скидки и эксклюзивные предложения от партнёров сообщества."
)

# --- Функции для работы с базой знаний ---

def convert_docx_to_txt(input_filename: str = "russia_base.docx", output_filename: str = "russia_base.txt") -> bool:
    try:
        from docx import Document
    except ImportError:
        logger.error("Не установлена библиотека python-docx. Установите её: pip install python-docx")
        return False
    try:
        document = Document(input_filename)
        full_text = [para.text for para in document.paragraphs]
        text = "\n".join(full_text)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Файл {input_filename} успешно конвертирован в {output_filename} (UTF-8).")
        return True
    except Exception as e:
        logger.error(f"Ошибка при конвертации файла {input_filename}: {e}")
        return False

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < text_length else text_length
    return chunks

# Функция получения эмбеддинга с повторными попытками
async def get_embedding(chunk: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    retries = 3
    for attempt in range(retries):
        try:
            result = await loop.run_in_executor(
                None, lambda: openai.Embedding.create(input=chunk, model="text-embedding-ada-002", request_timeout=30)
            )
            vector = np.array(result["data"][0]["embedding"], dtype="float32")
            return vector
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга для чанка (попытка {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return np.zeros(1536, dtype="float32")

async def load_and_index_base_async(filename: str = "russia_base.txt", docx_filename: str = "russia_base.docx", chunk_size: int = 500, overlap: int = 50):
    if not os.path.exists(filename):
        if os.path.exists(docx_filename):
            success = convert_docx_to_txt(docx_filename, filename)
            if not success:
                logger.error("Не удалось получить текстовую версию базы знаний.")
                return faiss.IndexFlatL2(1536), []
        else:
            logger.warning(f"Файл базы знаний {docx_filename} не найден. Продолжаем без базы знаний.")
            return faiss.IndexFlatL2(1536), []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Ошибка чтения файла базы знаний: {e}")
        return faiss.IndexFlatL2(1536), []
    
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))
    embeddings = np.stack(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"Создан FAISS индекс с {index.ntotal} элементами")
    return index, chunks

# Глобальные переменные для базы знаний
global_index = None
global_chunks = []

# --- Функция для генерации ответа через OpenAI GPT ---
async def generate_gpt_response(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": '''Ты — помощник Telegram-бота общественной организации "Опора России". Твоя задача — предоставить пользователю информацию о мероприятиях, услугах и преимуществах членства в организации. Используй краткие и информативные ответы на русском языке.
Реализуй человекоподобное общение: используй приветственные сообщения, фото руководителя и тональность, отражающую ценности сообщества.
Информируй о предстоящих мероприятиях с предоставлением ссылок на регистрацию.
Презентуй преимущества членства и их активно продавай (работай с возражениями, подсвечивая ключевые выгоды).
Проактивно предлагай услуги сообщества, имитируя стиль менеджера по продажам.
Мягко и ненавязчиво предлагай вступить в сообщество.
Используй только материалы файла russia_base.docx.
При вопросе о членских взносах отвечай, что они составляют 10 000 руб./год и оплачиваются либо банковской картой (Visa, Mastercard, МИР) через Альфа-Банк, либо наличными при получении.'''},

                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500,
        )
        text = response.choices[0].message["content"].strip()
        return text
    except Exception as e:
        logger.error(f"Ошибка вызова OpenAI: {e}")
        return "Извините, произошла ошибка при формировании ответа."

# --- Обработчики команд ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "Здравствуйте!\n\n"
        "Добро пожаловать в бот общественной организации \"Опора России\". "
        "Здесь вы можете узнать информацию о мероприятиях, услугах и преимуществах членства в нашей организации.\n\n"
        "Доступные команды:\n"
        "/start — запустить бота\n"
        "/help — справочная информация\n"
        "/stop — остановить бота (только для администратора)\n\n"
        "Выберите интересующую опцию ниже:"
    )
    keyboard = [
        [InlineKeyboardButton("Актуальные мероприятия", callback_data="events")],
        [InlineKeyboardButton("Преимущества членства", callback_data="membership")],
        [InlineKeyboardButton("Получить помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Справка по командам:\n\n"
        "/start — запустить бота и получить приветственное сообщение\n"
        "/help — получить справочную информацию\n"
        "/stop — остановить бота (только для администратора)\n\n"
        "Также вы можете отправлять текстовые сообщения для получения консультаций."
    )
    await update.message.reply_text(help_text)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Бот останавливается. До свидания!")
    context.application.stop()

# --- Обработчик inline-кнопок ---
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "events":
        text = (
            "Предстоящие мероприятия:\n"
            "• Мастер-класс от Геннадия Агапкина 'Делегирование как искусство: как строить команду для роста' — Регистрация: " + "http://forms.yandex.ru/u/6777dbb8..." +"\n"
            "• IV Всероссийский конкурс «Наследие выдающихся предпринимателей России» — Подробнее: http://vk.com/away.php?to=https://xn--80abkcbbjhf1atbh1aqfc.xn--p1ai/moscow&utf=1\n"
        )
        await query.edit_message_text(text=text)
    elif data == "membership":
        text = (
            "Преимущества членства в «Опора России»:\n" + MEMBERSHIP_ADVANTAGES +
            "\n\nХотите узнать, как вступить в наше сообщество? Нажмите кнопку ниже."
        )
        keyboard = [[InlineKeyboardButton("Вступить", callback_data="join")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=text, reply_markup=reply_markup)
    elif data == "join":
        text = (
            "Мы рады, что вы заинтересованы стать частью нашего динамичного сообщества!\n"
            "Для вступления заполните онлайн-заявку по ссылке:\n"
            "https://opora.ru/about/kak-stat-chlenom-opory-rossii/\n\n"
            "Если у вас есть вопросы, напишите их здесь — я с удовольствием помогу."
        )
        await query.edit_message_text(text=text)
    elif data == "help":
        text = (
            "Задайте ваш вопрос, и я постараюсь помочь.\n"
            "Например: \"Как получить консультацию по юридическим вопросам?\""
        )
        await query.edit_message_text(text=text)
    elif data == "show_photos":
        # Отправляем фотографии для федерального и тюменского уровней
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=PHOTO_PRESIDENT,
            caption="Александр Калинин — Президент «Опора России»"
        )
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=PHOTO_VP_URL,
            caption="Эдуард Омаров — Руководитель тюменского отделения"
        )
    else:
        await query.edit_message_text(text="Извините, не смог распознать команду.")

# --- Обработчик текстовых сообщений ---
async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text_original = update.message.text.strip()
    user_text = user_text_original.lower()

    # 1. Обработка запроса "Кто сейчас возглавляет Опору России? Покажи справку с фото." – федеральный уровень
    if re.search(r"\bкто\s+(?:сейчас\s+)?возглавляет\s+опору\s+россии\b", user_text) and re.search(r"покажи.*(фото|справка)", user_text):
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=PHOTO_PRESIDENT,
            caption="Александр Калинин — Президент «Опора России»"
        )
        answer = (
            "Александр Калинин возглавляет Опору России на федеральном уровне. "
            "Он осуществляет стратегическое руководство организацией и является её президентом."
        )
        await update.message.reply_text(answer)
        return

    # 2. Обработка запроса "Кто руководит тюменским отделением?" (в любых формулировках)
    if re.search(r"(руковод|глав|началь).*тюм", user_text):
        # Отправляем фото руководителя тюменского отделения
        try:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=PHOTO_VP_URL,
                caption="Эдуард Омаров — Руководитель тюменского отделения"
            )
        except Exception as e:
            logger.error(f"Ошибка отправки фото по URL: {e}")
            await update.message.reply_text(
                f"Не удалось отправить фотографию. Пожалуйста, перейдите по ссылке, чтобы увидеть фото: {PHOTO_VP_URL}"
            )
        answer = (
            "В тюменском отделении руководитель — Эдуард Омаров."
        )
        await update.message.reply_text(answer)
        return

    # 3. Обработка запроса на фото руководителя Александра Калинина (если отдельно)
    if "фото" in user_text and "калинин" in user_text:
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=PHOTO_PRESIDENT,
            caption="Александр Калинин — Президент «Опора России»"
        )
        info = (
            "Александр Калинин возглавляет организацию на федеральном уровне и осуществляет руководство поддержкой предпринимательства."
        )
        await update.message.reply_text(info)
        return

    # 4. Обработка запроса на фото руководителя Эдуарда Омарова (если отдельно)
    if "фото" in user_text and "омаров" in user_text:
        logger.info("Запрос на фото Эдуарда Омарова обнаружен. Используем метод отправки по URL.")
        try:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=PHOTO_VP_URL,
                caption="Эдуард Омаров — Руководитель тюменского отделения"
            )
            await update.message.reply_text(
                "Эдуард Омаров возглавляет тюменское отделение и отвечает за развитие предпринимательства в Тюменской области."
            )
        except Exception as e:
            logger.error(f"Ошибка отправки фото по URL: {e}")
            await update.message.reply_text(
                f"Не удалось отправить фотографию. Пожалуйста, перейдите по ссылке, чтобы увидеть фото: {PHOTO_VP_URL}"
            )
        return

    # 5. Обработка вопросов вида "А кто же такой Александр Калинин?"
    if re.search(r"калинин.*кто", user_text):
        answer = (
            "Александр Калинин — Президент Общероссийской общественной организации «Опора России». "
            "Он осуществляет стратегическое руководство организацией и отвечает за развитие бизнес-среды в стране."
        )
        await update.message.reply_text(answer)
        return

    # 6. Обработка общего запроса о руководстве организацией
    if ("руковод" in user_text or "глав" in user_text or "началь" in user_text) and ("опора россии" in user_text or "руководство" in user_text):
        if "калинин" in user_text or "федераль" in user_text:
            answer = "На федеральном уровне руководитель организации — Александр Калинин."
            await update.message.reply_text(answer)
            return
        elif "омаров" in user_text or "тюм" in user_text:
            answer = "В тюменском отделении руководитель — Эдуард Омаров."
            await update.message.reply_text(answer)
            return
        else:
            answer = (
                "Федеральное руководство осуществляется Александром Калининым, а руководство тюменского отделения — Эдуардом Омаровым."
            )
            await update.message.reply_text(answer)
            return

    # 7. Обработка возражений (например, "А если у меня не хватит времени...")
    if "не хватит" in user_text or "нет времени" in user_text:
        answer = (
            "Понимаем, что время — ценный ресурс. Членство в «Опора России» организовано так, "
            "что вы можете участвовать в мероприятиях и получать консультации даже при плотном графике. "
            "Мы предлагаем разнообразные форматы: онлайн-сессии, краткие встречи и записи вебинаров, чтобы вы могли выбирать тот формат, который подходит именно вам."
        )
        await update.message.reply_text(answer)
        return

    # 8. Если запрос не соответствует специальным условиям — используем FAISS для поиска релевантных чанков
    if global_index is not None and global_chunks:
        try:
            loop = asyncio.get_running_loop()
            embedding_result = await loop.run_in_executor(
                None, lambda: openai.Embedding.create(input=user_text_original, model="text-embedding-ada-002")
            )
            query_embedding = np.array(embedding_result["data"][0]["embedding"], dtype="float32").reshape(1, -1)
            k = 3
            distances, indices = global_index.search(query_embedding, k)
            retrieved_chunks = [global_chunks[i] for i in indices[0] if i < len(global_chunks)]
            context_text = "\n\n".join(retrieved_chunks)
            final_prompt = (
                f"Ниже приведены релевантные отрывки из базы знаний:\n{context_text}\n\n"
                f"Вопрос: {user_text_original}\nОтвет:"
            )
        except Exception as e:
            logger.error(f"Ошибка при поиске в базе знаний: {e}")
            final_prompt = (
                f"Пользователь задал вопрос: '{user_text_original}'. "
                "Дай развернутый, дружелюбный и профессиональный ответ с учетом ценностей сообщества 'Опора России'."
            )
    else:
        final_prompt = (
            f"Пользователь задал вопрос: '{user_text_original}'. "
            "Дай развернутый, дружелюбный и профессиональный ответ с учетом ценностей сообщества 'Опора России'."
        )

    response = await generate_gpt_response(final_prompt)
    await update.message.reply_text(response)

# --- Основная функция для запуска бота ---
async def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Загружаем базу знаний и создаём FAISS индекс (асинхронно)
    global global_index, global_chunks
    global_index, global_chunks = await load_and_index_base_async()

    # Регистрируем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stop", stop_command))

    # Регистрируем обработчики inline-кнопок и текстовых сообщений
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    await app.run_polling(close_loop=False)

if __name__ == '__main__':
    asyncio.run(main())
