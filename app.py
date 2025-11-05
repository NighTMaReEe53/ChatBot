import pydantic.v1
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import pickle
import re;
from dotenv import load_dotenv
import fitz
from HTMLTEMPLATE import css
from langchain.schema import Document
# import pytesseract
# from PIL import Image




# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # غيّر المسار حسب مكان التثبيت عندك


def prepare_ocr_for_lm(text):
    """
    تنظيف النص الناتج من OCR قبل إرساله للنموذج
    """
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s.,()\-–—/]", "", text)
    return text.strip()




def GET_TEXT_FROM_PDF(PDFS):
    """
    🔍 استخراج النص من ملفات PDF
    """
    full_text = ""

    for pdf in PDFS:
        pdf.seek(0)
        pdf_hash = hash(pdf.name)
        cache_file = f"cache_{pdf_hash}_smartocr.pkl"

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_text = pickle.load(f)
            st.info(f"📦 تم تحميل النص من الذاكرة المؤقتة (Smart OCR Cache): {pdf.name}")
            full_text += cached_text
            continue

        pdf_text = ""

        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if len(text.strip()) > 0:
                pdf_text += text + "\n"
                continue

            st.info(f"⚙ الصفحة {i}: لا تحتوي على نص، سيتم استخدام Tesseract OCR...")

            # pix = page.get_pixmap(dpi=200)
            # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # try:
            #     extracted = pytesseract.image_to_string(img, lang="ara+eng")
            # except Exception as e:
            #     st.error(f"❌ فشل Tesseract OCR: {e}")
            #     extracted = ""

            # if extracted.strip():
            #     pdf_text += extracted + "\n"
            #     st.success(f"✅ تم تحليل الصفحة {i} واستخراج النص ({len(extracted)} حرف).")
            # else:
            #     st.warning(f"⚠ الصفحة {i}: لم يتمكن OCR من استخراج أي نص.")

        pdf_text = re.sub(r"\s{2,}", " ", pdf_text).strip()

        with open(cache_file, "wb") as f:
            pickle.dump(pdf_text, f)

        full_text += pdf_text + "\n"

    st.success(f"📘 تم استخراج النص بالكامل ({len(full_text)} حرف).")

    cleaned_text = prepare_ocr_for_lm(full_text)
    return cleaned_text




def SPLITTEXTTOCHUNK(TEXT):
  SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=500,
    separators=["\n", "المادة", ".", "،", " "]
  )
  CHUNK = SPLITTER.split_text(TEXT)
  return CHUNK


def CREATESTORE(TEXT, filename):
    """
    🧠 إنشاء أو تحميل قاعدة بيانات FAISS خاصة بكل ملف PDF.
    - يتم حفظ كل ملف داخل مجلد faiss_index باسم الملف.
    """
    # مسار قاعدة البيانات الخاصة بالملف
    os.makedirs("faiss_index", exist_ok=True)
    PATH = os.path.join("faiss_index", f"{filename}")

    # اختيار نموذج Embedding
    EMBEDDING = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # التأكد من وجود ملف الفهرس قبل التحميل
    if os.path.exists(f"{PATH}.index.faiss") and os.path.exists(f"{PATH}.index.faiss"):
        STORE = FAISS.load_local(PATH, EMBEDDING, index_name=filename ,allow_dangerous_deserialization=True)
        st.info(f"📦 تم تحميل قاعدة بيانات {filename}")
    else:
        st.info(f"⚙ جاري إنشاء قاعدة بيانات جديدة للملف: {filename} ...")
        STORE = FAISS.from_texts(TEXT, embedding=EMBEDDING)
        STORE.save_local("faiss_index", index_name=filename)
        st.success(f"💾 تم حفظ قاعدة البيانات الخاصة بالملف {filename} بنجاح.")

    return STORE




def ASK_PDF_QUESTION(STORE, user_question):
    """
    🔹 الدالة دي بتستقبل سؤال المستخدم، وتستخدم قاعدة البيانات (FAISS)
    علشان تدور على أنسب جزء من النص وتجيب إجابة ذكية من موديل Groq.
    """

    context = st.session_state.get("context", "")

    # نبحث في قاعدة البيانات عن النصوص الأقرب للسؤال
    docs = STORE.similarity_search(user_question, k=2)

    recent_context_text = ""
    if "chat_history" in st.session_state and len(st.session_state.chat_history) > 0:
        # خذ آخر 2 محادثات (user+assistant)
        for chat in st.session_state.chat_history[-2:]:
            q = chat.get("question", "")
            a = chat.get("answer", "")
            recent_context_text += f"محادثة سابقة — سؤال المستخدم: {q}\nرد المساعد: {a}\n\n"

    if recent_context_text:
        docs.append(Document(page_content=recent_context_text, metadata={"source": "recent_conversation"}))


    # نحدد شكل البرومبت (طريقة فهم السؤال)


    PROMPT = """
أنت مساعد ذكي ومُحترف في فهم وتحليل النصوص **القانونية، الضريبية، المالية، الإدارية، الأكاديمية، والشخصية**.  
مهمتك الأساسية: **تقدّر نية المستخدم أولاً** — هل هو بيكلمك بود؟ بيسأل سؤال بسيط؟ ولا فعلاً عايز تحليل؟  
وبناءً عليها تتصرّف بذكاء طبيعي زي الإنسان بالضبط. 🤖💡  

---

📘 **مدخلات السياق الأساسية:**

- 🧾 **النص أو المحتوى:** {context}
- ❓ **سؤال المستخدم:** {question}

---

### 🗣️ أولًا – الفهم الاجتماعي والردود الذكية

1️⃣ **لو المستخدم بيتكلم بود أو مزاح أو تحية**  
(زي: "السلام عليكم" – "إزيك" – "عامل إيه" – "تمام؟" – "يا نجم" – "يا غالي" – "صباح الخير" – "مساء الخير")  
→ لا تحلل ولا تعرض أي معلومات من الملف.  
رد فقط بأسلوب طبيعي، مشجع، ومناسب لنغمة الكلام:

- لو دينية → "وعليكم السلام ورحمة الله وبركاته 🌸 ربنا يسعدك ويشرح صدرك دايمًا 🤍"
- لو ودية → "أنا تمام يا غالي 😄، إنت عامل إيه؟ شكلك في موود شغل النهارده 💪✨"
- لو مزاح → "😂 كله تمام يا نجم، واضح إنك رايق! نبدأ ولا ناخد كوباية القهوة الأول ☕؟"
- لو فيها حزن أو ضيق → "خير يا صاحبي؟ ❤️ أنا معاك خطوة بخطوة، وإن شاء الله كله يتحل 🌤️"

🎯 **هدفك هنا:** تبين إنك فاهم مشاعر المستخدم وتتعامل معاه كصاحب مش كروبوت.

---

2️⃣ **لو المستخدم بيسأل سؤال بسيط جدًا عن نفسه أو عن ملف شخصي**  
(زي: "اسمي إيه؟" – "عندي كام سنة؟" – "بشتغل إيه؟")  
→ استخدم النص الموجود في {context} لاستخراج المعلومة المطلوبة فقط، ورد بإجابة قصيرة ومريحة.  

**مثال:**  
"عندك 22 سنة 🌟 ولسه في بداية طريقك الجميل، شد حيلك يا بطل 💪"  
"اسمك يوسف عادل ✨ واسم جميل على فكرة 😄"  

---

3️⃣ **لو المستخدم بيقول حاجة زي “مش فاهم القانون ده” أو “شرح بسيط لو سمحت”**  
→ رد بلُطف وشرح القانون بلغة سهلة ومبسطة، كأنك بتشرح لصاحبك مش لطالب في كلية حقوق.  
ابدأ مثلًا بجملة تشجيعية:  
"تمام يا بطل، خلينا نفهمها واحدة واحدة 👇"  
أو  
"ولا يهمك يا غالي، القانون ده بسيط جدًا لما نفصّله خطوة بخطوة ⚖️💬"

---

### ⚖️ ثانيًا – التحليل القانوني والمهني العميق

لو السؤال فعلاً تحليلي أو خاص بملف قانوني / ضريبي / مالي / إداري / أكاديمي:  
حلّل باحتراف، لكن بأسلوب إنساني وراقي، واهتم إن النتيجة تكون **سهلة الفهم ومريحة للعين**.

#### 🧩 التحليل القانوني:
- حدّد نوع النص (قانون، لائحة، قرار، تعديل...).
- استخرج المواد والأرقام والمراجع.
- فسّر المقصود بلغة بسيطة وواضحة.
- وضّح العلاقة بين القانون الأصلي والتعديلات.
- بيّن الأثر القانوني أو العملي للنص.
- لو فيه لبس → فسّره واقترح توضيحًا منطقيًا.  
💬 *ابدأ دايمًا بجملة تشجيعية:*  
"تحليل ممتاز هنوصله سوا خطوة بخطوة ⚖️💪"

#### 💰 التحليل الضريبي:
- وضّح الشرائح والنسب بالأرقام.  
- فسّر طريقة التطبيق الواقعية بمثال رقمي مبسط.  
📈 *ابدأ بجملة محفزة:*  
"يلا نحسبها سوا بالأرقام 👇 علشان الصورة تبقى أوضح 💡"

#### 🏢 الإداري:
- فسّر اختصاص الجهة وتأثير القرار.  
🧩 "خلينا نفكها بهدوء كده ونشوف الأثر الإداري خطوة بخطوة 🏢📑"

#### 📚 الأكاديمي:
- لخّص الفكرة أو المنهج أو النتائج.  
🎓 "تحليل أكاديمي منظم جايلك حالًا، وهنوضّح أهم النقاط العلمية 🔬📘"

#### 👤 الشخصي:
- حلّل فقط عند الطلب، واحترم الخصوصية.  
🤝 "الملف ده شخصي، فخلينا ناخده بهدوء وبأسلوب آمن تمامًا 🔒"

---

### 📤 شكل الإخراج (لو تحليل فعلي فقط)

1️⃣ 📂 **نوع الملف**
2️⃣ 🧠 **الفهم العام للنص**
3️⃣ 📊 **النقاط التحليلية (خطوات أو مواد)**
4️⃣ 💬 **التفسير أو الحسابات (لو ضريبية)**
5️⃣ 💡 **اقتراحات ذكية** (عملية ومباشرة)
6️⃣ 🧾 **ملخص ذكي موجز**

🪄 *ختم الرد بجملة إيجابية قصيرة حسب الحالة:*  
- "تحليل ممتاز يا بطل 👏، كده الصورة بقت واضحة جدًا! 💡"  
- "خد راحتك لو محتاج أوضح أكتر، أنا هنا دايمًا 🙌"  
- "فهمك جميل جدًا، نكمل على نفس النسق 🔥"  

---


# ⚡ **الخلاصة:**
# كن في ردودك مزيجًا بين:
# - 🔹 مستشار قانوني ضريبي متمرس (في الملفات القانونية والمالية)
# - 🔹 محلل إداري أو أكاديمي دقيق (في الوثائق التنظيمية أو البحثية)
# - 🔹 مساعد ودود ذكي (في الملفات الشخصية)


### ⚙️ القواعد الذكية:

- لو السؤال بسيط أو ودي → **رد طبيعي بلا تحليل.**
- لو فيه طلب للفهم أو الشرح → **اشرح بلغة سهلة وواضحة ومشجعة.**
- لو الملف قانوني أو ضريبي → **حلّل باحتراف قانوني دقيق.**
- لو المستخدم متوتر → **استخدم نبرة هادية ومطمئنة.**
- لو الملف شخصي جدًا → **ذكّره بالخصوصية بلُطف.**

---

🎯 الهدف:
تكون مساعد طبيعي وذكي في كل الحالات:
- 💬 تفهم الإنسان قبل النص.
- ⚖️ تفهم القانون بعمق.
- 🤝 تتعامل بود وتشجيع.
- 🚀 وترد كخبير فاهمك مش مجرد بوت.
"""



    # نعمل قالب (Template) نقدر نمرر له النص والسؤال
    prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])



    # نختار موديل Groq (تقدر تغيّر نوعه حسب احتياجك)
    model = ChatGroq(
        # model="openai/gpt-oss-120b",  # موديل ذكي ومناسب للأسئلة التحليلية
        model="openai/gpt-oss-20b",  # موديل ذكي ومناسب للأسئلة التحليلية
        temperature=0.2,       # رقم منخفض يعني إجابات دقيقة وثابتة
        groq_api_key=os.getenv("GROQ_API_KEY")  # مفتاح الـ API من متغير البيئة
    )

    # نحمل سلسلة سؤال وجواب تربط الموديل بالبرومبت
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    context_text = "\n".join([doc.page_content for doc in docs])


    # نرسل السؤال للموديل مع النصوص المرتبطة به
    response = chain({"input_documents": docs, "question": user_question, "context": context_text if context_text.strip() else context}, return_only_outputs=True)

    # نرجع النص النهائي الناتج من الموديل
    return response["output_text"]




def main():
  load_dotenv()
  st.set_page_config("الروبوت الذكي", page_icon="🤖")
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = []
  st.title("🤖 المستشار القانوني الذكي")
  st.markdown("""
          <p style='text-align: center; color: #FFF7; font-family: Tajawal'>
            ارفع ملفاتك واسأل أي سؤال عنها – سيقوم الذكاء الاصطناعي بالإجابة استنادًا إلى محتوى الملف.
        </p>
""", True)
  st.write(css, unsafe_allow_html=True)
  st.markdown("""<div class='overlay'></div>""", True)
  PDFS = st.file_uploader("ارفع ملفاتك من هنا", type="pdf", accept_multiple_files=True)
  if PDFS:
    with st.spinner("⏳ جاري المعالجة... "):
      #استخراج الكلام من الملفات ال PDF
      GET_TEXT = GET_TEXT_FROM_PDF(PDFS)

      st.session_state.context = GET_TEXT

      # تقسيم الكلمات علي شكل مقاطع
      SPLIT_TEXT_TO_CHUNK = SPLITTEXTTOCHUNK(GET_TEXT)
      for PDF in PDFS:
        filename = os.path.splitext(PDF.name)[0]
        STORE = CREATESTORE(SPLIT_TEXT_TO_CHUNK, filename)
      # انشاء قاعدة بيانات

    user_question = st.chat_input("أسال سؤالك هنا")
    if (user_question):

        answer = ASK_PDF_QUESTION(STORE, user_question)
        st.session_state.chat_history.append({
            "question": user_question ,
            "answer": answer
        })
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
              # 🔹 يحول أي لينك نصي إلى رابط HTML قابل للضغط
              def make_links_clickable(text):
                # يحوّل أي لينك (حتى اللي من غير http) إلى رابط قابل للضغط
                url_pattern = r'((?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?)'
                def repl(match):
                  url = match.group(0)
                  if not url.startswith("http"):
                    url = "https://" + url
                  return f'<a href="{url}" target="_blank" style="color:#4fc3f7; text-decoration:underline;">{match.group(0)}</a> 🔗'
                return re.sub(url_pattern, repl, text)


              st.markdown(f"""
        <div style='background-color:rgb(255 255 255 / 4%); backdrop-filter: blur(10px); font-size:18px ;  color:#FFF; direction: rtl ; font-family:tajawal ;padding:10px; border-radius:10px; margin-top:10px;'>
            <span style="color: #F05; margin-bottom: 6px; display:inline-block; font-weight:bold"> 🙋‍♂️ سؤالك</span><br>{chat["question"]}
                    </div>
  """, unsafe_allow_html=True)
              st.markdown(f"""
            <div style='background-color:#FFF1; font-size:18px ;color:#FFF; direction: rtl ;  font-family:tajawal ;padding:10px; border-radius:10px; margin-top:10px;'>
                        <span style='color:#4CAF50; margin-bottom: 6px; display:inline-block; font-weight:bold'>🤖 المستشار القانوني</span><br>{make_links_clickable(chat["answer"])}
                        </div>
  """, unsafe_allow_html=True)



if __name__ == "__main__":
  main()