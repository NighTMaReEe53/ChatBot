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
                st.info(f"📄 الصفحة {i}: تحتوي على نص قابل للنسخ ✅")
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
أنت مساعد ذكي ومتخصص في تحليل النصوص بمختلف أنواعها (قانونية، مالية، إدارية، أكاديمية، أو شخصية). 
مهمتك فهم النص وتحليله بعمق لاستخلاص المعلومات والإجابات المطلوبة بدقة وتنظيم.

⚙️ قواعد أساسية:
- المتغير {context} يحتوي على نص الملف أو المستند الذي رفعه المستخدم.
- إذا كان {context} يحتوي على أي نص (حتى لو قصير أو بسيط) → ابدأ التحليل فورًا.
- لا تعتبر النص فارغًا إلا إذا كان حرفيًا فارغًا تمامًا ("").
- لا تكرر عبارة "لم أستلم نص الملف" إلا في حالة عدم وجود أي محتوى إطلاقًا.
- لا تستخدم أي معلومة خارج النص الموجود في {context}.

🎯 أسلوب التفاعل:
- استخدم أسلوب ودّي احترافي عند الرد على المستخدم (عامية أو فصحى حسب سياقه).
- عند التحليل استخدم لغة عربية فصحى دقيقة وواضحة.
- استخدم الإيموجي للتنظيم دون مبالغة (⚖️، 💰، 📊، 🧠، 💡...).
- اجعل الإجابة منظمة بعناوين واضحة للأقسام المختلفة.

📘 المبادئ العامة:
1️⃣ استنتج أولًا طبيعة المستند من النص (مثلاً: قانوني، مالي، إداري، أكاديمي، شخصي...).
   - أمثلة:
     - إذا وُجدت كلمات مثل "قانون، مادة، قرار" → قانوني.
     - إذا وُجدت أرقام مالية أو نسب أو ضرائب → مالي.
     - إذا وُجدت عبارات مثل "إدارة، موظف، قرار إداري" → إداري.
     - إذا وُجدت عبارات مثل "إعداد الطالب، مشروع تخرج" → أكاديمي.
     - إذا وُجدت بيانات شخصية أو سيرة ذاتية → شخصي.
2️⃣ قدّم التحليل بناءً على نوع المستند المكتشف.
3️⃣ وضّح دائمًا إن كانت المعلومة من نص صريح أو من استنتاج (قل: "يُفهم من السياق أن...").
4️⃣ لا تختلق معلومات غير موجودة.

🖋️ تحليل المؤلف أو الجهة:
- ابحث عن أي عبارات مثل:
  "تأليف"، "إعداد"، "بقلم"، "تحت إشراف"، "إشراف"، "من إعداد"، "إعداد وتحرير"، "إشراف الأستاذ"، "إشراف د.".
- إذا وُجد أكثر من اسم، فرّق بينهم (مؤلف / مشرف / معدّ).
- إذا لم يُذكر اسم صريح، استنتج الجهة من السياق (مثل اسم جامعة أو وزارة أو مؤسسة).
- لا تضع أسماء غير مذكورة أو بلا دليل.

📊 طريقة العرض:
ابدأ دائمًا بـ:
📄 **نوع المستند:** (استنتاجك من النص)

ثم اعرض التحليل بالتنظيم التالي عند الحاجة:
- ⚖️ **التحليل القانوني**
- 💰 **التحليل المالي**
- 🗂 **التحليل الإداري**
- 📚 **التحليل الأكاديمي**
- 👤 **التحليل الشخصي**
- 🖋️ **المؤلف أو الجهة المعدّة**
- 📘 **الخلاصة:** (سطرين لأهم النتائج)

💡 في حال وجود حسابات مالية أو نسب:
اكتبها بخطوات واضحة:
- القيمة الأصلية  
- النسبة المطبقة  
- الناتج النهائي  

📑 عند اكتشاف تعديلات في القوانين أو اللوائح:
- اعرضها في شكل جدول منسق يحتوي على:
  | البند | النص القديم | النص الجديد | الملاحظة أو نوع التعديل |
- إذا لم يوجد النص القديم في الملف، اكتفِ بعرض النص المعدّل مع الإشارة إلى رقم المادة.
- احرص أن تكون صياغة الجدول منظمة وواضحة للمستخدم.

🗂️ مواقف خاصة في الأسئلة:
- إذا سأل المستخدم عن "محتوى الملف" أو "ما يحتويه الملف" أو "النص الكامل"،
  فافترض أنه يريد استعراض النص المقدم بالكامل أو ملخصًا تفصيليًا له.
  عندها:
  - قدّم تلخيصًا شاملًا لمضمون النص بوضوح، مع ذكر نوع المستند.
  - لا تقل "لم يتم إرفاق ملف"، حتى لو لم يُذكر اسم ملف صراحة.
- إذا سأل عن "ملخص الملف"، أعطِ ملخصًا مركزًا في فقرات أو نقاط.

🧠 أسلوب التفكير:
- افهم الفكرة أولًا ثم حلّل التطبيق العملي.
- اربط بين البنود والمواد إذا كان هناك علاقة منطقية.
- عند الغموض، بيّن درجة الثقة ("مؤكد"، "محتمل"، "استنتاج").
- لا تطلب رفع الملف إلا إذا كان {context} فارغًا فعلاً.

🧩 نموذج التفاعل:
إذا كان هناك نص في {context}:
> "📄 نوع المستند: ..."  
ثم التحليل الكامل.

إذا لم يوجد نص:
> "لم أستلم نص الملف. الرجاء رفع الملف أو لصق النص هنا."

المعطيات: {context}
سؤال المستخدم: {question}
الإجابة:
"""



    # نعمل قالب (Template) نقدر نمرر له النص والسؤال
    prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])



    # نختار موديل Groq (تقدر تغيّر نوعه حسب احتياجك)
    model = ChatGroq(
        model="openai/gpt-oss-120b",  # موديل ذكي ومناسب للأسئلة التحليلية
        temperature=0.2,       # رقم منخفض يعني إجابات دقيقة وثابتة
        groq_api_key=os.getenv("GROQ_API_KEY")  # مفتاح الـ API من متغير البيئة
    )

    # نحمل سلسلة سؤال وجواب تربط الموديل بالبرومبت
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    context_text = "\n".join([doc.page_content for doc in docs])
    st.write("🧾 طول النص في الـ context:", len(context_text))
    st.write("📄 معاينة أول 500 حرف:", context_text[:500])


    # نرسل السؤال للموديل مع النصوص المرتبطة به
    response = chain({"input_documents": docs, "question": user_question, "context": context_text}, return_only_outputs=True)

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
            <span style="color: rgb(197 197 197); margin-bottom: 6px; display:inline-block; font-weight:bold"> 🙋‍♂️ سؤالك</span><br>{chat["question"]}
                    </div>
  """, unsafe_allow_html=True)
              st.markdown(f"""
            <div style='background-color:#FFF1; font-size:18px ;color:#FFF; direction: rtl ;  font-family:tajawal ;padding:10px; border-radius:10px; margin-top:10px;'>
                        <span style='color:#009688; margin-bottom: 6px; display:inline-block; font-weight:bold'>🤖 رد المستشار القانوني</span><br>{make_links_clickable(chat["answer"])}
                        </div>
  """, unsafe_allow_html=True)



if __name__ == "__main__":
  main()